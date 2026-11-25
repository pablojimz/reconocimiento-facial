"""Evaluate control and stress-test images against the reference gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypedDict

import numpy as np
import torch

from reconocimiento_facial import system as system_module
from reconocimiento_facial.database import UserDatabase, UserRecord
from reconocimiento_facial.detectors import FaceDetection
from reconocimiento_facial.models import RetinaFaceAssets, TransFaceAssets
from reconocimiento_facial.paths import EVALUATION_DATASET, EVALUATION_OUTPUT
from reconocimiento_facial.system import SistemaReconocimientoFacial

Policy = Literal[
    "recognize", "reject_unknown", "reject_representation", "reject_group"
]


class Trial(TypedDict):
    image: str
    group: Literal["controls", "cases"]
    category: str
    expected_identity: str | None
    policy: Policy
    claimed_identities: list[str]


class Score(TypedDict):
    identity: str
    similarity: float


class Result(TypedDict):
    trial: Trial
    image_sha256: str | None
    dimensions: list[int] | None
    detections: list[dict[str, object]]
    scores: list[Score]
    identification: str | None
    verification: dict[str, bool]
    status: str
    access_decision: str
    passed: bool
    error: str | None


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def evaluate(
    system: SistemaReconocimientoFacial,
    trial: Trial,
    dataset: Path,
    users: list[UserRecord],
) -> Result:
    path = dataset / trial["image"]
    result: Result = {
        "trial": trial,
        "image_sha256": None,
        "dimensions": None,
        "detections": [],
        "scores": [],
        "identification": None,
        "verification": {},
        "status": "",
        "access_decision": "error",
        "passed": False,
        "error": None,
    }
    stage = "load"
    try:
        result["image_sha256"] = sha256(path)
        image = system._load_image(path)
        result["dimensions"] = [image.shape[1], image.shape[0]]
        stage = "detection"
        faces: list[FaceDetection] = system.detectar(image)
        result["detections"] = [face.as_dict() for face in faces]
        if len(faces) != 1:
            result["status"] = "no_face" if not faces else "multiple_faces"
            result["access_decision"] = "capture_rejection"
            result["error"] = f"Expected one face, detected {len(faces)}."
            result["passed"] = trial["policy"] != "recognize"
            return result

        stage = "alignment"
        aligned = system.alinear(image, faces[0].landmarks)
        stage = "extraction"
        template = system.extraer(aligned)
        stage = "comparison"
        gallery_scores = system._gallery_scores(template, users)
        scores: list[Score] = [
            {"identity": identity, "similarity": similarity}
            for identity, similarity in gallery_scores.items()
        ]
        scores.sort(key=lambda score: score["similarity"], reverse=True)
        best = scores[0]
        identification = None
        if best["similarity"] >= system.identification_threshold:
            identification = best["identity"]
        verification = {
            identity: score >= system.verification_threshold
            for identity, score in gallery_scores.items()
        }
        accepted = identification is not None or any(verification.values())
        if trial["policy"] == "recognize":
            expected = trial["expected_identity"]
            passed = expected == identification and verification.get(
                expected or "", False
            )
            if passed:
                status = "recognized"
            else:
                status = (
                    "false_identification"
                    if identification
                    else "genuine_rejection"
                )
        elif trial["policy"] == "reject_group":
            passed, status = False, "group_not_rejected"
        else:
            passed = not accepted
            status = "correct_rejection" if passed else "false_acceptance"

        result.update(
            scores=scores,
            identification=identification,
            verification=verification,
            status=status,
            passed=passed,
            access_decision="acceptance"
            if accepted
            else "threshold_rejection",
        )
    except (OSError, RuntimeError, ValueError) as error:
        result["status"] = f"error_{stage}"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def group_metrics(rows: list[Result]) -> dict[str, object]:
    genuine = [row for row in rows if row["trial"]["policy"] == "recognize"]
    comparable = [row for row in genuine if row["scores"]]
    hits = sum(
        row["identification"] == row["trial"]["expected_identity"]
        for row in genuine
    )
    verified = sum(
        row["verification"].get(row["trial"]["expected_identity"] or "", False)
        for row in comparable
    )
    false_accepts = sum(
        accepted
        for row in comparable
        for identity, accepted in row["verification"].items()
        if identity != row["trial"]["expected_identity"]
    )
    impostor_pairs = sum(len(row["verification"]) - 1 for row in comparable)
    return {
        "images": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "passed": sum(row["passed"] for row in rows),
        "genuine": len(genuine),
        "comparable_genuine": len(comparable),
        "correct_identifications": hits,
        "operational_dir": rate(hits, len(genuine)),
        "operational_fnir": rate(len(genuine) - hits, len(genuine)),
        "correct_verifications": verified,
        "operational_rejections": len(genuine) - verified,
        "operational_frr": rate(len(genuine) - verified, len(genuine)),
        "conditional_frr": rate(len(comparable) - verified, len(comparable)),
        "false_accepts_between_enrolled": false_accepts,
        "impostor_pairs": impostor_pairs,
        "conditional_far": rate(false_accepts, impostor_pairs),
        "capture_failures": sum(
            row["access_decision"] == "capture_rejection" for row in rows
        ),
        "technical_errors": sum(
            row["access_decision"] == "error" for row in rows
        ),
    }


def summarize(rows: list[Result]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for group in ("controls", "cases"):
        subset = [row for row in rows if row["trial"]["group"] == group]
        if subset:
            summary[group] = group_metrics(subset)

    cases = [row for row in rows if row["trial"]["group"] == "cases"]
    if cases:
        unknown = [
            row
            for row in cases
            if row["trial"]["category"] in {"DIV", "CIC", "TBA"}
        ]
        comparable = [row for row in unknown if row["scores"]]
        false_ids = sum(
            row["identification"] is not None for row in comparable
        )
        pairs = sum(len(row["verification"]) for row in comparable)
        false_accepts = sum(
            sum(row["verification"].values()) for row in comparable
        )
        summary["unknown_DIV_CIC_TBA"] = {
            "images": len(unknown),
            "comparable": len(comparable),
            "false_identifications": false_ids,
            "conditional_fpir": rate(false_ids, len(comparable)),
            "images_with_false_accepts": sum(
                any(row["verification"].values()) for row in comparable
            ),
            "verification_pairs": pairs,
            "conditional_far": rate(false_accepts, pairs),
        }
        summary["impersonation"] = {
            category: {
                "images": len(subset),
                "false_accepts": sum(
                    row["access_decision"] == "acceptance" for row in subset
                ),
            }
            for category in ("GEM", "IA", "V3D")
            if (
                subset := [
                    row
                    for row in cases
                    if row["trial"]["category"] == category
                ]
            )
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EVALUATION_DATASET)
    parser.add_argument("--output", type=Path, default=EVALUATION_OUTPUT)
    parser.add_argument(
        "--group", choices=("all", "controls", "cases"), default="all"
    )
    args = parser.parse_args()

    manifest = args.dataset / "trials.json"
    trials: list[Trial] = json.loads(manifest.read_text(encoding="utf-8"))
    selected = [
        trial for trial in trials if args.group in ("all", trial["group"])
    ]
    if not selected or len({trial["image"] for trial in selected}) != len(
        selected
    ):
        parser.error("The selected trials must contain unique images.")

    reference = args.dataset / "reference_templates.npz"
    with np.load(reference, allow_pickle=False) as saved:
        names = [re.sub(r"_\d{4}$", "", str(name)) for name in saved["names"]]
        gallery = saved["gallery"]
    if (
        not names
        or len(set(names)) != len(names)
        or gallery.shape != (len(names), 512)
    ):
        parser.error("Invalid reference gallery.")
    for trial in selected:
        if (
            trial["policy"] == "recognize"
            and trial["expected_identity"] not in names
        ):
            parser.error(f"Unknown expected identity in {trial['image']}.")

    users = [
        UserRecord(
            name, "enrolled", "", UserDatabase._validate_template(template)
        )
        for name, template in zip(names, gallery, strict=True)
    ]
    torch.set_num_threads(4)
    system = SistemaReconocimientoFacial()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "torch",
                "torchvision",
                "opencv-python",
                "timm",
            )
        },
        "device": str(system.detector._device),
        "threads": torch.get_num_threads(),
        "verification_threshold": system.verification_threshold,
        "identification_threshold": system.identification_threshold,
        "detector_threshold": system.detector.confidence_threshold,
        "reference_sha256": sha256(reference),
        "manifest_sha256": sha256(manifest),
        "weights": {
            str(assets.weights_path): sha256(assets.weights_path)
            for assets in (RetinaFaceAssets(), TransFaceAssets())
        },
        "code_sha256": {
            path.name: sha256(path)
            for path in (
                *sorted(Path(system_module.__file__).parent.glob("*.py")),
                Path(__file__),
            )
        },
        "enrolled": len(names),
    }
    results = []
    for trial in selected:
        result = evaluate(system, trial, args.dataset, users)
        results.append(result)
        print(trial["image"], result["status"], flush=True)

    filename = (
        "images.json" if args.group == "all" else f"images_{args.group}.json"
    )
    target = args.output / filename
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": metadata,
        "metrics": summarize(results),
        "results": results,
    }
    with TemporaryDirectory(dir=args.output) as temporary:
        staged = Path(temporary) / filename
        staged.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        staged.replace(target)
    print(f"Results written to {target}")


if __name__ == "__main__":
    main()
