"""Calibrate thresholds using synthetic genuine pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from reconocimiento_facial.paths import CALIBRATION_OUTPUT, ENROLLMENT_DATASET
from reconocimiento_facial.system import SistemaReconocimientoFacial

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _rotate(
    image: np.ndarray, degrees: float, scale: float = 1.0
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, scale)
    return cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )


def _gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def _low_resolution(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, int(width * factor)), max(1, int(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    _, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _occlude_mouth(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Occlude the lower 40 percent of the face box."""
    x, y, width, height = bbox
    occluded = image.copy()
    cv2.rectangle(
        occluded,
        (x, y + int(height * 0.6)),
        (x + width, y + height),
        (90, 90, 90),
        thickness=-1,
    )
    return occluded


def variations(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield synthetic genuine probes one at a time to limit memory use."""
    yield "mirror", cv2.flip(image, 1)
    yield "rotation_+15", _rotate(image, 15)
    yield "rotation_-15", _rotate(image, -15)
    yield "zoom_0.8", _rotate(image, 0, 0.8)
    yield "blur", cv2.GaussianBlur(image, (0, 0), sigmaX=2.5)
    yield "low_resolution", _low_resolution(image, 0.25)
    yield "dark", _gamma(image, 2.2)
    yield "overexposed", _gamma(image, 0.45)
    yield "jpeg_q15", _jpeg(image, 15)
    yield (
        "grayscale",
        cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        ),
    )
    yield "face_mask", _occlude_mouth(image, bbox)
    yield (
        "combined",
        _jpeg(
            _gamma(_rotate(cv2.GaussianBlur(image, (0, 0), 1.5), -8), 1.6), 30
        ),
    )


def primary_template(
    system: SistemaReconocimientoFacial, image: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return the template and box of the most confident face, if any."""
    detections = system.detectar(image)
    if not detections:
        return None
    face = max(detections, key=lambda d: (d.confidence, d.bbox[2] * d.bbox[3]))
    return system.extraer(system.alinear(image, face.landmarks)), face.bbox


def far_frr(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    return far, frr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ENROLLMENT_DATASET)
    parser.add_argument("--output", type=Path, default=CALIBRATION_OUTPUT)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margin above the highest observed impostor score.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    system = SistemaReconocimientoFacial()

    names, references, probes = (
        [],
        [],
        [],
    )  # probes: (identity index, variation, template)
    skipped = []
    for path in sorted(
        p for p in args.dataset.iterdir() if p.suffix.lower() in EXTENSIONS
    ):
        image = cv2.imread(str(path))
        result = primary_template(system, image) if image is not None else None
        if result is None:
            skipped.append(path.name)
            continue
        template, bbox = result
        identity = len(names)
        names.append(path.stem)
        references.append(template)
        for variation, altered in variations(image, bbox):
            probe = primary_template(system, altered)
            if probe is None:
                skipped.append(f"{path.name}:{variation}")
                continue
            probes.append((identity, variation, probe[0]))
        print(f"[{identity + 1:2d}] {path.stem}")

    if len(references) < 2 or not probes:
        parser.error(
            "Calibration requires at least two identities and one valid probe."
        )

    gallery = np.stack(references)
    probe_ids = np.array([identity for identity, _, _ in probes])
    probe_kinds = [variation for _, variation, _ in probes]
    probe_matrix = np.stack([template for _, _, template in probes])
    np.savez_compressed(
        args.output / "templates.npz",
        names=names,
        gallery=gallery,
        probe_ids=probe_ids,
        probe_kinds=probe_kinds,
        probes=probe_matrix,
    )

    # One-to-one verification.
    gallery_scores = gallery @ gallery.T
    upper = np.triu_indices(len(names), k=1)
    impostor = gallery_scores[upper]
    probe_scores = probe_matrix @ gallery.T
    genuine = probe_scores[np.arange(len(probes)), probe_ids]
    mask = np.ones_like(probe_scores, dtype=bool)
    mask[np.arange(len(probes)), probe_ids] = False
    impostor_probes = probe_scores[mask]
    impostor_all = np.concatenate([impostor, impostor_probes])

    thresholds = np.round(np.arange(-0.2, 1.0001, 0.005), 3)
    far, frr = far_frr(genuine, impostor_all, thresholds)
    eer_index = int(np.argmin(np.abs(far - frr)))
    verification = float(
        np.ceil((impostor_all.max() + args.margin) * 100) / 100
    )

    # Test with and without the probe identity in the gallery.
    best_index = probe_scores.argmax(axis=1)
    best_score = probe_scores.max(axis=1)
    rank1 = float((best_index == probe_ids).mean())
    best_without_self = np.where(mask, probe_scores, -np.inf).max(axis=1)
    best_gallery_impostor = np.where(
        np.eye(len(names), dtype=bool), -np.inf, gallery_scores
    ).max(axis=1)
    open_set_impostor = np.concatenate(
        [best_without_self, best_gallery_impostor]
    )
    identification = float(
        np.ceil((open_set_impostor.max() + args.margin) * 100) / 100
    )

    def identification_rates(threshold: float) -> tuple[float, float]:
        correct = (
            (best_index == probe_ids) & (best_score >= threshold)
        ).mean()
        false_alarm = (open_set_impostor >= threshold).mean()
        return float(correct), float(false_alarm)

    hardest = np.argsort(impostor)[::-1][:5]
    pairs = [
        (names[upper[0][i]], names[upper[1][i]], float(impostor[i]))
        for i in hardest
    ]
    per_kind = {
        kind: float(
            np.mean([g for g, k in zip(genuine, probe_kinds) if k == kind])
        )
        for kind in dict.fromkeys(probe_kinds)
    }

    v_far, v_frr = far_frr(genuine, impostor_all, np.array([verification]))
    i_dir, i_far = identification_rates(identification)
    report = {
        "identities": len(names),
        "skipped": skipped,
        "impostor_pairs": int(impostor_all.size),
        "synthetic_genuine_pairs": int(genuine.size),
        "impostor": {
            "mean": float(impostor_all.mean()),
            "std": float(impostor_all.std()),
            "max": float(impostor_all.max()),
            "p99.9": float(np.percentile(impostor_all, 99.9)),
        },
        "genuine": {
            "mean": float(genuine.mean()),
            "std": float(genuine.std()),
            "min": float(genuine.min()),
            "p1": float(np.percentile(genuine, 1)),
            "mean_by_variation": per_kind,
        },
        "eer": {
            "threshold": float(thresholds[eer_index]),
            "far": float(far[eer_index]),
            "frr": float(frr[eer_index]),
        },
        "verification_threshold": verification,
        "verification_at_threshold": {
            "far": float(v_far[0]),
            "frr": float(v_frr[0]),
        },
        "identification_threshold": identification,
        "identification_rank1": rank1,
        "identification_at_threshold": {
            "hit_rate": i_dir,
            "false_alarm": i_far,
        },
        "hardest_impostor_pairs": pairs,
    }
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    np.savetxt(
        args.output / "far_frr_curve.csv",
        np.column_stack([thresholds, far, frr]),
        delimiter=",",
        header="threshold,far,frr",
        comments="",
        fmt="%.4f",
    )
    _plot(
        args.output,
        genuine,
        impostor_all,
        thresholds,
        far,
        frr,
        verification,
        identification,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _plot(
    output: Path,
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
    verification: float,
    identification: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install the calibration extra to generate plots.")
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(-0.3, 1.0, 66)
    left.hist(impostor, bins=bins, alpha=0.6, density=True, label="Impostors")
    left.hist(
        genuine,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Synthetic genuine pairs",
    )
    for value, label, style in (
        (verification, "verification", "--"),
        (identification, "identification", ":"),
    ):
        left.axvline(
            value,
            color="black",
            linestyle=style,
            label=f"Threshold {label} = {value:.2f}",
        )
    left.set_xlabel("Cosine similarity")
    left.set_ylabel("Density")
    left.legend()
    right.plot(thresholds, far, label="FAR")
    right.plot(thresholds, frr, label="FRR")
    right.axvline(verification, color="black", linestyle="--")
    right.set_xlabel("Threshold")
    right.set_ylabel("Rate")
    right.legend()
    figure.tight_layout()
    figure.savefig(output / "distributions.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
