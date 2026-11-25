"""Evaluate 1:1 verification against the enrolled gallery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reconocimiento_facial.database import UserDatabase
from reconocimiento_facial.paths import (
    ENROLLMENT_DATASET,
    EVALUATION_OUTPUT,
    USER_DATABASE,
)
from reconocimiento_facial.system import SistemaReconocimientoFacial

from .probes import load_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ENROLLMENT_DATASET)
    parser.add_argument("--database", type=Path, default=USER_DATABASE)
    parser.add_argument(
        "--probes",
        type=Path,
        help="Real probes. Defaults to synthetic variations.",
    )
    parser.add_argument("--output", type=Path, default=EVALUATION_OUTPUT)
    parser.add_argument(
        "--threshold", type=float, help="Override the verification threshold."
    )
    args = parser.parse_args()

    database = UserDatabase(args.database)
    settings = (
        {}
        if args.threshold is None
        else {"verification_threshold": args.threshold}
    )
    system = SistemaReconocimientoFacial(database=database, **settings)
    users = [user for user in database.list_users() if user.template]
    if not users:
        raise SystemExit(
            f"No enrolled users in {args.database}. Run make enroll."
        )
    enrolled = {user.user_id for user in users}
    cache = args.output / "probes.npz"
    probes = load_probes(system, args.dataset, args.probes, cache)

    genuine = {"accepted": 0, "total": 0}
    impostor = {"accepted": 0, "total": 0}
    capture_failures = 0
    rejected_by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    false_accepts, false_rejects = [], []
    for probe in probes:
        kind = probe.source.split(":")[-1] if ":" in probe.source else "real"
        if probe.template is None:
            capture_failures += 1
        scores = (
            system._gallery_scores(probe.template, users)
            if probe.template is not None
            else {}
        )
        if probe.identity in enrolled:
            score = scores.get(probe.identity)
            accepted = (
                score is not None and score >= system.verification_threshold
            )
            genuine["total"] += 1
            genuine["accepted"] += accepted
            rejected_by_kind[kind][0] += not accepted
            rejected_by_kind[kind][1] += 1
            if not accepted:
                false_rejects.append(
                    {
                        "probe": probe.source,
                        "similarity": round(score, 3)
                        if score is not None
                        else None,
                    }
                )
        if probe.template is None:
            impostor["total"] += len(enrolled) - (probe.identity in enrolled)
            continue
        for claimed, score in scores.items():
            if claimed == probe.identity:
                continue
            impostor["total"] += 1
            if score >= system.verification_threshold:
                impostor["accepted"] += 1
                false_accepts.append(
                    {
                        "probe": probe.source,
                        "claimed_identity": claimed,
                        "similarity": round(score, 3),
                    }
                )

    far = impostor["accepted"] / max(impostor["total"], 1)
    frr = 1 - genuine["accepted"] / max(genuine["total"], 1)
    report = {
        "verification_threshold": system.verification_threshold,
        "enrolled_users": len(enrolled),
        "probes": len(probes),
        "capture_failures": capture_failures,
        "genuine_attempts": genuine["total"],
        "impostor_attempts": impostor["total"],
        "FAR": far,
        "FRR": frr,
        "HTER": (far + frr) / 2,
        "accuracy": (
            genuine["accepted"] + impostor["total"] - impostor["accepted"]
        )
        / max(genuine["total"] + impostor["total"], 1),
        "FRR_by_variation": {
            kind: rejected / total
            for kind, (rejected, total) in rejected_by_kind.items()
        },
        "false_acceptances": sorted(
            false_accepts, key=lambda e: -e["similarity"]
        )[:20],
        "false_rejections": false_rejects[:20],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(
        f"\nVerification threshold {system.verification_threshold}: "
        f"{genuine['total']} genuine attempts, "
        f"{impostor['total']} impostor attempts, "
        f"{capture_failures} capture failures\n"
        f"  FAR  = {far:.4%} ({impostor['accepted']} accepted impostors)\n"
        f"  FRR  = {frr:.4%} "
        f"({genuine['total'] - genuine['accepted']} genuine rejections)\n"
        f"  HTER = {(far + frr) / 2:.4%}\n"
        f"Report: {args.output / 'verification.json'}"
    )


if __name__ == "__main__":
    main()
