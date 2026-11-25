"""Evaluate open-set and closed-set identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
        "--threshold",
        type=float,
        help="Override the identification threshold.",
    )
    parser.add_argument("--cmc-rank", type=int, default=5)
    args = parser.parse_args()
    if args.cmc_rank < 1:
        parser.error("--cmc-rank must be positive.")

    database = UserDatabase(args.database)
    settings = (
        {}
        if args.threshold is None
        else {"identification_threshold": args.threshold}
    )
    system = SistemaReconocimientoFacial(database=database, **settings)
    users = [user for user in database.list_users() if user.template]
    if not users:
        raise SystemExit(
            f"No enrolled users in {args.database}. Run make enroll."
        )
    names = [user.user_id for user in users]
    gallery = np.array([user.template for user in users])
    cache = args.output / "probes.npz"
    probes = load_probes(system, args.dataset, args.probes, cache)

    name_indices = {name: index for index, name in enumerate(names)}
    known_total = 0
    hits = 0
    misses: list[dict[str, str | None]] = []
    ranks: list[int] = []
    false_alarms: list[tuple[str, dict[str, str]]] = []

    for probe in probes:
        scores = (
            system._gallery_scores(probe.template, users)
            if probe.template is not None
            else {}
        )
        if probe.identity in name_indices:
            known_total += 1
            best = max(scores, key=scores.__getitem__, default=None)
            answer = (
                best
                if best is not None
                and scores[best] >= system.identification_threshold
                else None
            )
            if answer == probe.identity:
                hits += 1
            else:
                misses.append({"probe": probe.source, "answer": answer})
            if probe.template is not None:
                ranks_for_probe = gallery @ probe.template
                own_score = ranks_for_probe[name_indices[probe.identity]]
                ranks.append(int((ranks_for_probe > own_score).sum()) + 1)

        # Exclude the probe identity to simulate an unenrolled user.
        scores.pop(probe.identity, None)
        best = max(scores, key=scores.__getitem__, default=None)
        if (
            best is not None
            and scores[best] >= system.identification_threshold
        ):
            false_alarms.append(
                (probe.identity, {"probe": probe.source, "answer": best})
            )

    unknown_total = len(probes)
    unknown_hits = unknown_total - len(false_alarms)
    false_alarm_details = [
        row for _, row in sorted(false_alarms, key=lambda item: item[0])
    ]
    ranks_array = np.array(ranks)
    cmc = {
        f"rank_{k}": float((ranks_array <= k).sum() / max(known_total, 1))
        for k in range(1, args.cmc_rank + 1)
    }
    dir_rate = hits / max(known_total, 1)
    fpir = len(false_alarms) / max(unknown_total, 1)
    open_rank1 = (hits + unknown_hits) / max(known_total + unknown_total, 1)
    report = {
        "identification_threshold": system.identification_threshold,
        "enrolled_users": len(names),
        "enrolled_probes": known_total,
        "unenrolled_probes": unknown_total,
        "capture_failures": sum(p.template is None for p in probes),
        "open_set_rank1": open_rank1,
        "DIR": dir_rate,
        "FNIR": 1 - dir_rate,
        "FPIR": fpir,
        "closed_set_rank1": cmc["rank_1"],
        "CMC": cmc,
        "enrolled_misses": misses[:20],
        "false_alarms": false_alarm_details[:20],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "identification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(
        f"\nIdentification threshold {system.identification_threshold}: "
        f"{len(names)} enrolled, {known_total} enrolled probes, "
        f"{unknown_total} unenrolled probes\n"
        f"  Open-set rank-1 = {open_rank1:.4%}\n"
        f"  DIR  = {dir_rate:.4%}   FNIR = {1 - dir_rate:.4%}\n"
        f"  FPIR = {fpir:.4%} ({len(false_alarms)} false alarms)\n"
        f"  Closed-set rank-1 = {cmc['rank_1']:.4%}   "
        f"Rank-{args.cmc_rank} = {cmc[f'rank_{args.cmc_rank}']:.4%}\n"
        f"Report: {args.output / 'identification.json'}"
    )


if __name__ == "__main__":
    main()
