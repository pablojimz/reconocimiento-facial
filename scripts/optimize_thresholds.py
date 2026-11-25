"""Optimize 1:1 and 1:N thresholds and validate them on held-out identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reconocimiento_facial.paths import CALIBRATION_OUTPUT

THRESHOLDS = np.round(np.arange(-1.0, 1.0005, 0.001), 3)


def verification_scores(
    gallery: np.ndarray,
    probes: np.ndarray,
    probe_ids: np.ndarray,
    identities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute genuine and impostor scores within the selected identities."""
    selected = np.isin(probe_ids, identities)
    references = gallery[identities]
    scores = probes[selected] @ references.T
    own = probe_ids[selected][:, None] == identities[None, :]
    between = references @ references.T
    upper = np.triu_indices(len(identities), k=1)
    impostor = np.concatenate([scores[~own], between[upper]])
    return scores[own], impostor


def verification_cost(
    genuine: np.ndarray, impostor: np.ndarray, far_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    far = (impostor[None, :] >= THRESHOLDS[:, None]).mean(axis=1)
    frr = (genuine[None, :] < THRESHOLDS[:, None]).mean(axis=1)
    return far_weight * far + (1 - far_weight) * frr, far, frr


def identification_outcomes(
    gallery: np.ndarray,
    probes: np.ndarray,
    probe_ids: np.ndarray,
    identities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top scores with and without the probe identity."""
    selected = np.isin(probe_ids, identities)
    scores = probes[selected] @ gallery[identities].T
    own = probe_ids[selected][:, None] == identities[None, :]
    known_correct = identities[scores.argmax(axis=1)] == probe_ids[selected]
    known_best = scores.max(axis=1)
    between = gallery[identities] @ gallery[identities].T
    np.fill_diagonal(between, -np.inf)
    unknown_best = np.concatenate(
        [np.where(own, -np.inf, scores).max(axis=1), between.max(axis=1)]
    )
    return known_correct, known_best, unknown_best


def rank1_rate(
    known_correct: np.ndarray, known_best: np.ndarray, unknown_best: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute open-set rank-1 rates across thresholds."""
    hits_known = (
        known_correct[None, :] & (known_best[None, :] >= THRESHOLDS[:, None])
    ).sum(axis=1)
    hits_unknown = (unknown_best[None, :] < THRESHOLDS[:, None]).sum(axis=1)
    total = known_best.size + unknown_best.size
    return (
        (hits_known + hits_unknown) / total,
        hits_known / known_best.size,
        1 - hits_unknown / unknown_best.size,
    )


def best_threshold(
    values: np.ndarray, maximize: bool
) -> tuple[float, float, float]:
    """Return the midpoint and endpoints of the widest optimal interval."""
    target = values.max() if maximize else values.min()
    optimal = np.isclose(values, target, atol=1e-12)
    best_start = best_length = start = 0
    for index, flag in enumerate(np.append(optimal, False)):
        if flag and (index == 0 or not optimal[index - 1]):
            start = index
        if (
            not flag
            and index > 0
            and optimal[index - 1]
            and index - start > best_length
        ):
            best_start, best_length = start, index - start
    low, high = (
        THRESHOLDS[best_start],
        THRESHOLDS[best_start + best_length - 1],
    )
    return float(round((low + high) / 2, 3)), float(low), float(high)


def calibrate(
    gallery: np.ndarray,
    probes: np.ndarray,
    probe_ids: np.ndarray,
    identities: np.ndarray,
    far_weight: float,
) -> tuple[float, float]:
    genuine, impostor = verification_scores(
        gallery, probes, probe_ids, identities
    )
    cost, _, _ = verification_cost(genuine, impostor, far_weight)
    rate, _, _ = rank1_rate(
        *identification_outcomes(gallery, probes, probe_ids, identities)
    )
    return best_threshold(cost, maximize=False)[0], best_threshold(
        rate, maximize=True
    )[0]


def evaluate(
    gallery: np.ndarray,
    probes: np.ndarray,
    probe_ids: np.ndarray,
    identities: np.ndarray,
    verification: float,
    identification: float,
) -> dict[str, float]:
    genuine, impostor = verification_scores(
        gallery, probes, probe_ids, identities
    )
    far = float((impostor >= verification).mean())
    frr = float((genuine < verification).mean())
    known_correct, known_best, unknown_best = identification_outcomes(
        gallery, probes, probe_ids, identities
    )
    hits_known = known_correct & (known_best >= identification)
    hits_unknown = unknown_best < identification
    return {
        "far": far,
        "frr": frr,
        "rank1": float(
            (hits_known.sum() + hits_unknown.sum())
            / (hits_known.size + hits_unknown.size)
        ),
        "enrolled_rank1": float(hits_known.mean()),
        "unenrolled_false_alarm": float(1 - hits_unknown.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--templates", type=Path, default=CALIBRATION_OUTPUT / "templates.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=CALIBRATION_OUTPUT / "thresholds.json"
    )
    parser.add_argument(
        "--far-weight",
        type=float,
        default=0.5,
        help="FAR cost weight. The default of 0.5 computes HTER.",
    )
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.far_weight <= 1:
        parser.error("--far-weight must be between 0 and 1.")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive.")
    if not args.templates.is_file():
        raise SystemExit(
            f"Missing {args.templates}. Run make calibrate first."
        )
    with np.load(args.templates, allow_pickle=False) as data:
        gallery = data["gallery"]
        probes = data["probes"]
        probe_ids = data["probe_ids"]
    if len(gallery) < 4 or set(probe_ids) != set(range(len(gallery))):
        parser.error("Expected four or more identities with probes for each.")
    everyone = np.arange(len(gallery))

    genuine, impostor = verification_scores(
        gallery, probes, probe_ids, everyone
    )
    cost, far, frr = verification_cost(genuine, impostor, args.far_weight)
    verification, v_low, v_high = best_threshold(cost, maximize=False)
    eer = int(np.argmin(np.abs(far - frr)))
    rate, _, _ = rank1_rate(
        *identification_outcomes(gallery, probes, probe_ids, everyone)
    )
    identification, i_low, i_high = best_threshold(rate, maximize=True)

    # Calibrate on half of the identities and evaluate on the other half.
    rng = np.random.default_rng(0)
    held_out = []
    for _ in range(args.repetitions):
        order = rng.permutation(everyone)
        train, test = (
            np.sort(order[: len(order) // 2]),
            np.sort(order[len(order) // 2 :]),
        )
        thresholds = calibrate(
            gallery, probes, probe_ids, train, args.far_weight
        )
        held_out.append(
            (
                *thresholds,
                evaluate(gallery, probes, probe_ids, test, *thresholds),
            )
        )
    metrics = {
        key: [result[key] for _, _, result in held_out]
        for key in held_out[0][2]
    }

    full = evaluate(
        gallery, probes, probe_ids, everyone, verification, identification
    )
    report = {
        "verification": {
            "criterion": (
                f"min {args.far_weight}·FAR + {1 - args.far_weight:.2g}·FRR"
            ),
            "threshold": verification,
            "optimal_interval": [v_low, v_high],
            "cost": float(cost.min()),
            "far": full["far"],
            "frr": full["frr"],
            "eer": {
                "threshold": float(THRESHOLDS[eer]),
                "value": float((far[eer] + frr[eer]) / 2),
            },
        },
        "identification": {
            "criterion": "maximize open-set rank-1",
            "threshold": identification,
            "optimal_interval": [i_low, i_high],
            "rank1": full["rank1"],
            "enrolled_rank1": full["enrolled_rank1"],
            "unenrolled_false_alarm": full["unenrolled_false_alarm"],
            "closed_set_rank1": float(
                identification_outcomes(gallery, probes, probe_ids, everyone)[
                    0
                ].mean()
            ),
        },
        "validation_by_identity": {
            "repetitions": args.repetitions,
            "verification_threshold": [
                min(v for v, _, _ in held_out),
                max(v for v, _, _ in held_out),
            ],
            "identification_threshold": [
                min(i for _, i, _ in held_out),
                max(i for _, i, _ in held_out),
            ],
            **{
                f"{key}_mean": float(np.mean(values))
                for key, values in metrics.items()
            },
            **{
                f"{key}_worst": float(
                    max(values)
                    if key in ("far", "frr", "unenrolled_false_alarm")
                    else min(values)
                )
                for key, values in metrics.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
