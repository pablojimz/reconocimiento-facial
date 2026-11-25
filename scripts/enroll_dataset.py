"""Enroll dataset images, using the most confident face in group photos."""

from __future__ import annotations

import argparse
from pathlib import Path

from reconocimiento_facial.database import UserDatabase
from reconocimiento_facial.paths import ENROLLMENT_DATASET, USER_DATABASE
from reconocimiento_facial.system import SistemaReconocimientoFacial

from .probes import identity, image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ENROLLMENT_DATASET)
    parser.add_argument("--database", type=Path, default=USER_DATABASE)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace existing user records with templates from the dataset.",
    )
    args = parser.parse_args()

    database = UserDatabase(args.database)
    system = SistemaReconocimientoFacial(database=database)
    existing = {user.user_id for user in database.list_users()}
    enrolled = groups = skipped = failed = 0
    for path in image_paths(args.dataset):
        user_id = identity(path)
        if user_id in existing and not args.reset:
            skipped += 1
            print(f"= {user_id}: already enrolled")
            continue
        try:
            image = system._load_image(path)
            faces = system.detectar(image)
            if not faces:
                raise ValueError("no face detected")
            face = max(faces, key=lambda detection: detection.confidence)
            aligned = system.alinear(image, face.landmarks)
            template = system.extraer(aligned)
            source = path if len(faces) == 1 else aligned
            system._enroll_with_source(
                user_id, source, template, replace_existing=args.reset
            )
        except (OSError, ValueError) as error:
            failed += 1
            print(f"! {user_id}: {error}")
            continue

        existing.add(user_id)
        enrolled += 1
        if len(faces) == 1:
            print(f"+ {user_id}")
        else:
            groups += 1
            print(
                f"+ {user_id}: enrolled the best of {len(faces)} faces "
                f"({face.confidence:.2f})"
            )

    print(
        f"\n{enrolled} enrolled ({groups} with multiple faces), "
        f"{skipped} already enrolled, {failed} failed. "
        f"Database: {args.database}"
    )


if __name__ == "__main__":
    main()
