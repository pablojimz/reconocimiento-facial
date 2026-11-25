"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
import warnings
from getpass import GetPassWarning, getpass
from pathlib import Path

import cv2

from .database import UserDatabase
from .detectors import FaceDetection
from .models import RetinaFaceAssets, TransFaceAssets
from .paths import USER_DATABASE
from .system import SistemaReconocimientoFacial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Facial recognition system")
    parser.add_argument("--database", type=Path, default=USER_DATABASE)
    parser.add_argument(
        "--ask-password",
        action="store_true",
        help="Prompt for the database password without echoing it",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="Detect faces in an image")
    detect.add_argument("image", type=Path)
    detect.add_argument("--output", type=Path, help="Save the annotated image")
    enroll = commands.add_parser(
        "enroll", help="Enroll a user with a TransFace-B template"
    )
    enroll.add_argument("user_id")
    enroll.add_argument("image", type=Path)
    users = commands.add_parser("users", help="Manage user records")
    user_commands = users.add_subparsers(dest="users_command", required=True)
    user_commands.add_parser("list")
    add = user_commands.add_parser("add")
    add.add_argument("user_id")
    remove = user_commands.add_parser("remove")
    remove.add_argument("user_id")
    database = commands.add_parser(
        "database", help="Manage database encryption"
    )
    database_commands = database.add_subparsers(
        dest="database_command", required=True
    )
    database_commands.add_parser(
        "migrate", help="Encrypt an existing CSV in place"
    )
    database_commands.add_parser(
        "change-password", help="Change the database password"
    )
    models = commands.add_parser("models", help="Manage model artifacts")
    model_commands = models.add_subparsers(
        dest="models_command", required=True
    )
    download = model_commands.add_parser(
        "download", help="Download code and pretrained weights"
    )
    download.add_argument(
        "--model",
        choices=("retinaface", "transface", "all"),
        default="retinaface",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        password = None
        if args.ask_password and args.command in (
            "users",
            "enroll",
            "database",
        ):
            password = _prompt_password("Database password: ")
        database = UserDatabase(args.database, password=password)
        if args.command == "models":
            if args.model in ("retinaface", "all"):
                print(f"Weights available at {RetinaFaceAssets().download()}.")
            if args.model in ("transface", "all"):
                print(f"Weights available at {TransFaceAssets().download()}.")
        elif args.command == "detect":
            system = SistemaReconocimientoFacial(database=database)
            faces = system.detectar(args.image)
            if args.output:
                _save_annotated_image(args.image, args.output, faces)
            print(f"Faces detected: {len(faces)}")
            for index, face in enumerate(faces, start=1):
                x, y, width, height = face.bbox
                print(
                    f"Face {index}: confidence {face.confidence:.2%} | "
                    f"box: x={x}, y={y}, width={width}, height={height}"
                )
                landmarks = ", ".join(
                    f"({px:.1f}, {py:.1f})" for px, py in face.landmarks
                )
                print(f"  Landmarks: {landmarks}")
            if args.output:
                print(f"Annotated image saved to {args.output}")
        elif args.command == "enroll":
            SistemaReconocimientoFacial(database=database).inscribir(
                args.user_id, args.image
            )
            print(f"User '{args.user_id}' enrolled with TransFace-B.")
        elif args.command == "database":
            if args.database_command == "migrate":
                if database.migrate_csv():
                    print(f"Encrypted database at {database.path}.")
                else:
                    print(f"Database is already encrypted at {database.path}.")
            elif args.database_command == "change-password":
                new_password = _prompt_password("New database password: ")
                if new_password != _prompt_password(
                    "Repeat the new password: "
                ):
                    raise ValueError("The passwords do not match.")
                database.change_password(new_password)
                print("Database password updated.")
        elif args.users_command == "list":
            rows = [
                (user.user_id, user.status, user.source_image or "—")
                for user in database.list_users()
            ]
            headers = ("User", "Status", "Image")
            widths = [
                max(len(row[index]) for row in [headers, *rows])
                for index in range(3)
            ]
            for row in [headers, *rows]:
                print(
                    "  ".join(
                        value.ljust(width) for value, width in zip(row, widths)
                    ).rstrip()
                )
        elif args.users_command == "add":
            database.add_user(args.user_id)
            print(f"User '{args.user_id}' enrolled as pending_template.")
        elif args.users_command == "remove":
            if not database.remove_user(args.user_id):
                raise ValueError(f"User '{args.user_id}' does not exist.")
            print(f"User '{args.user_id}' removed.")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _prompt_password(prompt: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", GetPassWarning)
        try:
            return getpass(prompt)
        except (EOFError, GetPassWarning, KeyboardInterrupt) as error:
            raise ValueError(
                "Password entry requires an interactive terminal."
            ) from error


def _save_annotated_image(
    image_path: Path, output_path: Path, faces: list[FaceDetection]
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode {image_path}")
    for face in faces:
        x, y, width, height = face.bbox
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{face.confidence:.3f}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        for landmark in face.landmarks:
            cv2.circle(image, tuple(map(round, landmark)), 3, (0, 0, 255), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not save {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
