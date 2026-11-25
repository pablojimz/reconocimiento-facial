"""Default paths relative to the working directory."""

from pathlib import Path

DATABASE_DIR = Path("database")
USER_DATABASE = DATABASE_DIR / "registration/users.csv"
ENROLLMENT_DATASET = DATABASE_DIR / "enrollment"
EVALUATION_DATASET = DATABASE_DIR / "evaluation"

OUTPUT_DIR = Path("outputs")
EVALUATION_OUTPUT = OUTPUT_DIR / "evaluation"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration"
