"""Facial recognition API specified by the assignment."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .database import UserDatabase, UserRecord
from .detectors import FaceDetection, RetinaFaceR50Detector
from .extractors import TransFaceBExtractor

# Default thresholds from the documented synthetic calibration.
VERIFICATION_THRESHOLD = 0.406
IDENTIFICATION_THRESHOLD = 0.406


class SistemaReconocimientoFacial:
    """Coordinate face detection, feature extraction and identity storage."""

    def __init__(
        self,
        database: UserDatabase | None = None,
        detector: object | None = None,
        extractor: TransFaceBExtractor | None = None,
        verification_threshold: float = VERIFICATION_THRESHOLD,
        identification_threshold: float = IDENTIFICATION_THRESHOLD,
    ) -> None:
        for threshold in (verification_threshold, identification_threshold):
            if not -1.0 <= threshold <= 1.0:
                raise ValueError(
                    "The cosine similarity threshold must be in [-1, 1]."
                )
        self.db = database or UserDatabase()
        self.detector = detector or RetinaFaceR50Detector()
        self.extractor = extractor or TransFaceBExtractor()
        self.verification_threshold = verification_threshold
        self.identification_threshold = identification_threshold

    def detectar(self, photo: str | Path | np.ndarray) -> list[FaceDetection]:
        """Return all detections for a photo path or a BGR array."""
        return self.detector.detect(self._load_image(photo))

    def alinear(
        self, photo: str | Path | np.ndarray, landmarks: object
    ) -> np.ndarray:
        """Fit a similarity transform to five ArcFace landmarks."""
        image = self._load_image(photo)
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("Expected five finite two-dimensional landmarks.")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("The landmarks cannot be used to align the face.")
        destination = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float64,
        )
        # x' = a*x - b*y + tx; y' = b*x + a*y + ty.
        design = np.zeros((10, 4), dtype=np.float64)
        design[0::2, 0] = points[:, 0]
        design[0::2, 1] = -points[:, 1]
        design[0::2, 2] = 1
        design[1::2, 0] = points[:, 1]
        design[1::2, 1] = points[:, 0]
        design[1::2, 3] = 1
        a, b, tx, ty = np.linalg.lstsq(
            design, destination.ravel(), rcond=None
        )[0]
        if not np.isfinite([a, b, tx, ty]).all() or np.hypot(a, b) <= 1e-8:
            raise ValueError("Invalid alignment transform.")
        matrix = np.array([[a, -b, tx], [b, a, ty]])
        return cv2.warpAffine(image, matrix, (112, 112), borderValue=0)

    def extraer(self, aligned_photo: np.ndarray) -> np.ndarray:
        return self.extractor.extract(aligned_photo)

    def comparar(self, template_a: ArrayLike, template_b: ArrayLike) -> float:
        """Return cosine similarity in [-1, 1]."""
        a = np.asarray(template_a, dtype=np.float64).ravel()
        b = np.asarray(template_b, dtype=np.float64).ravel()
        if a.shape != b.shape or a.size == 0:
            raise ValueError("Templates must have the same nonzero dimension.")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Templates must contain only finite values.")
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a <= 0 or norm_b <= 0:
            raise ValueError("Templates must have nonzero norms.")
        return float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))

    def inscribir(self, user_id: str, photo: str | Path | np.ndarray) -> None:
        self._enroll_with_source(
            user_id, photo, self._template_from_image(photo)
        )

    def _template_from_image(
        self, photo: str | Path | np.ndarray
    ) -> np.ndarray:
        """Detect exactly one face, align it and extract its template."""
        image = self._load_image(photo)
        detections = self.detectar(image)
        if len(detections) != 1:
            raise ValueError(f"Expected one face, detected {len(detections)}.")
        return self.extraer(self.alinear(image, detections[0].landmarks))

    def _template_from_crop(self, photo: np.ndarray) -> np.ndarray:
        """Resize a pre-cropped face to 112 by 112 and extract its template."""
        face = self._load_image(photo)
        if face.ndim != 3 or face.shape[2] != 3:
            raise ValueError("Expected a face image with three BGR channels.")
        if face.shape[:2] != (112, 112):
            face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
        return self.extraer(face)

    def _enroll_with_source(
        self,
        user_id: str,
        photo: str | Path | np.ndarray,
        template: np.ndarray,
        *,
        replace_existing: bool = False,
    ) -> None:
        source = self._archive_source(photo)
        try:
            relative_source = source.relative_to(self.db.path.parent.resolve())
            self.db.add_user(
                user_id,
                relative_source.as_posix(),
                template=template,
                replace_existing=replace_existing,
            )
        except Exception:
            source.unlink(missing_ok=True)
            raise

    def _archive_source(self, photo: str | Path | np.ndarray) -> Path:
        """Archive original image bytes or encode an array as a PNG."""
        directory = self.db.path.parent / "images"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = (
            Path(photo).suffix if isinstance(photo, (str, Path)) else ".png"
        )
        descriptor, filename = tempfile.mkstemp(
            prefix="source_", suffix=suffix, dir=directory
        )
        target = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as archived:
                if isinstance(photo, (str, Path)):
                    with Path(photo).open("rb") as original:
                        shutil.copyfileobj(original, archived)
                else:
                    success, encoded = cv2.imencode(".png", photo)
                    if not success:
                        raise ValueError("Could not archive the source image.")
                    archived.write(encoded.tobytes())
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target.resolve()

    def verificar(self, user_id: str, photo: str | Path | np.ndarray) -> bool:
        return self._verify_template(user_id, self._template_from_image(photo))

    def identificar(self, photo: str | Path | np.ndarray) -> str | None:
        return self._identify_template(self._template_from_image(photo))

    def inscribir_sd(self, user_id: str, photo: np.ndarray) -> None:
        self._enroll_with_source(
            user_id, photo, self._template_from_crop(photo)
        )

    def verificar_sd(self, user_id: str, photo: np.ndarray) -> bool:
        return self._verify_template(user_id, self._template_from_crop(photo))

    def identificar_sd(self, photo: np.ndarray) -> str | None:
        return self._identify_template(self._template_from_crop(photo))

    def inscribir_t(self, user_id: str, template: np.ndarray) -> None:
        """Enroll a precomputed template without a source image."""
        values = np.asarray(template, dtype=np.float64).ravel()
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("The template must have a nonzero norm.")
        self.db.add_user(user_id, template=values / norm)

    def verificar_t(self, user_id: str, template: np.ndarray) -> bool:
        """Verify a direct template and reject exact enrollment copies."""
        return self._verify_template(user_id, template, reject_copy=True)

    def _verify_template(
        self, user_id: str, template: np.ndarray, *, reject_copy: bool = False
    ) -> bool:
        users = self.db.list_users()
        if reject_copy:
            self._reject_enrolled_copy(template, users)
        record = next(
            (user for user in users if user.user_id == user_id.strip()),
            None,
        )
        if record is None or not record.template:
            return False
        return (
            self.comparar(template, record.template)
            >= self.verification_threshold
        )

    def identificar_t(self, template: np.ndarray) -> str | None:
        """Identify a direct template and reject exact enrollment copies."""
        return self._identify_template(template, reject_copy=True)

    def _identify_template(
        self, template: np.ndarray, *, reject_copy: bool = False
    ) -> str | None:
        scores = self._gallery_scores(
            template, self.db.list_users(), reject_copy=reject_copy
        )
        best_id = max(scores, key=scores.__getitem__, default=None)
        if best_id is None or scores[best_id] < self.identification_threshold:
            return None
        return best_id

    def _gallery_scores(
        self,
        template: np.ndarray,
        users: Sequence[UserRecord],
        *,
        reject_copy: bool = True,
    ) -> dict[str, float]:
        """Score enrolled users in gallery order."""
        if reject_copy:
            self._reject_enrolled_copy(template, users)
        return {
            user.user_id: self.comparar(template, user.template)
            for user in users
            if user.template
        }

    @staticmethod
    def _reject_enrolled_copy(
        template: np.ndarray, users: Sequence[UserRecord]
    ) -> None:
        values = np.asarray(template).ravel()
        if any(
            user.template and np.array_equal(values, user.template)
            for user in users
        ):
            raise ValueError(
                "Possible template compromise: exact enrollment match."
            )

    @staticmethod
    def _load_image(photo: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(photo, np.ndarray):
            if photo.size == 0:
                raise ValueError("The image array is empty.")
            return photo
        image_path = Path(photo)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not decode {image_path}")
        return image
