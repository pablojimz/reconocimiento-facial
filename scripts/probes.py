"""Load real probes or generate synthetic probes from enrollment images."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from reconocimiento_facial.system import SistemaReconocimientoFacial

from .calibrate_thresholds import EXTENSIONS, variations

YEAR_SUFFIX = re.compile(r"_\d{4}$")


def identity(path: Path) -> str:
    """Derive an identity from the filename without its extension or year."""
    return YEAR_SUFFIX.sub("", path.stem)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


@dataclass(frozen=True)
class Probe:
    identity: str
    source: str
    template: np.ndarray | None


def _extract_probe(
    system: SistemaReconocimientoFacial, image: np.ndarray | None
) -> np.ndarray | None:
    if image is None:
        return None
    faces = system.detectar(image)
    if not faces:
        return None
    face = max(faces, key=lambda detection: detection.confidence)
    return system.extraer(system.alinear(image, face.landmarks))


def load_probes(
    system: SistemaReconocimientoFacial,
    dataset: Path,
    probes: Path | None,
    cache: Path,
) -> list[Probe]:
    paths = image_paths(probes if probes is not None else dataset)
    if not paths:
        raise ValueError("The probe dataset contains no images.")
    signature = hashlib.sha256(b"real" if probes is not None else b"synthetic")
    for dependency in (
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "timm",
    ):
        signature.update(version(dependency).encode("utf-8") + b"\0")
    signature.update(str(system.detector._device).encode("utf-8"))
    for path in paths:
        signature.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as source:
            signature.update(hashlib.file_digest(source, "sha256").digest())
    for path in (
        Path(__file__),
        Path(__file__).with_name("calibrate_thresholds.py"),
        *sorted(
            (
                Path(__file__).resolve().parents[1] / "reconocimiento_facial"
            ).glob("*.py")
        ),
        system.detector.assets.weights_path,
        system.extractor.assets.weights_path,
    ):
        with path.open("rb") as source:
            signature.update(hashlib.file_digest(source, "sha256").digest())
    cache_key = signature.hexdigest()

    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["signature"]) == cache_key:
                print(f"Loaded probes from {cache}.")
                return [
                    Probe(str(name), str(source), template if valid else None)
                    for name, source, template, valid in zip(
                        data["identities"],
                        data["sources"],
                        data["templates"],
                        data["valid"],
                    )
                ]

    result = []
    for path in paths:
        image = cv2.imread(str(path))
        if probes is not None:
            template = _extract_probe(system, image)
            result.append(Probe(identity(path), path.name, template))
            continue
        detections = system.detectar(image) if image is not None else []
        if not detections:
            print(f"  {path.name}: no face, skipping synthetic probes")
            continue
        face = max(detections, key=lambda detection: detection.confidence)
        for kind, altered in variations(image, face.bbox):
            template = _extract_probe(system, altered)
            result.append(
                Probe(identity(path), f"{path.name}:{kind}", template)
            )
        print(f"  Extracted probes for {path.name}")

    if not result:
        raise ValueError("No probes were generated from the dataset.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as directory:
        staged = Path(directory) / cache.name
        np.savez_compressed(
            staged,
            signature=np.array(cache_key),
            identities=np.array([probe.identity for probe in result]),
            sources=np.array([probe.source for probe in result]),
            templates=np.stack(
                [
                    probe.template
                    if probe.template is not None
                    else np.zeros(512, np.float32)
                    for probe in result
                ]
            ),
            valid=np.array([probe.template is not None for probe in result]),
        )
        staged.replace(cache)
    return result
