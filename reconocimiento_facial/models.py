"""Reproducible download of model code and weights."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen


class RetinaFaceAssets:
    """Manage pinned upstream code and weights outside the tracked tree."""

    repository_url = "https://github.com/biubug6/Pytorch_Retinaface.git"
    repository_commit = "b984b4b775b2c4dced95c1eadd195a5c7d32a60b"
    weights_url = (
        "https://huggingface.co/towzeur/deepfake_data_weights/resolve/main/"
        "detectors/Pytorch_Retinaface/Resnet50_Final.pth?download=true"
    )
    weights_sha256 = (
        "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d"
    )
    weights_filename = "Resnet50_Final.pth"

    def __init__(
        self, directory: str | Path = "models/retinaface-r50"
    ) -> None:
        self.directory = Path(directory)

    @property
    def source_dir(self) -> Path:
        return self.directory / "upstream"

    @property
    def weights_path(self) -> Path:
        return self.directory / "weights" / self.weights_filename

    def require_ready(self) -> tuple[Path, Path]:
        if not self.source_dir.is_dir() or not self.weights_path.is_file():
            raise RuntimeError(
                "Missing RetinaFace-R50 source or weights. Run "
                "reconocimiento-facial models download"
            )
        return self.source_dir, self.weights_path

    def download(self) -> Path:
        """Fetch the pinned upstream commit and the R50 weights."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            subprocess.run(
                ["git", "clone", self.repository_url, str(self.source_dir)],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_dir),
                    "checkout",
                    self.repository_commit,
                ],
                check=True,
            )
        if not self.weights_path.exists():
            self._download_weights()
        return self.weights_path

    def _download_weights(self) -> None:
        """Verify the downloaded checkpoint before installing it."""
        temporary_path = self.directory / f".{self.weights_filename}.download"
        try:
            with (
                urlopen(self.weights_url, timeout=60) as source,
                temporary_path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            if digest != self.weights_sha256:
                raise RuntimeError(
                    "Invalid SHA-256 digest for RetinaFace-R50 weights."
                )
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.replace(self.weights_path)
        finally:
            temporary_path.unlink(missing_ok=True)


class TransFaceAssets:
    """Pinned TransFace-B source and Glint360K checkpoint."""

    repository_url = "https://github.com/DanJun6737/TransFace.git"
    repository_commit = "fb6cd56d04c7ea525e328219f5d5066adac03ada"
    weights_id = "13IezvOo5GvtGVsRap2s5RVqtIl1y0ke5"
    weights_filename = "glint360k_model_TransFace_B.pt"
    weights_sha256 = (
        "95e5db485b2755ecfd50c0ae7b7681d229d295bbd15e18c7ddc7a24f47298c78"
    )

    def __init__(self, directory: str | Path = "models/transface-b") -> None:
        self.directory = Path(directory)

    @property
    def source_dir(self) -> Path:
        return self.directory / "upstream"

    @property
    def weights_path(self) -> Path:
        return self.directory / "weights" / self.weights_filename

    def require_ready(self) -> tuple[Path, Path]:
        implementation = self.source_dir / "backbones" / "vit.py"
        if not implementation.is_file() or not self.weights_path.is_file():
            raise RuntimeError(
                "Missing TransFace-B source or weights. Run "
                "reconocimiento-facial models download --model transface"
            )
        self._validate_weights(self.weights_path)
        return self.source_dir, self.weights_path

    def download(self) -> Path:
        """Descarga los pesos oficiales y comprueba su SHA-256."""
        import gdown

        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            subprocess.run(
                ["git", "clone", self.repository_url, str(self.source_dir)],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_dir),
                    "checkout",
                    self.repository_commit,
                ],
                check=True,
            )
        commit = subprocess.check_output(
            ["git", "-C", str(self.source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != self.repository_commit:
            raise RuntimeError(
                "Local TransFace code does not match the pinned commit."
            )
        if not self.weights_path.exists():
            temporary = self.directory / f".{self.weights_filename}.download"
            try:
                result = gdown.download(
                    id=self.weights_id,
                    output=str(temporary),
                    use_cookies=False,
                )
                if result is None:
                    raise RuntimeError(
                        "No se pudieron descargar los pesos de TransFace-B."
                    )
                self._validate_weights(temporary)
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(self.weights_path)
            finally:
                temporary.unlink(missing_ok=True)
        self.require_ready()
        return self.weights_path

    def _validate_weights(self, path: Path) -> None:
        with path.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if digest != self.weights_sha256:
            raise RuntimeError(
                "Invalid SHA-256 digest for TransFace-B weights."
            )
