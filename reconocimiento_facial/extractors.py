"""Facial feature extraction with TransFace."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .dependencies import quiet_model_warnings
from .models import TransFaceAssets

if TYPE_CHECKING:
    import torch


class TransFaceBExtractor:
    """Carga una vez el ViT-B oficial y genera templates L2 de 512 valores."""

    def __init__(
        self,
        assets: TransFaceAssets | None = None,
        device: str | None = None,
    ) -> None:
        self.assets = assets or TransFaceAssets()
        self._device_name = device
        self._model: torch.nn.Module | None = None

    @quiet_model_warnings()
    def _load_model(self) -> None:
        import torch

        source_dir, weights_path = self.assets.require_ready()
        # Load only vit.py to avoid collisions with detector modules.
        spec = importlib.util.spec_from_file_location(
            "transface_vit", source_dir / "backbones" / "vit.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the TransFace implementation.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=512,
            embed_dim=512,
            depth=24,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.05,
            using_checkpoint=True,
        )
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint, strict=True)
        # Run upstream inference on CPU when CUDA is unavailable.
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_name or default_device)
        self._model = model.float().eval().to(self._device)

    @quiet_model_warnings()
    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Recibe BGR uint8 de 112×112 y devuelve un vector unitario."""
        if (
            aligned_face.shape != (112, 112, 3)
            or aligned_face.dtype != np.uint8
        ):
            raise ValueError(
                "TransFace requiere una cara BGR uint8 de 112×112×3."
            )
        if self._model is None:
            self._load_model()
        import torch

        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        tensor = (tensor.unsqueeze(0) / 255.0 - 0.5) / 0.5
        with torch.inference_mode():
            features, _, _ = self._model(tensor.to(self._device))
        embedding = features[0].float().cpu().numpy()
        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise ValueError("TransFace produced an invalid embedding.")
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(
                "TransFace produced an embedding with an invalid norm."
            )
        return embedding / norm
