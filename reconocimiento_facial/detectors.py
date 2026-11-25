"""Face detection with RetinaFace."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass

import numpy as np

from .dependencies import quiet_model_warnings
from .models import RetinaFaceAssets


@dataclass(frozen=True)
class FaceDetection:
    """A bounding box, confidence and landmarks in image coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "landmarks": self.landmarks,
        }


class RetinaFaceR50Detector:
    """RetinaFace inference using the pinned upstream ResNet-50 model."""

    def __init__(
        self,
        assets: RetinaFaceAssets | None = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        self.assets = assets or RetinaFaceAssets()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._device_name = device
        self._load_model()

    @quiet_model_warnings()
    def _load_model(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required. Run python -m pip install -e ."
            ) from error
        source_dir, weights_path = self.assets.require_ready()
        source_location = str(source_dir)
        if source_location not in sys.path:
            sys.path.insert(0, source_location)
        from data.config import cfg_re50
        from models.retinaface import RetinaFace

        configuration = copy.deepcopy(cfg_re50)
        configuration["pretrain"] = False
        device_name = self._device_name or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._torch = torch
        self._device = torch.device(device_name)
        self._cfg = configuration
        model = RetinaFace(cfg=configuration, phase="test")
        checkpoint = torch.load(weights_path, map_location=self._device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        # Cast the float64 checkpoint tensors to float32 for inference.
        self._model = model.float().eval().to(self._device)

    @quiet_model_warnings()
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        if image is None or image.size == 0:
            raise ValueError("The image is empty or could not be decoded.")
        torch = self._torch
        height, width = image.shape[:2]
        scale = torch.tensor(
            [width, height, width, height], device=self._device
        )
        scale_landmarks = torch.tensor(
            [width, height] * 5, device=self._device
        )
        normalized = image.astype(np.float32) - np.array(
            (104, 117, 123), dtype=np.float32
        )
        tensor = torch.from_numpy(normalized)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            locations, confidences, landmarks = self._model(tensor)
        priors = self._prior_boxes(height, width)
        boxes = self._decode_boxes(locations.squeeze(0), priors) * scale
        points = (
            self._decode_landmarks(landmarks.squeeze(0), priors)
            * scale_landmarks
        )
        scores = confidences.squeeze(0)[:, 1]
        keep = scores >= self.confidence_threshold
        if not bool(keep.any()):
            return []
        boxes = boxes[keep].cpu().numpy()
        points = points[keep].cpu().numpy().reshape(-1, 5, 2)
        scores = scores[keep].cpu().numpy()
        selected = self._nms(boxes, scores)
        detections: list[FaceDetection] = []
        for index in selected:
            x1, y1, x2, y2 = boxes[index]
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            detections.append(
                FaceDetection(
                    bbox=(
                        x,
                        y,
                        max(0, int(round(x2)) - x),
                        max(0, int(round(y2)) - y),
                    ),
                    confidence=float(scores[index]),
                    landmarks=tuple(
                        (float(x), float(y)) for x, y in points[index]
                    ),
                )
            )
        return detections

    def _prior_boxes(self, height: int, width: int):
        from layers.functions.prior_box import PriorBox

        return (
            PriorBox(self._cfg, image_size=(height, width))
            .forward()
            .to(self._device)
        )

    def _decode_boxes(self, locations, priors):
        torch = self._torch
        boxes = torch.cat(
            (
                priors[:, :2]
                + locations[:, :2] * self._cfg["variance"][0] * priors[:, 2:],
                priors[:, 2:]
                * torch.exp(locations[:, 2:] * self._cfg["variance"][1]),
            ),
            dim=1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _decode_landmarks(self, predictions, priors):
        torch = self._torch
        return torch.cat(
            tuple(
                priors[:, :2]
                + predictions[:, position : position + 2]
                * self._cfg["variance"][0]
                * priors[:, 2:]
                for position in range(0, 10, 2)
            ),
            dim=1,
        )

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        selected: list[int] = []
        while order.size:
            current = int(order[0])
            selected.append(current)
            xx1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            current_area = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1]
            )
            other_areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1]
            )
            iou = intersection / np.maximum(
                current_area + other_areas - intersection, 1e-12
            )
            order = order[1:][iou <= self.nms_threshold]
        return selected
