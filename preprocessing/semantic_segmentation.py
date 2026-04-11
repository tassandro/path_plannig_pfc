from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)

from data_ingestion.interfaces import ImageIngestedData

from .entities import SegmentationMask, SegmentedImageData
from .interfaces import PreProcessor

VOC_CLASSES: list[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# For aerial imagery the ground is "background" (class 0).
# Everything the model explicitly recognises is an object / obstacle.
DEFAULT_WALKABLE_CLASSES: set[int] = {0}


class DeepLabV3PreProcessor(PreProcessor):
    """Semantic segmentation with DeepLabV3+ (ResNet-101, COCO/VOC).

    For aerial / drone imagery most of the terrain is classified as
    *background* (class 0) while detected objects (vehicles, people …)
    become obstacles.

    Parameters
    ----------
    walkable_classes :
        Set of VOC class indices considered walkable.
        Default: ``{0}`` (background = terrain).
    confidence_threshold :
        Minimum softmax probability to trust a prediction.
    max_input_size :
        Large images are resized so the longest edge equals this value
        before inference, then the mask is upscaled back.
    device :
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected when
        ``None``.
    """

    def __init__(
        self,
        walkable_classes: set[int] | None = None,
        confidence_threshold: float = 0.5,
        max_input_size: int = 1024,
        device: str | None = None,
    ):
        self._walkable_classes = walkable_classes or DEFAULT_WALKABLE_CLASSES
        self._confidence_threshold = confidence_threshold
        self._max_input_size = max_input_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        print(f"[DeepLabV3] Loading model on {self._device} …")
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self._model = deeplabv3_resnet101(weights=weights).to(self._device)
        self._model.eval()

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def name(self) -> str:
        return "deeplabv3"

    # ------------------------------------------------------------------

    def process(self, data: ImageIngestedData) -> SegmentedImageData:
        pixels = data.get_pixels()  # (H, W, C) BGR
        h_orig, w_orig = pixels.shape[:2]
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

        # Resize large images to keep inference tractable
        scale = min(self._max_input_size / max(h_orig, w_orig), 1.0)
        if scale < 1.0:
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_small = rgb

        tensor = transforms.functional.to_tensor(rgb_small)   # (3, H, W) float [0,1]
        tensor = self._normalize(tensor)
        batch = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(batch)["out"]  # (1, 21, H', W')
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            conf = probs.max(dim=1).values.squeeze(0).cpu().numpy()

        # Walkability mask
        walkable = np.isin(pred, list(self._walkable_classes))
        walkable &= conf >= self._confidence_threshold

        # Up-scale back to original resolution
        if scale < 1.0:
            walkable = cv2.resize(
                walkable.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            conf = cv2.resize(conf, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        segmentation = SegmentationMask(
            mask=walkable,
            confidence=conf,
            class_map={i: name for i, name in enumerate(VOC_CLASSES)},
        )

        return SegmentedImageData(
            source_path=data.get_source(),
            pixels=pixels,
            segmentation=segmentation,
            resolution_m_per_px=getattr(data, "resolution_m_per_px", None),
            metadata=getattr(data, "metadata", {}),
        )
