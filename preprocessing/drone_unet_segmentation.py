"""Drone U-Net semantic segmentation preprocessor.

Wraps the pre-trained U-Net checkpoints released by Major Fábio Luiz Junior
(https://github.com/fabioluiz1987/semantic_segmentation_models), trained on
the **Semantic Drone Dataset** (23 classes, low-altitude aerial imagery) to
identify traversable vs. non-traversable terrain for a UGV.

Each checkpoint is a *whole* ``segmentation_models_pytorch`` U-Net object
(``torch.save(model)``), not a state-dict, so loading requires
``segmentation_models_pytorch`` **< 0.5** (0.5 renamed internal decoder
classes and can no longer unpickle these files) plus ``efficientnet_pytorch``
for the EfficientNet-backboned models.

The checkpoints are Git-LFS files that are *not* vendored in this repo.
Download them once with::

    python scripts/download_drone_models.py

which populates ``models/<Backbone>/<file>.pt``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from data_ingestion.interfaces import ImageIngestedData

from ._mask_cleanup import cleanup_mask
from .entities import SegmentationMask, SegmentedImageData
from .interfaces import PreProcessor

# ── Semantic Drone Dataset — 23 classes (label order, index 0 = unlabeled) ──
SDD_CLASSES: list[str] = [
    "unlabeled",   # 0
    "paved-area",  # 1
    "dirt",        # 2
    "grass",       # 3
    "gravel",      # 4
    "water",       # 5
    "rocks",       # 6
    "pool",        # 7
    "vegetation",  # 8
    "roof",        # 9
    "wall",        # 10
    "window",      # 11
    "door",        # 12
    "fence",       # 13
    "fence-pole",  # 14
    "person",      # 15
    "dog",         # 16
    "car",         # 17
    "bicycle",     # 18
    "tree",        # 19
    "bald-tree",   # 20
    "ar-marker",   # 21
    "obstacle",    # 22
]

# Classes a ground robot can traverse: paved-area, dirt, grass, gravel.
DEFAULT_WALKABLE_IDS: set[int] = {1, 2, 3, 4}

# Friendly model name → checkpoint location relative to the ``models/`` dir
MODEL_REGISTRY: dict[str, str] = {
    "resnet50": os.path.join("ResNet 50", "resnet50.pt"),
    "resnet101": os.path.join("ResNet 101", "resnet101.pt"),
    "efficientnet-b4": os.path.join("EfficientNet-B4", "efficientnet-b4.pt"),
    "efficientnet-b5": os.path.join("EfficientNet-B5", "efficientnet-b5.pt"),
    "mobilenet_v2": os.path.join("MobileNet V2", "Unet-Mobilenet_v2.pt"),
}

# Repository-root-relative ``models/`` directory.
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

# ImageNet statistics — the checkpoints were trained on ImageNet-normalized RGB.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def available_models() -> list[str]:
    """Friendly model names known to this preprocessor."""
    return list(MODEL_REGISTRY)


def _normalize_shadows(bgr: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE on the LAB L-channel to reduce hard building shadows."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


class DroneUNetPreProcessor(PreProcessor):
    """Semantic segmentation with Junior's Semantic-Drone-Dataset U-Nets.

    Parameters
    ----------
    model_name :
        One of :func:`available_models` (``resnet50``, ``resnet101``,
        ``efficientnet-b4``, ``efficientnet-b5``). Ignored when
        ``model_path`` is given.
    model_path :
        Explicit path to a ``.pt`` checkpoint. Overrides ``model_name``.
    walkable_classes :
        SDD class indices considered walkable. Default:
        ``{1, 2, 3, 4}`` (paved-area, dirt, grass, gravel).
    confidence_threshold :
        Minimum softmax probability required to trust a walkable pixel.
        ``None`` (default) disables confidence filtering — every argmax
        prediction is kept.
    max_input_size :
        Longest-edge size images are resized to before inference; the mask
        is upscaled back to the original resolution. ``None`` (default)
        disables resizing — inference runs at the original resolution
        (slower / more memory on large images).
    shadow_clip_limit :
        CLAHE clip-limit for shadow normalization. ``0`` disables it
        (default — the models were trained on un-normalized imagery).
    min_area :
        When set, the raw mask is passed through ``cleanup_mask``
        (morphological open/close + removal of connected components smaller
        than this). ``None`` (default) skips cleanup entirely, returning the
        raw per-pixel model output.
    device :
        ``"cpu"``, ``"cuda"`` or ``"mps"``. Auto-detected when ``None``.

    Notes
    -----
    The three optional post-processing steps (confidence filtering, input
    resizing, mask cleanup) are **all off by default**, so the out-of-the-box
    result is the raw model segmentation. Enable them individually — via the
    GUI toggles or the ``main.py`` CLI flags — to study their effect.
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b4",
        model_path: str | None = None,
        walkable_classes: set[int] | None = None,
        confidence_threshold: float | None = None,
        max_input_size: int | None = None,
        shadow_clip_limit: float = 0.0,
        min_area: int | None = None,
        device: str | None = None,
    ):
        self._model_name = model_name
        self._walkable_classes = walkable_classes or DEFAULT_WALKABLE_IDS
        self._confidence_threshold = confidence_threshold
        self._max_input_size = max_input_size
        self._shadow_clip_limit = shadow_clip_limit
        self._min_area = min_area

        # ── Resolve the checkpoint path ──────────────────────────────────
        if model_path is None:
            if model_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown drone model '{model_name}'. "
                    f"Available: {available_models()} (or pass model_path=…)."
                )
            model_path = os.path.join(_MODELS_DIR, MODEL_REGISTRY[model_name])

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}\n"
                "The Semantic-Drone-Dataset checkpoints are Git-LFS files that "
                "are not vendored in this repo. Download them with:\n"
                "    python scripts/download_drone_models.py"
            )

        # ── Device selection ─────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        # ── Load the whole-model checkpoint ──────────────────────────────
        print(f"[DroneUNet] Loading {model_name} from {model_path} on {self._device} …")
        try:
            model = torch.load(model_path, map_location=self._device, weights_only=False)
        except AttributeError as exc:  # smp >= 0.5 renamed internal classes
            raise ImportError(
                "Failed to unpickle the checkpoint. These models require "
                "segmentation_models_pytorch < 0.5 (0.5 renamed decoder "
                "classes). Install a compatible version:\n"
                "    pip install 'segmentation-models-pytorch>=0.3,<0.5'"
            ) from exc
        model.to(self._device)
        model.eval()
        self._model = model

    def name(self) -> str:
        return f"drone_{self._model_name}"

    # ------------------------------------------------------------------

    def process(self, data: ImageIngestedData) -> SegmentedImageData:
        pixels = data.get_pixels()  # (H, W, C) BGR
        h_orig, w_orig = pixels.shape[:2]

        # Optional shadow normalization
        if self._shadow_clip_limit > 0:
            pixels_norm = _normalize_shadows(pixels, self._shadow_clip_limit)
        else:
            pixels_norm = pixels

        rgb = cv2.cvtColor(pixels_norm, cv2.COLOR_BGR2RGB)

        # Optional resize to keep inference tractable (disabled when None)
        if self._max_input_size is not None:
            scale = min(self._max_input_size / max(h_orig, w_orig), 1.0)
        else:
            scale = 1.0
        if scale < 1.0:
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_small = rgb

        # The U-Net encoder downsamples by 32, so H and W must be multiples of
        # 32. Reflect-pad up to the next multiple, then crop the output back.
        h_s, w_s = rgb_small.shape[:2]
        pad_h = (32 - h_s % 32) % 32
        pad_w = (32 - w_s % 32) % 32
        if pad_h or pad_w:
            rgb_pad = cv2.copyMakeBorder(
                rgb_small, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )
        else:
            rgb_pad = rgb_small

        # ImageNet-normalized CHW tensor (matches training preprocessing)
        norm = (rgb_pad.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)  # (1, num_classes, H', W')
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            conf = probs.max(dim=1).values.squeeze(0).cpu().numpy()

        # Remove the reflect padding
        pred = pred[:h_s, :w_s]
        conf = conf[:h_s, :w_s]

        # Walkability mask (optional confidence filtering when threshold set)
        walkable = np.isin(pred, list(self._walkable_classes))
        if self._confidence_threshold is not None:
            walkable &= conf >= self._confidence_threshold

        # Up-scale back to original resolution
        if scale < 1.0:
            walkable = cv2.resize(
                walkable.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            conf = cv2.resize(conf, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Optional morphological cleanup + island removal (disabled when None)
        if self._min_area is not None:
            walkable = cleanup_mask(walkable, min_area=self._min_area)

        segmentation = SegmentationMask(
            mask=walkable,
            confidence=conf,
            class_map={i: name for i, name in enumerate(SDD_CLASSES)},
        )

        return SegmentedImageData(
            source_path=data.get_source(),
            pixels=data.get_pixels(),
            segmentation=segmentation,
            resolution_m_per_px=getattr(data, "resolution_m_per_px", None),
            metadata=getattr(data, "metadata", {}),
        )
