"""Segformer (ADE20K, 150 classes) semantic segmentation preprocessor.

Uses ``nvidia/segformer-b5-finetuned-ade-640-640`` from Hugging Face to
produce a per-pixel walkability mask based on terrain-aware classes like
road, sidewalk, grass, earth, floor, path, etc.

Shadow normalization (CLAHE on the LAB L-channel) is applied before
inference to make the model robust to heavy shadows from buildings.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from data_ingestion.interfaces import ImageIngestedData

from ._mask_cleanup import cleanup_mask
from .entities import SegmentationMask, SegmentedImageData
from .interfaces import PreProcessor

# ── ADE20K class list (150 classes, 0-indexed) ────────────────────────
# Full list: https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640
ADE20K_CLASSES: list[str] = [
    "wall", "building", "sky", "floor", "tree",               # 0-4
    "ceiling", "road", "bed", "windowpane", "grass",           # 5-9
    "cabinet", "sidewalk", "person", "earth", "door",          # 10-14
    "table", "mountain", "plant", "curtain", "chair",          # 15-19
    "car", "water", "painting", "sofa", "shelf",               # 20-24
    "house", "sea", "mirror", "rug", "field",                  # 25-29
    "armchair", "seat", "fence", "desk", "rock",               # 30-34
    "wardrobe", "lamp", "bathtub", "railing", "cushion",       # 35-39
    "base", "box", "column", "signboard", "chest",             # 40-44
    "counter", "sand", "sink", "skyscraper", "fireplace",      # 45-49
    "refrigerator", "grandstand", "path", "stairs", "runway",  # 50-54
    "case", "pool", "pillow", "screen", "stairway",            # 55-59
    "river", "bridge", "bookcase", "blind", "coffee_table",    # 60-64
    "toilet", "flower", "book", "hill", "bench",               # 65-69
    "countertop", "stove", "palm", "kitchen_island",           # 70-73
    "computer", "swivel_chair", "boat", "bar", "arcade",       # 74-78
    "hovel", "bus", "towel", "light", "truck",                 # 79-83
    "tower", "chandelier", "awning", "streetlight", "booth",   # 84-88
    "television", "airplane", "dirt_track", "apparel", "pole", # 89-93
    "land", "bannister", "escalator", "ottoman", "bottle",     # 94-98
    "buffet", "poster", "stage", "van", "ship",                # 99-103
    "fountain", "conveyor", "canopy", "washer", "plaything",   # 104-108
    "swimming_pool", "stool", "barrel", "basket", "waterfall", # 109-113
    "tent", "bag", "minibike", "cradle", "oven",               # 114-118
    "ball", "food", "step", "tank", "trade_name",              # 119-123
    "microwave", "pot", "animal", "bicycle", "lake",           # 124-128
    "dishwasher", "screen_door", "blanket", "sculpture",       # 129-132
    "hood", "sconce", "vase", "traffic_light", "tray",         # 133-137
    "ashcan", "fan", "pier", "crt_screen", "plate",            # 138-142
    "monitor", "bulletin_board", "shower", "radiator",         # 143-146
    "glass", "clock", "flag",                                  # 147-149
]

# Classes considered walkable ground for a robot
DEFAULT_WALKABLE_IDS: set[int] = {
    3,   # floor / flooring
    6,   # road / route
    9,   # grass
    11,  # sidewalk / pavement
    13,  # earth / ground
    29,  # field
    46,  # sand
    52,  # path
    53,  # stairs
    59,  # stairway
    69,  # bench area (ground near benches)
    91,  # dirt_track
    94,  # land
    121, # step
}

_DEFAULT_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"


def _normalize_shadows(bgr: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB to reduce shadow effects."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


class SegformerPreProcessor(PreProcessor):
    """Semantic segmentation with Segformer B5 (ADE20K, 150 classes).

    Parameters
    ----------
    model_name :
        Hugging Face model ID.
    walkable_classes :
        Set of ADE20K class indices considered walkable.
    confidence_threshold :
        Minimum softmax probability to trust a prediction.
    max_input_size :
        Large images are resized before inference; mask is upscaled back.
    shadow_clip_limit :
        CLAHE clip-limit for shadow normalization (0 to disable).
    min_area :
        Connected components smaller than this are removed from the mask.
    device :
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        walkable_classes: set[int] | None = None,
        confidence_threshold: float = 0.5,
        max_input_size: int = 1024,
        shadow_clip_limit: float = 3.0,
        min_area: int = 1000,
        device: str | None = None,
    ):
        self._walkable_classes = walkable_classes or DEFAULT_WALKABLE_IDS
        self._confidence_threshold = confidence_threshold
        self._max_input_size = max_input_size
        self._shadow_clip_limit = shadow_clip_limit
        self._min_area = min_area

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        print(f"[Segformer] Loading {model_name} on {self._device} …")
        self._processor = SegformerImageProcessor.from_pretrained(model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            model_name
        ).to(self._device)
        self._model.eval()

    def name(self) -> str:
        return "segformer"

    # ------------------------------------------------------------------

    def process(self, data: ImageIngestedData) -> SegmentedImageData:
        pixels = data.get_pixels()  # (H, W, C) BGR
        h_orig, w_orig = pixels.shape[:2]

        # Shadow normalization
        if self._shadow_clip_limit > 0:
            pixels_norm = _normalize_shadows(pixels, self._shadow_clip_limit)
        else:
            pixels_norm = pixels

        rgb = cv2.cvtColor(pixels_norm, cv2.COLOR_BGR2RGB)

        # Resize large images
        scale = min(self._max_input_size / max(h_orig, w_orig), 1.0)
        if scale < 1.0:
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_small = rgb

        # Inference
        inputs = self._processor(images=rgb_small, return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, 150, H', W')

        # Upsample logits to the resized image dimensions
        logits_up = torch.nn.functional.interpolate(
            logits,
            size=rgb_small.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.softmax(logits_up, dim=1)
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()
        conf = probs.max(dim=1).values.squeeze(0).cpu().numpy()

        # Walkability mask
        walkable = np.isin(pred, list(self._walkable_classes))
        walkable &= conf >= self._confidence_threshold

        # Upscale to original resolution
        if scale < 1.0:
            walkable = cv2.resize(
                walkable.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            conf = cv2.resize(conf, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Morphological cleanup + island removal
        walkable = cleanup_mask(walkable, min_area=self._min_area)

        segmentation = SegmentationMask(
            mask=walkable,
            confidence=conf,
            class_map={i: name for i, name in enumerate(ADE20K_CLASSES)},
        )

        return SegmentedImageData(
            source_path=data.get_source(),
            pixels=data.get_pixels(),
            segmentation=segmentation,
            resolution_m_per_px=getattr(data, "resolution_m_per_px", None),
            metadata=getattr(data, "metadata", {}),
        )
