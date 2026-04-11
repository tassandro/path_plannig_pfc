from .entities import SegmentationMask, SegmentedImageData
from .hsv_threshold import HSVThresholdPreProcessor
from .index import PreProcessorFactory
from .interfaces import PreProcessor

__all__ = [
    "PreProcessor",
    "PreProcessorFactory",
    "SegmentationMask",
    "SegmentedImageData",
    "HSVThresholdPreProcessor",
]

try:
    from .semantic_segmentation import DeepLabV3PreProcessor

    __all__.append("DeepLabV3PreProcessor")
except ImportError:
    pass

try:
    from .segformer_segmentation import SegformerPreProcessor

    __all__.append("SegformerPreProcessor")
except ImportError:
    pass
