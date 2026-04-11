from __future__ import annotations

from .hsv_threshold import HSVThresholdPreProcessor
from .interfaces import PreProcessor


class PreProcessorFactory:
    """Registry-based factory for ``PreProcessor`` implementations.

    Register a preprocessor once::

        PreProcessorFactory.register("my_segmenter", MySegmenter)

    Then create instances anywhere::

        pre = PreProcessorFactory.create("my_segmenter")
        segmented = pre.process(image_data)
    """

    _registry: dict[str, type[PreProcessor]] = {}

    @classmethod
    def register(cls, name: str, preprocessor_class: type[PreProcessor]) -> None:
        cls._registry[name] = preprocessor_class

    @classmethod
    def create(cls, name: str, **kwargs) -> PreProcessor:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown preprocessor: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


# Always-available baseline
PreProcessorFactory.register("hsv_threshold", HSVThresholdPreProcessor)

# DeepLabV3+ requires torch/torchvision — register only when available
try:
    from .semantic_segmentation import DeepLabV3PreProcessor

    PreProcessorFactory.register("deeplabv3", DeepLabV3PreProcessor)
except ImportError:
    pass

# Segformer requires transformers + torch — register only when available
try:
    from .segformer_segmentation import SegformerPreProcessor

    PreProcessorFactory.register("segformer", SegformerPreProcessor)
except ImportError:
    pass
