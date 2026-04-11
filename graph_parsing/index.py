from __future__ import annotations

from data_ingestion.interfaces import IngestedData

from .grid_image_parser import GridImageGraphParser
from .interfaces import GraphParser
from .segmented_image_parser import SegmentedImageGraphParser


class GraphParserFactory:
    """Registry-based factory for ``GraphParser`` implementations.

    Register a parser once::

        GraphParserFactory.register("grid_image", GridImageParser)

    Then create instances anywhere::

        parser = GraphParserFactory.create("grid_image")
        graph  = parser.parse(image_data)
    """

    _registry: dict[str, type[GraphParser[IngestedData]]] = {}

    @classmethod
    def register(cls, name: str, parser_class: type[GraphParser[IngestedData]]) -> None:
        cls._registry[name] = parser_class

    @classmethod
    def create(cls, name: str, **kwargs) -> GraphParser[IngestedData]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown graph parser: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


GraphParserFactory.register("grid_image", GridImageGraphParser)
GraphParserFactory.register("segmented_image", SegmentedImageGraphParser)
