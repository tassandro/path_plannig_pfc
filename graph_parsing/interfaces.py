from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from data_ingestion.interfaces import IngestedData

from .entities import Graph

T = TypeVar("T", bound=IngestedData)


class GraphParser(ABC, Generic[T]):
    """Base interface for converting ingested data into a ``Graph``.

    Type-parameter ``T`` narrows the accepted data type:

    * ``GraphParser[IngestedData]``  — works with any data source.
    * ``GraphParser[ImageIngestedData]`` — image-only parser.
    * ``GraphParser[PointCloudIngestedData]`` — point-cloud-only parser.
    """

    @abstractmethod
    def parse(self, data: T) -> Graph:
        """Convert *data* into a graph representation."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this parsing strategy."""
        ...
