from __future__ import annotations

from abc import ABC, abstractmethod

from data_ingestion.interfaces import IngestedData


class PreProcessor(ABC):
    """Base interface for data preprocessing steps.

    A preprocessor transforms raw ingested data into an enriched form
    suitable for graph parsing — e.g. adding a segmentation mask that
    marks each pixel as walkable or obstacle.
    """

    @abstractmethod
    def process(self, data: IngestedData) -> IngestedData:
        """Return a (possibly new) ``IngestedData`` enriched with
        preprocessing results."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...
