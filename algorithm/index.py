from __future__ import annotations

from .interfaces import PathAlgorithm


class AlgorithmFactory:
    """Registry-based factory for ``PathAlgorithm`` implementations.

    Register an algorithm once::

        AlgorithmFactory.register("a_star", AStarAlgorithm)

    Then create instances anywhere::

        algo = AlgorithmFactory.create("a_star")
        result = algo.find_path(graph, start=0, end=42)
    """

    _registry: dict[str, type[PathAlgorithm]] = {}

    @classmethod
    def register(cls, name: str, algorithm_class: type[PathAlgorithm]) -> None:
        cls._registry[name] = algorithm_class

    @classmethod
    def create(cls, name: str, **kwargs) -> PathAlgorithm:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown algorithm: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
