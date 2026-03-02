from __future__ import annotations
from pathlib import Path
from ..model_types.dataset import MemoryDataset
from typing import Any


class MemBaseDataset(MemoryDataset):
    """Intermediate base for all MemBase dataset implementations.

    It provides a default (empty) metadata generator, a JSON-based ``read_raw_data``
    loader that leverages Pydantic deserialization, and a ``save_dataset`` method
    for persisting the dataset to disk.  Dataset-specific subclasses should
    override ``read_raw_data`` to parse their own raw formats and may override
    ``_generate_metadata`` for richer statistics.
    """

    @classmethod
    def read_dataset(cls, path: str) -> MemBaseDataset:
        """Read the standardized dataset from the given path.
        
        Args:
            path (`str`):
                The path to the standardized dataset.
        
        Returns:
            `MemBaseDataset`:
                The standardized dataset.
        """
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def _generate_metadata(self) -> dict[str, Any]:
        return {}

    @classmethod
    def read_raw_data(cls, path: str) -> MemBaseDataset:
        return cls.read_dataset(path)

    def save_dataset(self, name: str) -> None:
        """Serialize the dataset to a JSON file.

        Args:
            name (`str`):
                File stem without the ".json" extension.
        """
        Path(f"{name}.json").write_text(
            self.model_dump_json(indent=4), encoding="utf-8"
        )
