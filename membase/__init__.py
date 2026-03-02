import importlib
from collections import OrderedDict
from typing import (
    Any, 
    Type, 
    Literal, 
    Iterator, 
)

# Mapping of memory layer types to their class names.
MEMORY_LAYERS_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("A-MEM", "AMEMLayer"),
        ("LangMem", "LangMemLayer"),
        ("Long-Context", "LongContextLayer"),
        ("NaiveRAG", "NaiveRAGLayer"),
        ("MemOS", "MemOSLayer"),
        ("EverMemOS", "EverMemOSLayer"),
        ("HippoRAG2", "HippoRAGLayer"),
    ]
)

# Mapping of memory config types to their class names.
CONFIG_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("A-MEM", "AMEMConfig"),
        ("LangMem", "LangMemConfig"),
        ("Long-Context", "LongContextConfig"),
        ("NaiveRAG", "NaiveRAGConfig"),
        ("MemOS", "MemOSConfig"),
        ("EverMemOS", "EverMemOSConfig"),
        ("HippoRAG2", "HippoRAGConfig"),
    ]
)

# Mapping of dataset types to their class names.
DATASET_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("MemBase", "MemBaseDataset"),
        ("LongMemEval", "LongMemEval"),
        ("LoCoMo", "LoCoMo"),
    ]
)


def type_to_module_name(key: str, mapping_type: Literal["layer", "config", "dataset"]) -> str:
    """Convert a type key to the corresponding module path.
    
    Args:
        key (`str`): 
            The type key (e.g., ``"A-MEM"``, ``"LongMemEval"``).
        mapping_type (`Literal["layer", "config", "dataset"]`): 
            The type of mapping.

    Returns:
        `str`: 
            The module path relative to the ``membase`` package.
    """
    match mapping_type:
        case "layer":
            match key:
                case "A-MEM":
                    return "layers.amem"
                case "LangMem":
                    return "layers.langmem"
                case "MemZero":
                    return "layers.memzero"
                case "MemZeroGraph":
                    return "layers.memzero"
                case "NaiveRAG":
                    return "layers.naive_rag"
                case "Long-Context":
                    return "layers.long_context"
                case "MemOS":
                    return "layers.memos"
                case "EverMemOS":
                    return "layers.evermemos"
                case "HippoRAG2":
                    return "layers.hipporag"
        case "config":
            match key:
                case "A-MEM":
                    return "configs.amem"
                case "LangMem":
                    return "configs.langmem"
                case "MemZero":
                    return "configs.memzero"
                case "MemZeroGraph":
                    return "configs.memzero"
                case "Long-Context":
                    return "configs.long_context"
                case "NaiveRAG":
                    return "configs.naive_rag"
                case "MemOS":
                    return "configs.memos"
                case "EverMemOS":
                    return "configs.evermemos"
                case "HippoRAG2":
                    return "configs.hipporag"
        case "dataset":
            match key:
                case "LongMemEval":
                    return "datasets.longmemeval"
                case "LoCoMo":
                    return "datasets.locomo"

    # Default: convert key to module name.
    return key.lower().replace("-", "_")


class _LazyMapping(OrderedDict):
    """A mapping that lazily loads its values when they are requested.
    
    Its design is inspired by [Hugging Face Transformers lazy loading mechanism](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/auto/configuration_auto.py).
    """
    
    def __init__(
        self, 
        mapping: OrderedDict[str, str], 
        mapping_type: Literal["layer", "config", "dataset"],
    ) -> None:
        """Initialize the lazy mapping.
        
        Args:
            mapping (`OrderedDict[str, str]`): 
                A mapping from type keys to class names.
            mapping_type (`Literal["layer", "config", "dataset"]`): 
                The type of mapping.
        """
        self._mapping = mapping
        self._mapping_type = mapping_type
        self._extra_content = {}
        self._modules = {}
    
    def __getitem__(self, key: str) -> Type[Any]:
        """Lazily load and return the class associated with the provided key."""
        if key in self._extra_content:
            return self._extra_content[key]
        
        if key not in self._mapping:
            raise KeyError(
                f"'{key}' is not found. Available keys are {list(self._mapping.keys())}."
            )
        
        class_name = self._mapping[key]
        module_name = type_to_module_name(key, self._mapping_type)
        
        # Cache the module if not already loaded. 
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(
                    f".{module_name}", 
                    "memories"
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {module_name} for {key}: {e}"
                )
        
        # Get the class from the module.
        if hasattr(self._modules[module_name], class_name):
            return getattr(self._modules[module_name], class_name)
        
        raise AttributeError(
            f"Module '{module_name}' does not have class '{class_name}'."
        )
    
    def keys(self) -> list[str]:
        """Return all available keys."""
        return list(self._mapping.keys()) + list(self._extra_content.keys())
    
    def values(self) -> list[Type[Any]]:
        """Return all values (load them if necessary)."""
        return [self[k] for k in self.keys()]
    
    def items(self) -> list[tuple[str, Type[Any]]]:
        """Return all key-value pairs."""
        return [(k, self[k]) for k in self.keys()]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())
    
    def __contains__(self, item: object) -> bool:
        """Check if a key exists in the mapping."""
        return item in self._mapping or item in self._extra_content
    
    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self._mapping) + len(self._extra_content)
    
    def register(self, key: str, value: Type[Any], exist_ok: bool = False) -> None:
        """Register a new class in this mapping.
        
        Args:
            key (`str`): 
                The key to register the class under.
            value (`Type[Any]`): 
                The class to register.
            exist_ok (`bool`, defaults to `False`): 
                If enabled, it allows overwriting existing keys. Otherwise, it raises an error.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(
                f"'{key}' is already registered in {self._mapping_type} mapping. "
                f"Use exist_ok=True to overwrite."
            )
        self._extra_content[key] = value
    

MEMORY_LAYERS_MAPPING = _LazyMapping(MEMORY_LAYERS_MAPPING_NAMES, "layer")
CONFIG_MAPPING = _LazyMapping(CONFIG_MAPPING_NAMES, "config")
DATASET_MAPPING = _LazyMapping(DATASET_MAPPING_NAMES, "dataset")


# Export the public APIs.
__all__ = [
    "CONFIG_MAPPING",
    "MEMORY_LAYERS_MAPPING", 
    "DATASET_MAPPING",
]
