from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping


_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ("MemBase", "MemBaseDataset"),
        ("LongMemEval", "LongMemEval"),
        ("LoCoMo", "LoCoMo"),
    ]
)

_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("MemBase", "base"),
        ("LongMemEval", "longmemeval"),
        ("LoCoMo", "locomo"),
    ]
)

DATASET_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)

__all__ = ["DATASET_MAPPING"]
