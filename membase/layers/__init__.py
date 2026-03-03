from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping


_MAPPING_NAMES = OrderedDict[str, str](
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

_MODULE_MAPPING = OrderedDict[str, str](
    [
        ("A-MEM", "amem"),
        ("LangMem", "langmem"),
        ("Long-Context", "long_context"),
        ("NaiveRAG", "naive_rag"),
        ("MemOS", "memos"),
        ("EverMemOS", "evermemos"),
        ("HippoRAG2", "hipporag"),
    ]
)

MEMORY_LAYERS_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)


__all__ = ["MEMORY_LAYERS_MAPPING"]
