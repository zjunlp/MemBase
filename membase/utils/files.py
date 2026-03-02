from pathlib import Path
import importlib
import sys
from typing import Callable


def import_function_from_path(function_path: str) -> Callable:
    """Dynamically import a function from a module path.
    
    This function supports two formats:
    1. Standard Python module path: "module.submodule.function_name"
    2. File path: "path/to/file.py:function_name"
    
    Args:
        function_path (`str`):
            The function path in the format "module.submodule.function_name"
            or "path/to/file.py:function_name" for local files.
    
    Returns:
        `Callable`:
            The imported function.
    """
    # Check if it's a file path (contains .py:).
    if ".py:" in function_path:
        file_path, func_name = function_path.rsplit(":", 1)
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' is not found.")
        
        # Add the directory to sys.path temporarily.
        module_dir = str(file_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Import the module.
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from '{file_path}'.")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function.
        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' is not found in '{file_path}'.")
        
        return getattr(module, func_name)
    else:
        # Standard module import (e.g., "package.module.function").
        module_path, func_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        
        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' is not found in module '{module_path}'.")
        
        return getattr(module, func_name)