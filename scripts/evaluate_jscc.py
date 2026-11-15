"""Compatibility entry point for src.jscc.inference."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.jscc.inference import main

if __name__ == "__main__":
    main()
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("src.jscc.inference")
