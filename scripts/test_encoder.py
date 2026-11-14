"""Compatibility entry point for scripts.jscc.test_encoder."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.jscc.test_encoder import main

if __name__ == "__main__":
    main()
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("scripts.jscc.test_encoder")
