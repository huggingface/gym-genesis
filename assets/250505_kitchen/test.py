import os
import sys

LRP_PATH = "/Users/jade/Development/Genesis/genesis/ext/LuisaRender/build/bin"

sys.path.append(LRP_PATH)

try:
    import LuisaRenderPy
except ImportError as e:
    print(f"Failed to import LuisaRenderer. {e.__class__.__name__}: {e}")
