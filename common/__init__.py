import os
import sys

# Offscreen rendering on Linux without a display needs EGL, and MuJoCo picks its
# GL backend when first imported, which happens in common.duck / common.run / common.overlay.
if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"
