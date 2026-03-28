"""qualiax — Perceptual Speech & Audio Quality Analyzer"""
__version__ = "0.2.0"

from . import gpu

def device_info() -> str:
    """Return the active compute device string."""
    return gpu.get_device_str()
