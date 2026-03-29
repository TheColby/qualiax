"""qualiax — Perceptual Speech & Audio Quality Analyzer"""

from .version import __version__, OUTPUT_SCHEMA_VERSION

from . import gpu

def device_info() -> str:
    """Return the active compute device string."""
    return gpu.get_device_str()
