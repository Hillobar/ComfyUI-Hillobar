"""Node modules for ComfyUI-Hillobar.

Each module exposes its own ``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS``;
this package merges them and the pack's root ``__init__`` re-exports the result.
Add a new module here and to the merge below to register it.
"""

from .minimax_h3_progressive_sampler import (
    NODE_CLASS_MAPPINGS as _MMPROG_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MMPROG_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **_MMPROG_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_MMPROG_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
