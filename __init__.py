from .ltxv_block_loop import LTXVBlockLoop

# ─── registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LTXVBlockLoop": LTXVBlockLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVBlockLoop": "LTXV Block Loop Patcher",
}



__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
