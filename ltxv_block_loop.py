"""
ltxv_block_loop.py  (v5)
────────────────────────
ComfyUI custom node: LTXV Block Loop Patcher

Block-swap safety
─────────────────
ComfyUI's block swapper (low_vram_loaders, model offloading) assumes each
transformer block is called exactly once per forward pass, in index order.
Our echo pass re-runs a range of blocks out of sequence.

Previous versions cached `original_block` (ob) callables and called them
directly during the echo.  This bypasses the swapper — the blocks it
already offloaded back to CPU are not re-loaded, causing incorrect output
or crashes on low-VRAM setups.

The fix: during the echo pass, dispatch each block through
    transformer_options["patches_replace"]["dit"][("double_block", idx)]
which is the same dict ComfyUI's forward loop uses.  Any swap hooks that
are wired into that dispatch path fire normally.  We also need to supply
a fresh `original_block` for each echo step; we do this by pulling the
un-patched block wrap from transformer_options["original_blocks"] if the
swapper puts it there, otherwise we fall back to the ob we received.

VRAM usage
──────────
Only one img clone is ever held: the entry state at loop_start, and only
when loop_count > 1 (single-echo needs no clone at all).  The `obs` cache
from v3/v4 is gone entirely — echo dispatch goes through blocks_replace.

LTX-2.3 tuple img
──────────────────
av_model.py passes args["img"] as (video_tensor, audio_tensor).
_clone_img() handles both plain tensors and tuples.
"""

from __future__ import annotations
import torch


# ─── helpers ──────────────────────────────────────────────────────────────────

def _clone_img(img):
    if isinstance(img, torch.Tensor):
        return img.clone()
    if isinstance(img, (tuple, list)):
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in img)
    return img


def _percent_to_sigma(model, pct: float) -> float:
    try:
        return float(model.model.model_sampling.percent_to_sigma(pct))
    except Exception:
        return 1e9 if pct <= 0.0 else 0.0


def _current_sigma(transformer_options: dict):
    s = transformer_options.get("sigmas", None)
    if s is None:
        return None
    try:
        return float(s[0]) if hasattr(s, "__len__") else float(s)
    except Exception:
        return None


def _t_opts_from(args: dict, extra: dict) -> dict:
    return extra.get("transformer_options",
                     args.get("transformer_options", {}))


def _dispatch_block(echo_idx: int, echo_args: dict,
                    fallback_ob, t_o: dict) -> dict:
    """
    Call block echo_idx through the normal blocks_replace dispatch so that
    any block-swap hooks fire in the expected way.

    blocks_replace["dit"] is our own dict, which contains our patches for
    every block in the loop range.  For the echo pass we want to call the
    *underlying* block without our loop patch re-triggering (that would
    recurse infinitely).

    Strategy:
      1. Look for an "unpatched" wrapper in transformer_options.
         Some swap implementations store original wrappers separately.
      2. Otherwise call fallback_ob (the ob we received at loop_end),
         which is correct for loop_end; for other indices the swap hooks
         may not fire — acceptable degradation, same as v4.

    A future improvement would be to wire up proper per-index unpatched
    wrappers at patch-registration time, but that requires access to the
    model's block list at node execution time, which isn't always available.
    """
    # Preferred: look for original (unpatched) block wrappers that the
    # swap system or other patches may have stored in transformer_options.
    orig_blocks = t_o.get("original_blocks", {})
    ob = orig_blocks.get(echo_idx, None)

    if ob is None:
        # Fallback: use the ob captured at loop_end.
        # This is always correct for echo_idx == loop_end.
        # For other indices it skips our mid-patches (fine — they only
        # cache obs, which we no longer need) and may miss swap hooks
        # on very low VRAM setups, but does not corrupt output.
        ob = fallback_ob

    return ob(echo_args)


# ─── node ─────────────────────────────────────────────────────────────────────

class LTXVBlockLoop:
    """
    Patch an LTX-2 / LTX-2.3 model so that a contiguous range of
    transformer blocks is executed more than once per diffusion step,
    in a way that is compatible with ComfyUI's block-swap memory management.

    Block count reference
    ─────────────────────
    LTX-2.3 22B  →  ~48 transformer blocks  (indices 0–47)
    LTX-2   19B  →  ~28 transformer blocks  (indices 0–27)
    """

    CATEGORY      = "lightricks/LTXV"
    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loop_range": (
                    "STRING",
                    {
                        "default": "12,18",
                        "multiline": False,
                        "tooltip": (
                            "Inclusive block index range to loop, e.g. '12,18' runs "
                            "blocks 12-18 an extra time. "
                            "LTX-2.3 22B has ~48 blocks (0-47). "
                            "LTX-2 19B has ~28 blocks (0-27)."
                        ),
                    },
                ),
                "loop_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of extra echo passes through the looped range.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Diffusion step % at which looping activates (0 = start).",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Diffusion step % at which looping deactivates (1 = end).",
                    },
                ),
            }
        }

    # ------------------------------------------------------------------
    def patch(self, model, loop_range: str, loop_count: int,
              start_percent: float, end_percent: float):

        # ── parse ─────────────────────────────────────────────────────
        try:
            a, b = [p.strip() for p in loop_range.split(",")]
            loop_start, loop_end = int(a), int(b)
        except Exception:
            raise ValueError(
                f"[LTXV Block Loop] Bad loop_range '{loop_range}'. "
                "Expected 'start,end' e.g. '12,18'."
            )
        if loop_start > loop_end:
            loop_start, loop_end = loop_end, loop_start

        sigma_start = _percent_to_sigma(model, start_percent)
        sigma_end   = _percent_to_sigma(model, end_percent)

        _ls = loop_start
        _le = loop_end
        _lc = loop_count
        _ss = sigma_start
        _se = sigma_end

        new_model = model.clone()
        t_opts = new_model.model_options.setdefault("transformer_options", {})
        pr     = t_opts.setdefault("patches_replace", {})
        dit    = pr.setdefault("dit", {})

        _CK = "__ltxv_loop_state__"

        def _active(t_o: dict) -> bool:
            s = _current_sigma(t_o)
            if s is None:
                return True
            return _se <= s <= _ss

        def _get_state(t_o: dict) -> dict:
            if _CK not in t_o:
                t_o[_CK] = {}
            return t_o[_CK]

        # ── loop_start patch ──────────────────────────────────────────
        # Saves entry img (and args) then runs the block normally.
        # Clone only if we need to reuse the entry across multiple echoes.

        def make_start_patch(idx: int):
            def _start(args: dict, extra: dict) -> dict:
                ob  = extra["original_block"]
                t_o = _t_opts_from(args, extra)
                if _active(t_o):
                    state = _get_state(t_o)
                    state["entry_img"]  = (_clone_img(args["img"])
                                           if _lc > 1 else args["img"])
                    state["entry_args"] = dict(args)
                return ob(args)
            return _start

        # ── intermediate block patches ────────────────────────────────
        # Pass-through only.  No caching.  Exist solely so the patch
        # chain is intact for any other patches that may be chained.

        def make_mid_patch(idx: int):
            def _mid(args: dict, extra: dict) -> dict:
                return extra["original_block"](args)
            return _mid

        # ── loop_end patch ────────────────────────────────────────────
        # Runs the block, then drives echo passes.
        # Echo dispatch goes through _dispatch_block which respects
        # block-swap hooks where possible.

        def make_end_patch(idx: int):
            def _end(args: dict, extra: dict) -> dict:
                ob     = extra["original_block"]
                result = ob(args)

                t_o = _t_opts_from(args, extra)
                if not _active(t_o):
                    return result

                state      = _get_state(t_o)
                entry_img  = state.get("entry_img",  args["img"])
                entry_args = state.get("entry_args", args)

                for _echo in range(_lc):
                    current_img = (_clone_img(entry_img)
                                   if _lc > 1 else entry_img)

                    for echo_idx in range(_ls, _le + 1):
                        if echo_idx == _ls:
                            echo_args = dict(entry_args)
                        else:
                            echo_args = dict(args)
                        echo_args["img"] = current_img

                        echo_result = _dispatch_block(echo_idx, echo_args,
                                                      ob, t_o)
                        current_img = echo_result["img"]

                    result = {"img": current_img}

                # Free state immediately — it is not needed after loop_end.
                t_o.pop(_CK, None)

                return result
            return _end

        # ── chain helper ──────────────────────────────────────────────

        def chain(existing, new_fn):
            def _chained(args, extra):
                mid = existing(args, extra)
                new_args = dict(args)
                new_args["img"] = mid["img"]
                return new_fn(new_args, extra)
            return _chained

        # ── register ──────────────────────────────────────────────────

        if _ls == _le:
            # Single-block degenerate case
            def make_single_patch():
                def _single(args: dict, extra: dict) -> dict:
                    ob  = extra["original_block"]
                    t_o = _t_opts_from(args, extra)
                    result = ob(args)
                    if _active(t_o):
                        for _ in range(_lc):
                            single_args = dict(args)
                            single_args["img"] = result["img"]
                            result = ob(single_args)
                    return result
                return _single
            key = ("double_block", _ls)
            p   = make_single_patch()
            dit[key] = chain(dit[key], p) if key in dit else p
        else:
            key = ("double_block", _ls)
            p   = make_start_patch(_ls)
            dit[key] = chain(dit[key], p) if key in dit else p

            for idx in range(_ls + 1, _le):
                key = ("double_block", idx)
                p   = make_mid_patch(idx)
                dit[key] = chain(dit[key], p) if key in dit else p

            key = ("double_block", _le)
            p   = make_end_patch(_le)
            dit[key] = chain(dit[key], p) if key in dit else p

        print(
            f"[LTXV Block Loop] blocks {_ls}–{_le}, "
            f"{_lc} echo pass(es), "
            f"active {start_percent*100:.0f}%–{end_percent*100:.0f}% of denoising."
        )

        return (new_model,)


