"""Progressive-resolution sampling for MiniMax H3 (staged).

Early flow-matching steps only settle coarse structure, so they do not need
full token density. This node runs the first part of the schedule on a smaller
video latent grid, then upscales the x0 estimate and finishes at the target
resolution.

H3 suits this unusually well. ``_axis_from_sqrt_area`` in
``comfy/ldm/minimax/model.py`` builds each spatial RoPE axis as

    ((arange(n) / n) * ratio + (1 - ratio) / 2) * 32,  ratio = dim / sqrt(h * w)

which is an interval centred on 16 with half-width ``ratio * 16`` -- set by the
aspect ratio alone and *independent of the token count* (a square grid gives
exactly [0, 32) at any resolution). Halving the grid therefore changes token
density inside an unchanged interval rather than shrinking the positional
extent, which is position interpolation by construction. Mochi uses the same
sqrt-area trick for the same reason. ``MiniMaxH3Model._forward`` also already
rebuilds ``PackedLayout`` whenever the shape signature changes.

Video tokens scale with ``latent_t * (h // 2) * (w // 2)``, so a 0.5x spatial
scale is roughly 4x fewer tokens: about 16x cheaper attention and 4x cheaper
MLP for the steps that run small. End to end the win is bounded by how much of
the schedule runs small -- 55% of steps at 0.5x lands near 1.6-2x, not more.

Each stage is its own ``guider.sample()`` call, so conds, the packed layout,
``latent_shapes`` and the VRAM reservation are all rebuilt per stage. No
ComfyUI internals are mutated.

Takes a latent already at the *target* resolution and shrinks internally. That
keeps ``memory_required`` (evaluated once from the initial shape) an
over-reservation rather than an under-reservation, and keeps the final output
consistent with the ``latent_shapes`` ``CFGGuider.sample`` captured on entry.

Supported: t2va and ref2va. Reference blocks carry their own ``latent_h`` /
``latent_w`` so they stay self-consistent when the target grid changes.
Rejected: fl2va keyframes, whose cond rows share the *target* spatial grid
while their cond latents stay full size.
"""

import logging
import math
import time

import torch

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

UPSCALE_METHODS = ["bicubic", "bilinear", "area", "nearest-exact", "bislerp"]
DEFAULT_SCHEDULE = "0.5:0.55, 1.0:1.0"

SCHEDULE_TOOLTIP = (
    "Comma-separated 'scale:end_percent' stages, e.g. '0.5:0.55, 1.0:1.0'. "
    "scale is the spatial latent scale, end_percent is where that stage ends "
    "as a fraction of the step count. Must end with '1.0:1.0' so sampling "
    "finishes at the target resolution. Use '1.0:1.0' alone as an A/B baseline."
)


# --------------------------------------------------------------------------
# schedule helpers
# --------------------------------------------------------------------------

def parse_schedule(text):
    """'0.5:0.55, 1.0:1.0' -> [(0.5, 0.55), (1.0, 1.0)]

    Stages may be separated by commas, whitespace, or both.
    """
    stages = []
    for chunk in text.replace(",", " ").split():
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("schedule entry %r is not 'scale:end_percent'" % chunk)
        scale_s, pct_s = chunk.split(":", 1)
        try:
            stages.append((float(scale_s), float(pct_s)))
        except ValueError:
            raise ValueError("schedule entry %r is not numeric" % chunk)

    if not stages:
        raise ValueError("schedule is empty")

    last_pct, last_scale = 0.0, 0.0
    for scale, pct in stages:
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale %s must be in (0, 1]" % scale)
        if not 0.0 < pct <= 1.0:
            raise ValueError("end_percent %s must be in (0, 1]" % pct)
        if pct < last_pct:
            raise ValueError("end_percent values must ascend")
        if scale < last_scale:
            raise ValueError(
                "scale %s follows %s: stages must not shrink the grid mid-run, which would "
                "discard detail already resolved" % (scale, last_scale)
            )
        last_pct, last_scale = pct, scale

    if stages[-1][0] != 1.0 or stages[-1][1] != 1.0:
        raise ValueError(
            "the last stage must be '1.0:1.0' so sampling ends at the target resolution"
        )
    return stages


def stage_dims(base_h, base_w, scale):
    """Latent dims for a stage, snapped to even so the DiT's 2x2 patch is exact."""
    if scale >= 1.0:
        return base_h, base_w
    h = max(2, int(round(base_h * scale / 2.0)) * 2)
    w = max(2, int(round(base_w * scale / 2.0)) * 2)
    return h, w


def split_sigmas(sigmas, stages):
    """[(scale, sigma_slice), ...] with at least one step per stage."""
    steps = sigmas.shape[-1] - 1
    n = len(stages)
    if steps < n:
        raise ValueError("need at least %d steps for %d stages, got %d" % (n, n, steps))

    bounds = []
    for i, (_, pct) in enumerate(stages[:-1]):
        b = int(round(pct * steps))
        b = max(i + 1, min(steps - (n - 1 - i), b))
        if bounds:
            b = max(b, bounds[-1] + 1)
        bounds.append(b)
    bounds.append(steps)

    plan, start = [], 0
    for (scale, _), end in zip(stages, bounds):
        plan.append((scale, sigmas[start:end + 1]))
        start = end
    return plan


def resize_video(video, h, w, method):
    if video.shape[-2] == h and video.shape[-1] == w:
        return video
    return comfy.utils.common_upscale(video, w, h, method, "disabled")


def video_tokens(shape):
    """DiT token count for a [B, C, T, h, w] video latent."""
    return int(shape[-3]) * (int(shape[-2]) // 2) * (int(shape[-1]) // 2)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def _iter_conds(guider):
    for attr in ("original_conds", "conds"):
        conds = getattr(guider, attr, None)
        if not conds:
            continue
        for group in conds.values():
            for cond in group or []:
                yield cond


def pin_memory_estimate(model_options, full_noise_shape):
    """Make every stage plan VRAM for the largest stage.

    ``prepare_sampling`` sizes the weights-in-VRAM split from ``noise_shape``
    (comfy/sampler_helpers.py estimate_memory). Staged sampling calls it once per
    stage with a *growing* shape, so the planner commits spare VRAM to weights
    during a small early stage and then has to evict them when the latent grows.
    On a ~19B model held partially resident that re-partition is real work, and
    it repeats at every boundary.

    Pinning the estimate to the final packed shape keeps the split stable, so the
    model is planned once and left alone.
    """
    def wrapper(executor, model, noise_shape, conds, **kwargs):
        return executor(model, full_noise_shape, conds, **kwargs)

    opts = comfy.model_patcher.create_model_options_clone(model_options)
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, wrapper, opts, is_model_options=True)
    return opts


def packed_shape(video_shape, audio_shape):
    """Flat [B, 1, N] shape pack_latents would produce for these two streams."""
    n = math.prod(tuple(video_shape)[1:]) + math.prod(tuple(audio_shape)[1:])
    return torch.Size([int(video_shape[0]), 1, int(n)])


def check_supported(model, guider):
    if not isinstance(model, comfy.model_base.MiniMaxH3):
        raise ValueError(
            "MiniMax H3 progressive sampling only supports MiniMax H3 models, got %s"
            % type(model).__name__
        )
    for cond in _iter_conds(guider):
        if cond.get("minimax_keyframes"):
            raise ValueError(
                "keyframe (fl2va) conditioning is not supported: keyframe cond rows share "
                "the target spatial grid while their cond latents stay full size. "
                "Use MiniMaxH3ReferenceToVideo (ref2va) or a text-only prompt."
            )
        payload = cond.get("model_conds", {}).get("minimax_payload")
        if payload is not None and isinstance(payload.cond, dict) and payload.cond.get("keyframes"):
            raise ValueError(
                "keyframe (fl2va) conditioning is not supported: keyframe cond rows share "
                "the target spatial grid while their cond latents stay full size. "
                "Use MiniMaxH3ReferenceToVideo (ref2va) or a text-only prompt."
            )


# --------------------------------------------------------------------------
# staged sampler
# --------------------------------------------------------------------------

class MiniMaxH3ProgressiveSampler:
    """Sample MiniMax H3 in resolution stages, upscaling the x0 estimate between them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "schedule": ("STRING", {"default": DEFAULT_SCHEDULE, "tooltip": SCHEDULE_TOOLTIP}),
                "upscale_method": (UPSCALE_METHODS, {"default": "bicubic"}),
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Log per-stage grid size, token count and elapsed seconds.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "Hillobar"
    DESCRIPTION = (
        "Progressive-resolution sampling for MiniMax H3. Runs early steps on a smaller "
        "video latent grid, upscales the x0 estimate at each stage boundary, and finishes "
        "at the target resolution. Feed a latent already at the final size."
    )

    def sample(self, guider, sampler, sigmas, latent_image, noise_seed,
               schedule, upscale_method, verbose):
        stages = parse_schedule(schedule)
        model = guider.model_patcher.model
        check_supported(model, guider)

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))

        if latent.get("noise_mask") is not None:
            raise ValueError(
                "a noise mask is packed against the input latent shape and cannot follow a "
                "resolution change; remove it or use a single-stage '1.0:1.0' schedule"
            )
        if not getattr(samples, "is_nested", False):
            raise ValueError(
                "expected a MiniMax H3 AV latent (nested video+audio pair) from "
                "EmptyMiniMaxH3LatentAV or MiniMaxH3ReferenceToVideo"
            )

        video, audio = samples.unbind()[0], samples.unbind()[1]
        base_h, base_w = int(video.shape[-2]), int(video.shape[-1])
        plan = split_sigmas(sigmas, stages)

        cur_v = resize_video(video, *stage_dims(base_h, base_w, plan[0][0]), method=upscale_method)
        cur_a = audio

        # plan VRAM once, for the largest stage, so growing the latent between
        # stages does not force the partial-load split to be recomputed
        orig_model_options = guider.model_options
        guider.model_options = pin_memory_estimate(
            orig_model_options, packed_shape(video.shape, audio.shape))
        try:
            out_samples, last_x0 = self._run_stages(
                guider, sampler, plan, latent, cur_v, cur_a, video, audio,
                base_h, base_w, noise_seed, upscale_method, verbose, model)
        finally:
            guider.model_options = orig_model_options

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = out_samples.to(comfy.model_management.intermediate_device())

        if last_x0 is not None:
            denoised = out.copy()
            denoised["samples"] = model.process_latent_out(last_x0.cpu())
        else:
            denoised = out
        return (out, denoised)

    def _run_stages(self, guider, sampler, plan, latent, cur_v, cur_a, video, audio,
                    base_h, base_w, noise_seed, upscale_method, verbose, model):
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out_samples = None
        last_x0 = None

        for idx, (scale, stage_sigmas) in enumerate(plan):
            nested = comfy.nested_tensor.NestedTensor((cur_v, cur_a))
            stage_latent = dict(latent)
            stage_latent["samples"] = nested

            noise = Noise_RandomNoise(noise_seed + idx).generate_noise(stage_latent)
            x0_capture = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, stage_sigmas.shape[-1] - 1, x0_capture)

            started = time.perf_counter()
            out_samples = guider.sample(
                noise, nested, sampler, stage_sigmas, denoise_mask=None,
                callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
            elapsed = time.perf_counter() - started

            if verbose:
                logging.info(
                    "MiniMaxH3ProgressiveSampler: stage %d/%d scale=%.3g grid=%dx%d "
                    "video_tokens=%d steps=%d elapsed=%.1fs",
                    idx + 1, len(plan), scale, cur_v.shape[-2], cur_v.shape[-1],
                    video_tokens(cur_v.shape), stage_sigmas.shape[-1] - 1, elapsed)

            last_x0 = self._nested_x0(x0_capture, nested)
            if idx == len(plan) - 1:
                break
            if last_x0 is None:
                raise RuntimeError(
                    "the sampler produced no x0 estimate, so the stage boundary cannot change "
                    "resolution; use a sampler that reports 'denoised' in its callback"
                )

            # undo the audio carry scale here; the next stage's process_latent_in re-applies it
            x0 = model.process_latent_out(last_x0)
            v0, a0 = x0.unbind()[0], x0.unbind()[1]
            next_h, next_w = stage_dims(base_h, base_w, plan[idx + 1][0])
            cur_v = resize_video(v0, next_h, next_w, upscale_method).to(video.dtype)
            cur_a = a0.to(audio.dtype)

        return out_samples, last_x0

    @staticmethod
    def _nested_x0(x0_capture, nested):
        x0 = x0_capture.get("x0")
        if x0 is None:
            return None
        if not getattr(x0, "is_nested", False):
            shapes = [t.shape for t in nested.unbind()]
            x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, shapes))
        return x0


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ProgressiveSampler": MiniMaxH3ProgressiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ProgressiveSampler": "MiniMax H3 Progressive Sampler (staged)",
}
