# ComfyUI-Hillobar

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Everything
registers under a single top-level **Hillobar** menu category.

## Nodes

### MiniMax H3 Progressive Sampler (staged)

Progressive-resolution sampling for MiniMax H3. Early flow-matching steps only
settle coarse structure, so they do not need full token density — this node runs
the first part of the schedule on a smaller video latent grid, upscales the x0
estimate at each stage boundary, and finishes at the target resolution.

Video tokens scale with `latent_t * (h // 2) * (w // 2)`, so a 0.5x spatial scale
is roughly 4x fewer tokens: about 16x cheaper attention and 4x cheaper MLP for the
steps that run small. End to end the win is bounded by how much of the schedule
runs small — 55% of steps at 0.5x lands near 1.6–2x, not more.

Each stage is its own `guider.sample()` call, so conds, the packed layout,
`latent_shapes` and the VRAM reservation are rebuilt per stage. No ComfyUI
internals are mutated.

**Inputs** — `guider`, `sampler`, `sigmas`, `latent_image`, `noise_seed`,
`schedule`, `upscale_method`, `verbose`.
**Outputs** — `output`, `denoised_output` (both `LATENT`).

Feed it a latent **already at the final resolution**; it shrinks internally. That
keeps the VRAM estimate an over-reservation rather than an under-reservation.

#### schedule

Comma-separated `scale:end_percent` stages, e.g. the default `0.5:0.55, 1.0:1.0`
— run 55% of the steps at half the spatial latent grid, then finish at full size.
`scale` is the spatial latent scale, `end_percent` is where that stage ends as a
fraction of the step count. The schedule must end with `1.0:1.0`. Stages may not
shrink the grid mid-run. Use `1.0:1.0` alone as an A/B baseline.

#### Supported conditioning

t2va and ref2va. Reference blocks carry their own `latent_h` / `latent_w`, so they
stay self-consistent when the target grid changes.

Not supported: fl2va keyframes (their cond rows share the target spatial grid
while the cond latents stay full size) and noise masks (packed against the input
latent shape, so they cannot follow a resolution change). Both raise a clear
error rather than producing a silently wrong result.

## Installation

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. No dependencies beyond
what ComfyUI already installs.

## Structure

```
ComfyUI-Hillobar/
├── __init__.py                            # re-exports the node mappings
├── nodes/
│   ├── __init__.py                        # merges each module's mappings
│   └── minimax_h3_progressive_sampler.py
├── pyproject.toml                         # Comfy Registry metadata
├── requirements.txt
├── LICENSE
└── README.md
```
