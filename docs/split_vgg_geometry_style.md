# Split appearance and geometry style supervision

This opt-in mode uses one style image for two frozen teachers:

- VGG16 nearest-neighbor feature matching (NNFM) updates only Gaussian SH.
- SDXL + IP-Adapter + AnimateDiff + depth ControlNet video CSD updates only
  `_uvw`, `_scaling`, and `_rotation`.

The model remains mesh-bound. `_uvw` determines world-space `xyz` through the
known mesh sequence. The deformer is frozen and never called, and there are no
new style residual parameters, Poisson solver, densification, opacity updates,
or RGB reconstruction loss. Geometry guidance still uses Gaussian RGB renders;
nvdiffrast only produces detached GT depth conditions for ControlNet.

## Run

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml
```

The preset uses paths relative to the repository root, style image
`dataset/styles/2.jpg`, and a separate output root. Supply the pretrained
checkpoint and model weights described in the README; none are bundled.
Override paths with the existing dotlist syntax. For example:

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml \
  optim.gaussian_style_update.style_path=dataset/styles/1.jpg \
  optim.gaussian_style_update.appearance.vgg_weights_path=models/vgg16-397923af.pth
```

VGG uses the full torchvision VGG16 state dict, not VGG19 or a custom normalized
VGG checkpoint. The classifier is discarded after loading; only feature layers
remain. If `vgg_weights_path` is empty, the standard torch.hub VGG16 cache is
used. Downloads are disabled by default. To explicitly permit downloading the
official torchvision weights, set
`optim.gaussian_style_update.appearance.allow_download=true`.

## Schedule and gradient ownership

- Frame-iterations 1-1000: VGG updates DC and SH-rest; no diffusion predictions.
- Starting at 1001: VGG continues updating SH, while CSD updates all three
  geometry groups together.
- Every two frames form one video window. Both groups differentiate the same
  render snapshot, then step once after the entire window is complete. Thus
  5000 frame-iterations mean 2500 SH updates and 2000 geometry updates.
- `torch.autograd.grad` selects the parameters for each loss. A summed loss is
  logged, but is not jointly backpropagated into all trainable parameters.
- VGG activations are released before the video teacher forward. The default
  `appearance.offload_teacher=true` also moves the large denoisers to CPU
  between branches for small GPUs. This trades speed and host RAM for VRAM.
  Set it false on a GPU with sufficient memory; SH-only initialization still
  parks the denoisers until geometry guidance is used.
- VGG and diffusion weights stay frozen. Input derivatives through VGG and the
  float32 VAE remain enabled. SH and geometry have separate Adam optimizers and
  separate clipping thresholds. Invalid losses/gradients abort before stepping.

`appearance.loss_type` selects `nnfm` (default), `gram`, or `adain`. These are
alternatives, not a sum. NNFM uses block-3 VGG features and bounded-memory cosine
nearest-neighbor matching, averaged across selected layers and frames. The
fixed crop is the padded union of GT foreground masks across training frames;
it is not allowed to move with optimized Gaussians. No content loss is added.

`appearance.weight` controls the SH objective; `sds_weight` now controls only the
geometry objective in this mode. `direct_geometry.start_iter` must begin a
complete video window. To run a VGG-only ablation, set
`direct_geometry.enabled=false`; geometry weights cannot accidentally train SH.

## Conditions and checkpoints

The CSD source is a fixed, no-deformer render of the original pretrained model,
not the GT RGB and not a render regenerated from a stylized resume checkpoint.
These CPU images and a camera/mesh/mask signature are stored in every checkpoint.
Resume rejects changes to the reference inputs, background, style image, VGG
feature selection/crop, and the SH/geometry schedule.

The new depth mode interpolates camera-space depth, converts it to relative
inverse depth (near white, far black), and applies one shared normalization
range across the sequence. The old `ndc` mode remains the default for old runs.
Inspect the depth overlays to verify pixel alignment on the actual GPU renderer.

Every invocation uses a new, separate output directory. Checkpoints store the
updated Gaussian base parameters, both optimizer states, schedule, and fixed
reference. The original checkpoint is never overwritten. Resume a split-mode
checkpoint using `start_checkpoint`, preserve the schedule, and increase
`optim.iterations`. Old diffusion-only checkpoints remain usable in the old
mode; switching objectives requires starting from the original MaGS checkpoint
instead of silently reusing incompatible optimizer/reference state.

## Diagnostics

- `style_progress/source_reference/`: initial no-deformer source RGB frames.
- `style_progress/control_reference/`: normalized depth and RGB/depth overlays.
- `style_progress/iter_*_styled.png`: appearance progress and video windows.
- TensorBoard: VGG loss, CSD surrogate, branch RGB gradient norms, parameter
  gradient norms, RGB saturation fraction, and actual per-step parameter RMS
  changes (`gaussian_style_update_rms/*`).

An incomplete window reports zero loss because no update has been performed.
CSD's signed surrogate is not a perceptual style-quality metric. Separating
parameter gradients does not prove that the teacher supplies useful 3D shape
directions: compare VGG-only and split runs with identical seeds and inputs.
CPU tests verify routing/numerics/resume, not pretrained teacher quality or CUDA
rasterizer behavior; GPU rendering and visual style validation are still needed.
