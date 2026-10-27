# Direct Gaussian style training

For VGG-only SH updates plus diffusion-only geometry updates, see
[split supervision](split_vgg_geometry_style.md). The profile on this page
retains the earlier diffusion-only objective for comparison.

This mode optimizes the Gaussian parameters themselves, not additive style
residuals. Known per-frame meshes drive the Gaussian binding. The loaded
deformer is frozen and is never called during training or evaluation.

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/direct_gaussian.yaml
```

The example uses relative checkpoint/model paths and style image 2. These files
are not bundled. Override paths with the usual dotlist flags as needed.

- Iterations 1-1000 optimize `_features_dc` and `_features_rest`.
- Starting at 1001, `_uvw`, `_scaling`, and `_rotation` train together with SH.
- `_opacity`, mesh features, face IDs, GT meshes, and diffusion weights stay fixed.
- Densification, pruning, opacity resets, and the ordinary RGB reconstruction
  loss do not run. The objective is the existing CSD surrogate only.
- Geometry uses absolute mesh-local parameters. Existing scale clamping and
  quaternion normalization still apply, but there are no residual amplitude
  bounds or residual regularization terms.
- VAE encoding and CSD arithmetic remain float32, with finite-value checks.

The two-frame video window performs one optimizer update after both renders.
Set `direct_geometry.start_iter` to the start of a complete window. A save
interval falling inside a window is deferred until that window completes.

## Separate outputs

Every invocation creates a fresh directory such as:

```text
outputs/jumpingjacks-direct-gaussian-style/direct-YYYYMMDD-HHMMSS-<unique>/
  cfg_args.yaml
  style_progress/initial_no_deformer.png
  style_progress/iter_00500_view_4_styled.png
  checkpoints/iteration_1000.pth
```

The input experiment directory and its descendants cannot be selected as output.
Existing checkpoint files are never overwritten in direct mode. The original
checkpoint is read only; `.pth` outputs contain the updated Gaussian state,
binding IDs, direct optimizer/schedule state, and source provenance. They do not
contain `gaussian_style_delta` or `gaussian_style_geometry_residual`.

`xyz` in a checkpoint is a frame snapshot; `_uvw` and the known mesh sequence
remain necessary for dynamic replay. The deformer weights are retained for
checkpoint compatibility, but direct-mode replay must keep the deformer off.

## Resume and baseline

Set `gaussian_style_update.start_checkpoint` to a direct-style checkpoint and
choose a different output root. The iteration counter and optimizer resume;
keep the original SH/geometry schedule and video-window size. Increase
`optim.iterations` when continuing beyond the original run length. A residual
style checkpoint is rejected rather than silently dropping its edits.

`initial_no_deformer.png` shows the loaded model before new updates, with the
deformer off. It can differ from the original pretrained rendering because
the pretrained model was reconstructed jointly with deformer corrections.
