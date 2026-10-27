# BindStyle

**Single-image dynamic Gaussian stylization via mesh-bound style propagation**

Maintainer: [yongangYu](https://github.com/yongangYu).

This repository provides the BindStyle research-code snapshot. It does not
include datasets, model weights, experiment outputs, or the development
repository's history. This separately maintained public edition does not alter
the review snapshot. See the dependency and licensing limitations below before
running or redistributing the code.

## Method and entry point

The main entry point is `train.py`, with
`config/style/split_vgg_geometry.yaml` loaded last.

- A pretrained mesh-bound Gaussian model and the known per-frame meshes provide
  the dynamic scene. The selected style preset never calls the frozen deformer.
- Frozen VGG16 nearest-neighbor feature matching updates Gaussian SH appearance.
- SDXL, IP-Adapter, AnimateDiff, and depth ControlNet provide video CSD guidance
  for mesh-local coordinates, scales, and rotations. The two losses have
  separate parameter routes and optimizers.
- Gaussian RGB renders carry the optimization gradients. nvdiffrast supplies
  detached GT-mesh depth conditions, not the styled image being optimized.
- The preset first updates SH, then enables geometry at frame-iteration 1001.
  With two frames per window, 5000 frame-iterations give 2500 appearance updates
  and 2000 geometry updates.

See [the method configuration and gradient routing](docs/split_vgg_geometry_style.md).
The earlier diffusion-only comparison is documented
[separately](docs/direct_gaussian_style.md).

## Environment

Use the working Linux/CUDA training environment first. The added requirement
files are dependency inventories, not an exact lock of the server environment.
Do not replace a working PyTorch/CUDA installation just to prepare a release.

Install a compatible PyTorch/torchvision pair and compiled extensions for that
pair. Inspect any dependency changes before accepting them:

```bash
python -m pip install --dry-run -r requirements-style.txt
python -m pip install -r requirements-style.txt
# Optional evaluation and plotting:
python -m pip install -r requirements-evaluation.txt
```

The CUDA extensions are not bundled. Obtain their source and license from the
upstream repositories, then install in the same environment:

```bash
git clone --recursive https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
git clone --recursive https://github.com/NVlabs/nvdiffrast.git submodules/nvdiffrast
python -m pip install --no-build-isolation -e submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e submodules/nvdiffrast
```

For exact reproduction, record the versions and extension commits used by the
successful training environment; an arbitrary latest checkout is not a
validated substitute. The original `.gitmodules` is retained for provenance,
but this snapshot has no Git submodule entries or vendored extension source.

## Data and local assets

Use corresponding RGB frames, cameras, and meshes with consistent topology.
Dataset and mesh-download references are in the
[original MaGS documentation](docs/UPSTREAM_MAGS.md). Prepare the following
paths relative to the repository root, or override them on the command line:

```text
dataset/dnerf/jumpingjacks/          # RGB frames, camera JSON, mesh sequence
dataset/styles/reference.jpg        # user-provided reference image
checkpoints/content/iteration_40000.pth
models/stable-diffusion-xl-base-1.0/
models/animatediff-motion-adapter-sdxl-beta/
models/controlnet-depth-sdxl-1.0/
models/IP-Adapter/
models/vgg16-397923af.pth
```

The Gaussian checkpoint must be compatible with the scene's binding/topology.
Only load trusted checkpoints. Model folders must contain their complete
Diffusers configuration and weights. The IP-Adapter preset expects
`sdxl_models/ip-adapter_sdxl.bin` and
`sdxl_models/image_encoder/` under `models/IP-Adapter/`.
Check the actual loader requirements before choosing a different adapter.

VGG uses torchvision VGG16 weights, not VGG19. Supply the path below or populate
the expected local Torch cache. Style-teacher downloads are disabled in the
preset. Ordinary training diagnostics may separately require cached LPIPS or
AlexNet assets.

## Run

From the repository root after preparing the assets:

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml \
  optim.gaussian_style_update.style_path=dataset/styles/reference.jpg \
  optim.gaussian_style_update.appearance.vgg_weights_path=models/vgg16-397923af.pth
```

The default preset limits training/testing to the first ten frames for a pilot
run. Do not label this as full-sequence evaluation. The supplied 200-time
evaluation protocol explicitly removes the limit and validates the sequence:

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml,config/evaluation/paper.yaml \
  optim.gaussian_style_update.style_path=dataset/styles/reference.jpg \
  optim.gaussian_style_update.appearance.vgg_weights_path=models/vgg16-397923af.pth
```

Use the appropriate scene config and a matching pretrained checkpoint for other
scenes. For sequences with a different number of times, set and document a
matching evaluation protocol rather than suppressing its validation.

Outputs are written to a separate run directory. The input checkpoint is not
overwritten. See [evaluation](docs/paper_evaluation.md) and
[measured A/B/C sweeps](docs/paper_curves.md) for output schemas, metrics,
budget matching, and limitations.

## Validation and current limitations

```bash
python -m unittest discover -s tests -p 'test_style_split.py'
python -m unittest discover -s tests -p 'test_paper_curve*.py'
```

CPU unit tests are not a replacement for loading the actual teachers and running
a CUDA rendering/backward smoke test. No fresh GPU reproduction is claimed for
this prepared snapshot.

The exported snapshot currently omits the server's `arguments` module, which
`scene/__init__.py` imports. Restore that exact module from the working source
before running this as a standalone checkout; this preparation does not replace
it with guessed code. The optional InstantAvatar loader also references
`utils.smplx_utils`, which is not included. The documented D-NeRF/DG-Mesh setup
does not use that optional loader.

## Attribution and release status

This work builds on MaGS and Gaussian Splatting. Their attribution remains
intact in the source and the [upstream README](docs/UPSTREAM_MAGS.md).
See [third-party notices](THIRD_PARTY_NOTICES.md) and the inherited
[Gaussian Splatting license](LICENSE.md). The MaGS-specific redistribution
terms still require confirmation; this is not a blanket MIT-licensed project.

The public edition retains the upstream notices and known limitations of the
source snapshot. Making the repository public does not resolve the outstanding
MaGS-specific licensing question or establish a complete standalone
reproduction package. Private preparation records, credentials, and experiment
assets are not part of this repository.
