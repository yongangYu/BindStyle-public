# Third-party code, weights, and licensing

This snapshot builds on [MaGS](https://github.com/wcwac/MaGS) and contains
code derived from [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
Existing copyright and attribution headers are retained without modification.
They identify upstream contributors and are separate from BindStyle authorship.

## Gaussian Splatting

`LICENSE.md` is an unmodified copy of the Gaussian Splatting license retrieved
from its official repository. It accompanies the inherited source files that
refer to that license. It is not a claim that Inria owns new BindStyle code,
nor a blanket license grant for unrelated third-party components.

Source: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

The upstream license restricts use to non-commercial research/evaluation and
requires retention of notices and the license when redistributing covered
code. Consult the full text, not this summary, for the applicable terms.

## MaGS

The original MaGS README is preserved at `docs/UPSTREAM_MAGS.md`.
The inspected source snapshot did not contain a standalone MaGS license file.
The license for Gaussian Splatting-derived files does not by itself establish
redistribution permission for every MaGS-specific contribution. Before public
or anonymous redistribution, confirm the applicable terms with the upstream
maintainers or supply an existing permission record. No MIT/Apache license is
assigned to the whole snapshot here.

## External dependencies and model weights

CUDA extensions, pretrained diffusion/vision weights, and datasets are not
bundled. Obtain them from their respective maintainers and retain their terms:

- [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [Diffusers](https://github.com/huggingface/diffusers) and
  [Transformers](https://github.com/huggingface/transformers)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [D-NeRF](https://github.com/albertpumarola/D-NeRF) and
  [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh)

Model-repository licenses may differ from the inference library license.
See the evaluation documentation for LPIPS, ArtScore, ArtFID, and HistoGAN
attributions. Public distribution must retain applicable third-party credits.
