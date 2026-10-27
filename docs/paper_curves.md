# A/B/C measured curve experiments

These tools start from the original, trusted MaGS checkpoint and optimize direct
Gaussian parameters. They do not change `train.py` or overwrite the source model.
All plots use measurements from this machine, not estimated initial scores.
Default numbers below are experiment settings, not expected results.

## Run on the server

Use the existing working MaGS CUDA environment. The extra plotting dependency is
`matplotlib`; planning uses the project's existing `omegaconf`. C also needs the
existing evaluator dependencies `lpips`, `torchvision`, and `scikit-image`.
Do not upgrade PyTorch/CUDA just to run these tools.

```bash
# Run from the repository root in your working CUDA environment.
python -m pip install matplotlib

# Writes the exact plan without loading models or using CUDA.
python tools/run_paper_curves.py --output outputs/paper-curves --groups A B C

# Engineering check: seven small jobs, one seed, NOT paper results.
python tools/run_paper_curves.py --output outputs/paper-curves-smoke --smoke --execute
python tools/plot_paper_curves.py outputs/paper-curves-smoke

# Full sweep: 99 isolated jobs, three seeds. A may OOM on a small GPU.
python tools/run_paper_curves.py --output outputs/paper-curves --groups A B C --execute --resume
python tools/plot_paper_curves.py outputs/paper-curves
```

Defaults use the relative paths in `config/style/split_vgg_geometry.yaml`, reference
`dataset/styles/2.jpg`, the 40000-iteration Jumping Jacks checkpoint, and 200
training times. The worker rejects a missing sequence or a nonsquare camera.
Update paths without editing Python, using the same overrides for planning and
execution. For example:

```bash
python tools/run_paper_curves.py --output outputs/curves-B-style6 --groups B --execute \
  --seeds 1234 \
  --set optim.gaussian_style_update.style_path=dataset/styles/6.jpg \
        optim.gaussian_style_update.appearance.vgg_weights_path=models/vgg16-397923af.pth
```

The VGG path above is an example; use an actual torchvision VGG16 checkpoint.
VGG, LPIPS AlexNet, and RAFT weights are loaded from their existing Torch cache
by default. Nothing downloads automatically. For C, configure local paths when
not cached:

```bash
python tools/run_paper_curves.py --output outputs/curves-C --groups C --execute \
  --seeds 1234 \
  --set evaluation.lpips.alexnet_weights=models/alexnet-owt-7be5be79.pth \
        evaluation.raft.weights=models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
```

These are paths to files you must already have. C checks LPIPS and RAFT loading
before training. ArtScore/ArtFID are not required for A/B/C. The general evaluator
can mark unrelated metrics unavailable; that does not invalidate a valid C LPIPS.

## A: temporal window scaling

- Frames per window: 2, 4, 8, 12, 16. RGB, diffusion, and VGG maximum input sizes:
  256, 384, 512. VGG uses the method's aspect-preserving foreground crop.
- Each job is a fresh process, with 10 warmup updates and 50 measured updates.
  All five Gaussian tensors are active from the start. No deformer, opacity
  optimization, densification, or topology changes.
- An update renders the complete window, computes both gradient branches, clips
  gradients, and steps both optimizers. CUDA synchronization surrounds the
  wall-clock measurement. It includes CPU/GPU teacher offloading when enabled.
  Loading, reference caching, metric evaluation, checkpoint writing, and log
  writing are excluded.
- Memory is PyTorch peak **allocated** bytes divided by 1e9, not total GPU use
  reported by `nvidia-smi`. Peak reserved GB is also recorded. The plotted time is
  the mean over measured updates per seed; memory is the maximum over those
  updates per seed. Across-seed bands are sample standard deviations.
- OOM is recorded as `oom`. No smaller-resolution fallback, fabricated value,
  zero, extrapolation, or line across the missing A point is used.

## B: optimization order

Use two-frame windows and 256 inputs. All schedules start from exactly the same
checkpoint and match **2500 SH updates and 2000 geometry updates**:

| Schedule | First phase | Joint phase | Remaining phase |
| --- | --- | --- | --- |
| Simultaneous | none | 2000 SH+geometry | 500 SH |
| Geometry-first | 500 geometry | 1500 SH+geometry | 1000 SH |
| Appearance-first | 500 SH | 2000 SH+geometry | none |

The simultaneous tail compensates for unequal group budgets; it must be reported
in the paper rather than described as joint optimization at every step. To study
strictly simultaneous optimization, set equal group budgets for all three runs.

Progress is `100 * (SH updates + geometry updates) / 4500`, not frame visits or
wall time. Reported x coordinates are actual progress at evaluation checkpoints.
CM is evaluated initially and approximately every 5%, on 20 evenly spaced times
with the same fixed camera and GT-derived foreground boxes for every schedule.
The RGB-uv histogram Hellinger distance follows the existing paper evaluator;
the reference style histogram is fixed. CM is an evaluation metric, not a new
training loss. Per-frame original cameras are used for optimization.

Learning rates are DC 0.002, SH-rest 0.0002, UVW 0.0001, log-scale/rotation
0.00005. SH and geometry gradients are clipped separately at 1.0. VGG supervises
SH only; video CSD supervises geometry only. Opacity and the mesh sequence stay
fixed. A step is committed only after both gradient groups pass finite checks.

## C: limited temporal coverage

- Coverage 1/5/10/20/40% of 200 times means 2/10/20/40/80 distinct optimized
  frames, grouped into 1/5/10/20/40 disjoint two-frame windows. Windows are
  nested within a seed, and exactly identical across the three variants.
- Fixed held-out indices are `[80,100)` and `[160,180)` (zero-based, exclusive
  upper bounds). These 40 times never receive a style optimizer update. Sampling
  visits every declared training window before starting the next shuffled epoch.
- Evaluate at the same fixed camera, gap 2 within each held-out block, giving
  36 candidate pairs. Source-only bidirectional RAFT flow, forward/backward
  validity, foreground masks, and erosion define alignment. Compute masked
  LPIPS on the aligned styled frames, averaged over both valid directions.
  Exported bundles record timestamps, masks, valid/excluded pairs, and source
  checkpoint/style hashes. Invalid pairs are never counted as zero.
  Plotting rejects comparisons whose valid temporal pair sets differ.
- This is propagation over **known motion**, with held-out *style optimization*
  times. Original reconstruction pretraining may have seen the whole sequence.
  Full-sequence GT masks/depth establish a shared crop and depth normalization;
  source RGB diffusion conditions are cached only for optimized times.

| Variant | SH gradient | Geometry gradient | Temporal teacher |
| --- | --- | --- | --- |
| VGG on SH and geometry | VGG | same VGG objective | none |
| w/o MotionAdapter | VGG | CSD with IP-Adapter + ControlNet | original 2D SDXL UNet |
| Ours | VGG | CSD with IP-Adapter + ControlNet | AnimateDiff UNetMotion |

The second variant genuinely omits MotionAdapter weights and temporal modules;
it is not a one-frame window or a zero motion scale. It retains the full method's
condition pooling, frame order, noise, and depth controls. All variants use the
same appearance-first budgets and optimize the same Gaussian parameter groups.

## Outputs and restart behavior

Each output directory contains `plan.json`, `results.csv`, and one directory per
job. Each job keeps `run.log`, `status.json`, `manifest.json`, `updates.jsonl`,
`measurements.jsonl`, and its effective configuration. B/C also save styled
Gaussian checkpoints every 500 window updates and at completion. C exports
`paper_eval/iteration_*/` for independent metric recomputation.

`--resume` skips completed jobs and **restarts failed jobs from the original
checkpoint**. Prior incomplete artifacts are renamed with `.previous` instead
of mixed into the new trial. This is sweep-level recovery, not mid-job optimizer
resume. A changed plan requires a new output directory. Checkpoints remain
separate from the pretrained model and are not claimed to support the legacy
`train.py` controller's resume format.

Plotting writes `plots/A_temporal_window_scaling.{pdf,png}`, corresponding B/C
files, summary CSV/JSON, and warnings. It only includes completed finite trials,
aggregates within a seed before computing across-seed means/std, and does not
smooth curves. One seed has no uncertainty band. Smoke figures are visibly
marked and must not be inserted as experimental evidence.

These defaults define one scene-style experiment. Repeat on additional
scene-style combinations to support dataset-wide claims; seeds alone do not
measure variation across scenes or styles.

## Checks

```bash
python -m unittest discover -s tests -p 'test_paper_curve*.py'
python -m unittest discover -s tests -p 'test_style_split.py'
```

The CPU tests cover budgets, nested coverage, disjoint held-out sets, missing
measurements, seed aggregation, gradient routing, and framewise UNet condition
ordering. Real model loading, CUDA extension rendering/backward, timing, and
memory still require the server smoke test.
