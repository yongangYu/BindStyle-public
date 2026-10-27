# 论文评估与服务器保存

本页记录源码实现的评估协议，在最终优化器更新和 checkpoint 保存之后执行，不修改 VGG/CSD 训练损失或梯度路径。当前支持 `parameterization=direct`、已知 GT mesh、无 deformer 前向的版本。所有评价网络冻结，评价模型只读取本地权重，不在训练结束时自动联网下载。

## 运行与保存

当前 `config/style/split_vgg_geometry.yaml` 已开启训练后评估，但保留原来的十帧训练设置。这样的结果明确标为 `pilot`，长间隔帧对可能不存在，不能混入论文完整序列结果。

200 帧协议在项目根目录下运行下面的命令。最后一个配置文件必须放在风格配置之后；它取消十帧限制并检查实际是否有 200 个不同的时间点。其他序列长度需显式修改协议设置，不能将截断后的序列视为完整序列。保留当前预设的风格图 `2.jpg`，也可以显式修改 `style_path`。5000 仍是原有训练的 frame-iterations，不是 5000 个完整视频窗口更新。

```bash
python train.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml,config/evaluation/paper.yaml \
  model.model_path=outputs/jumpingjacks-paper-eval \
  optim.gaussian_style_update.style_path=dataset/styles/2.jpg
```

结果随本次实验保存在服务器磁盘，而不是上传到另一个服务。日志会打印实际路径，例如

```text
outputs/jumpingjacks-paper-eval/<独立运行目录>/
  checkpoints/...
  paper_eval/iteration_005000/
    training_config.json
    manifest.json
    status.json
    summary.json
    summary.csv
    per_image_and_pair_metrics.csv
    pair_directions.csv
    user_study_template.csv
    style.png
    source/*.png
    styled/*.png
    time_*.npz
    view_*.npz
    correspondence/*.npz
```

PNG 用于查看；指标使用 NPZ 中的浮点图像。每次运行保存源 checkpoint 与网格序列指纹、风格图指纹、实际相机/时间、对应掩码、评价模型 SHA256、软件版本、有效面积和排除数量。RGB 统一截断到 `[0,1]`，另外记录渲染越界比例。原始模型不会被评估器覆盖。

默认 `evaluation.device=cpu`，避免与显存占用较大的视频扩散教师竞争；全序列 RAFT 评估可能耗时较长。只需先导出时添加 `evaluation.compute_metrics=false`。训练退出后，可以在 GPU 上单独重新计算

```bash
python tools/evaluate_paper.py \
  outputs/你的实验/实际运行目录/paper_eval/iteration_005000 \
  --device cuda
```

评估失败不会删除训练 checkpoint。错误保存在 `paper_eval/failure.json` 和 traceback 文件。指标缺失、无有效对应、非有限值分别记录状态，绝不填成零。重新计算不会覆盖已填写的人工评分 CSV。正式汇总前应检查 `status.json`、`summary.csv` 和排除原因；`partial` 不表示全部指标已成功。

## 指标与固定协议

| 论文表格 | 保存的指标 | 当前计算方式 |
| --- | --- | --- |
| 表 1 | `view_narrow/wide.lpips, rmse` | 固定时间，GT mesh 提供双向表面对应与深度遮挡过滤；窄基线 `(0,10]` 度，宽基线 `[30,60]` 度 |
| 表 2 | `foreground.artscore`、汇总 `artfid` | ArtScore 按作者网络与预处理；ArtFID 在多场景、多风格集合上另算 |
| 表 3 | `temporal_short/long.lpips, rmse` | 固定相机，仅对共同的未风格化源渲染运行冻结 RAFT，直接估计间隔 2/20 帧的双向流 |
| 表 4 | `foreground.cm` 与时间一致性指标 | HistoGAN RGB-uv 色彩直方图的 Hellinger 距离；不是训练 CSD loss |
| 表 5 | `foreground.lpips_gray`、`geometry.delta_pos/delta_scale` | 与同一未风格化基底、相同 primitive ID 比较；位移除以源网格包围盒对角线，尺度是局部 log-scale 变化 |
| 表 6 | 同样的 CM/时间指标 | 当前没有执行独立的未优化时刻实验，保留 unavailable；不能把本次训练用过的时刻当 held-out |
| 表 7 | 人工 `style/geometry/temporal` | 保存空白 1–5 分评分模板，不能自动生成用户评价 |

默认评价分辨率最长边 256；固定第一个时间点的相机，所有动态时间复用这个相机。视角评价选均匀分布的十个时间点，使用共同中心、共同相机半径的八个轨道角度，每个时间点每类最多 20 对。每对先双向平均，再在场景—风格组合内平均。非均匀时间保存实际归一化间隔，不伪造秒数。

前景 crop 使用源网格掩码及 10% padding，所有方法必须共用。完整图像与前景 crop 的单图指标分别保存。对应误差使用原渲染坐标，3 像素掩码腐蚀，至少 5% 源前景有效覆盖；LPIPS 先提取空间距离图再按掩码平均，不把背景清零冒充 masked LPIPS。过小的 LPIPS crop 报 unavailable，不悄悄改分辨率。几何编辑较大时，源网格/源光流对应不再精确，这是这类一致性指标的局限，需结合风格强度分析。

另存 source RGB LPIPS、PSNR、SSIM，CFSD 仅为可选诊断，默认关闭。这些指标只评价相对初始无 deformer 渲染的内容偏离，不能称为风格图保真度或原始数据集重建精度。当前未实现独立的动态 mesh warp 交叉验证、人工评分统计、匹配预算的其他优化顺序或 held-out 窗口采样；本次修改不宣称自动完成七张表的所有实验。

## 评价依赖与权重

保持已能训练的 Torch/CUDA 环境，不要为评价升级 Torch。需要 `lpips`、`scikit-image`。ArtFID 的集合汇总另需作者的 `art-fid`、`scipy` 和 `scikit-learn`；建议固定版本并保留实际安装记录，避免依赖安装意外改动训练环境。

AlexNet 与 RAFT 可以使用已有 Torch Hub 缓存，或者通过以下字段提供本地 checkpoint。ArtScore 必须额外准备作者发布的评价权重，不可使用普通 ImageNet ResNet50 代替。

```text
evaluation.lpips.alexnet_weights=/实际路径/alexnet-owt-7be5be79.pth
evaluation.raft.weights=/实际路径/raft_large_C_T_SKHT_V2-ff5fadd5.pth
evaluation.artscore.weights=/实际路径/ArtScore官方评价权重.pth
```

上面是字段示例，不是假定服务器已经存在这些文件。路径留空且缓存不存在时，相应指标会标记 unavailable，但 CM、几何变化量和不依赖缺失模型的指标仍会保留。可以补齐权重后重算，无须重新训练

```bash
python tools/evaluate_paper.py /实际实验目录/paper_eval/iteration_005000 \
  --device cpu \
  --alexnet-weights /实际路径/alexnet.pth \
  --raft-weights /实际路径/raft.pth \
  --artscore-weights /实际路径/artscore.pth
```

权重及实现来源为 [LPIPS](https://github.com/richzhang/PerceptualSimilarity)、[Torchvision RAFT](https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py)、[ArtScore](https://github.com/jchen175/ArtScore) 和 [ArtFID](https://github.com/matthias-wright/art-fid)。ArtScore 使用官方 README 中 `loss@listMLE_model@resnet50_denseLayer@True_batch_size@16_lr@0.0001_dropout@0.5_E_8.pth` 对应的结构。Art-Inception 权重见 [作者 Hugging Face 仓库](https://huggingface.co/matthias-wright/art_inception)。CM 按 [HistoGAN](https://github.com/mahmoudnafifi/HistoGAN) 的 RGB-uv 公式独立实现，固定 64 bins、inverse-quadratic 核、带宽 0.02、强度加权以及最长边超过 150 时缩放至 150×150 的规则。

## 多实验汇总

```bash
python tools/collect_paper_results.py \
  /实际实验A/paper_eval/iteration_005000 \
  /实际实验B/paper_eval/iteration_005000 \
  --output outputs/paper_results \
  --art-inception-weights /实际路径/art_inception.pth \
  --device cpu
```

汇总保存 `paper_results.csv` 和 `collection.json`。不同方法需用 `evaluation.method` 与 `evaluation.variant` 区分，例如 `StyleGaussian (2024)`。检查场景—风格集合、源 mesh/相机/裁剪/帧对、评价网络权重及设置一致；先平均重复运行，再等权平均组合。未估计置信区间，重复帧也不能当成独立种子。

当前导出器只对 MaGS 直接参数风格训练提供自动接入。外部 baseline 需要在同一个 manifest 下导入其真实渲染，不能只改方法名称就声称跑过 baseline。汇总器也不是 baseline 训练器。

ArtFID 使用 `art_fid` 模式而不是 `art_fid_inf`，保存分布 FID、内容 LPIPS 及乘积；每次运行均匀抽 20 个固定相机时间点。默认至少 20 张不同参考风格，并要求每个组合的重复运行数量相同。20 是本实验的样本多样性门槛，不是 ArtFID 的数学定义或统计可靠性保证；还需用不同样本规模验证敏感性。只有一张参考图时保持 unavailable，不重复同一张图伪造多风格数据集。

## 运行目录与结果共享

在安装好依赖的 CUDA 环境中从项目根目录运行上述命令。数据集、预训练模型、输出目录和服务器凭据均不随源码分发。分享评估结果前另行检查配置快照和 manifest 中的本地路径及身份信息；源码已匿名化不代表运行时生成的结果文件也已匿名化。
