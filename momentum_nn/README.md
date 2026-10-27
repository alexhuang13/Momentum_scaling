# 神经网络上的 momentum stability 实验

> 这是较早的 ImageNet stability notebook，保留其独立协议与原实现。
> 新增的论文实验入口：[ImageNet 固定样本预算](../data_scaling/imagenet/README.md)
> 与 [CIFAR-10 稳定性](../stability/cifar10/README.md)。
> 下文历史记录提到的 `autodl_stability/` 未包含在当前仓库中。

本实验研究：在相同初始化、数据抽样、固定训练步数下，SGD、Polyak、Nesterov
允许的学习率范围如何随 batch size 和 momentum gap 改变。

它沿用 [`imagenet_momentum_experiment.py`](imagenet_momentum_experiment.py) 的
ImageNet / ResNet / GroupNorm 设置和优化器，采用 [`../stability/README.md`](../stability/README.md)
的固定时长扫描思路。原有 `imagenet_momentum_experiment.py` 仍负责固定样本预算的
data-scaling 实验；新入口为 [`stability_experiment.py`](stability_experiment.py)。

## 能回答的问题与局限

报告量是 **finite-horizon empirical stability boundary**：在预先声明的时长、
loss 阈值、probe 子集和随机种子下，有多少次训练保持有界。它不是 uniform stability
或泛化稳定性，也不是无穷时长的 mean-square stability 证明。

原来的 PLK 模型有固定谱、平方损失和可精确计算的 population excess risk。
这里的特征和曲率随训练改变，cross-entropy 没有对应的已知 excess-risk 基准。
因此不能直接套用 PLK 的 `loss < 1`，不能预先给 NN 指定谱指数 beta，也不能把
Nesterov 的边界强行拟合成 B^beta。即使 NN 中趋势不明显，这也是该迁移实验的有效结果。

## 预先固定的协议

| 项目 | 默认设置 |
|---|---|
| 数据与模型 | ImageNet `train/<class>/*`；随机初始化 ResNet-18，全部层训练 |
| 归一化 | GroupNorm；避免 batch statistics 随 batch 改变 |
| 算法 | SGD、Polyak、显式 look-ahead Nesterov，直接复用现有更新代码 |
| Batch 扫描 | B = 1, 2, 4, 8, 16, 32, 64, 128, 256；gap = 1/64 |
| Gap 扫描 | gap = 2^-6, 2^-5.5, …, 2^-1.5，共 10 点；B = 4 |
| 学习率 | 所有算法、B 共用 50 个绝对学习率，logspace(2^-10, 8) |
| 时长与种子 | K = 2000 个 optimizer updates；seeds = 0, 1, 2, 3, 4 |
| 阻尼检查 | 每个 momentum 点强制 K × gap ≥ 30；默认最小值 31.25 |
| 训练设置 | 恒定学习率；零 weight decay；无 warmup、schedule、梯度裁剪、AMP、label smoothing |
| 抽样 | 从训练集有放回抽样；随机裁剪、翻转；配对种子 |
| 监测 | 固定 1024 张训练图片，固定中心裁剪；每 50 步检查一次实际参数位置 |
| 尾部 | 最后 5 个 probe loss；与前面 5 个 probe loss 比较 |

固定 K 意味着 D = BK 随 B 改变。这是实验设计的一部分：固定 D 会使大 batch
对应更短的动力学时长，容易混入 `stability` 文档指出的有限时长伪影。
K × gap ≥ 30 只是沿用的最低检查，不能保证非线性 NN 已进入渐近状态。

两种扫描的交点 `(B, gap) = (4, 1/64)` 只运行一次并复用原始结果。
SGD 使用 rho = 0，gap 扫描中的 SGD 行复用同一组 baseline。
同一 seed 在各算法、学习率下共享初始化和抽样种子；probe seed 固定为 1729。
主分析使用训练 probe，不使用验证集来挑稳定阈值。probe accuracy 仅作诊断。

原 `stability/batch_size.ipynb` 正式扫描为 `3 × 9 × 50 × 5 = 6,750` trial；
`stability/momentum_gap.ipynb` 在 SGD 内部复用后为 `(2 × 10 + 1) × 50 × 5 = 5,250` trial。
原两份 notebook 独立运行合计 **12,000 trial**，交点使用相同种子但各自计算。
本实现进一步复用交点的 `3 × 50 × 5 = 750` 次重复计算，因此默认共 **11,250 个唯一 trial**：
SGD 2,250，Polyak 和 Nesterov 各 4,500。以上不计 notebook 中的调试调用。

本次恢复的是原扫描的网格、seed 数量和训练时长。NN 的 CE 判据仍与原线性模型的
population excess loss 末尾 20 步均值 < 1 不同；边界统计和绘图也尚未恢复原版。
原版对逐 seed 最大稳定学习率求均值并拟合 `min{L, c x}`；当前 NN 输出全部 seed
通过的边界区间及其几何中心、稳定比例图，没有原版 scaling 拟合线或指数估计。

## 判定和边界

记初始化 probe CE 为 L0，最后五次 probe CE 的均值为 Ltail，之前五次的均值为 Lprev。

- `survived`：完成 K 次更新，没有触发非有限值或计算停止阈值。
- `bounded`：survived，并且 Ltail/L0 ≤ 2、Ltail/Lprev ≤ 1.25。
- `learned`：bounded，并且 Ltail/L0 ≤ 0.95；单独报告，避免把不学习也不发散混作有效训练。

非有限 train loss / gradient / parameter / probe loss 立即停止。默认还设置
probe CE > 100 L0、gradient norm > 10^6、parameter norm > 100 × 初始 norm
的计算停止阈值。日志区分非有限值与这些有限阈值；触及阈值表示按本协议停止，
不等于已证明数学发散。检查的是实际更新后的参数；Nesterov 训练 CE 在 look-ahead
处计算并单独标注，不与 base-point probe CE 混用。

默认一个学习率须 **全部种子 bounded** 才算通过。`--required-fraction` 可以更改，
但必须预先设置。五个种子的通过比例以 0.2 为间隔；正式报告应同时展示逐 seed 结果。

`boundaries.csv` 的状态解释：

| 状态 | 解释 |
|---|---|
| `bracketed` | 完整扫描先通过后失败；边界位于最后通过与首次失败学习率之间 |
| `left_censored` | 所有点失败；只能记录低于最小扫描学习率的界限，需向下扩展 |
| `right_censored` | 所有点通过；只能记录高于最大扫描学习率的界限，需向上扩展 |
| `nonmonotone` | 失败之后又通过；不输出单一边界，查看完整稳定比例图 |
| `incomplete` | 缺少任意一个学习率或种子；不输出完整边界 |

这些区间以采样网格及“单一边界”的假设为条件；网格之间的行为仍未知。
图中误差棒表示学习率网格区间，**不是置信区间**。不对无稳定点的 seed 做 NaN
过滤后平均，也不把计算未完成、OOM、数据加载失败当成算法不稳定。
图中点是区间端点的几何中心；不同方法在水平方向略微错开，以便看清重合区间。

## 运行

AutoDL 上直接运行 Python：独立文件夹 `autodl_stability/`（历史目录，当前仓库未收录）。
它包含启动脚本、安装脚本和完整训练/绘图依赖，无需 notebook 或其他仓库目录。
进入该文件夹后执行 `bash run_all.sh`，即可完成 Kaggle 数据下载、两轴原扫描和绘图；
需要预先上传自己的 Kaggle 凭证并在网页接受比赛规则，详见独立文件夹说明。
已准备好数据时，执行 `bash setup.sh`，再执行
`bash start.sh --data-root /root/autodl-tmp/imagenet` 即可只运行实验。
两个入口默认 `--preset original`，共 11,250 个去重 trial，输出到该文件夹的 `output/original/`。
`--preset full` 是同一扫描的别名，输出到 `output/full/`；40-trial 的小范围扫描需显式
使用 `--preset pilot`，输出到 `output/pilot/`。`--preset smoke` 仍为 16 个 FakeData trial。
交付物是独立文件夹，无需 ZIP。

更新本地文件不会改变 AutoDL 上已启动的进程；旧进程仍使用启动时的参数。
已上传旧版文件夹时，只需将新增的 `autodl_stability/run_original.sh` 上传到 `run_all.sh`
同一目录，旧任务结束或停止后执行 `bash run_original.sh`。该入口显式传入全部原扫描参数，
兼容旧版和新版 runner，无需覆盖核心 Python 文件；输出到 `output/original/`。
也可先执行 `bash run_original.sh --dry-run` 查看清单。已有下载数据可以复用；
新协议不能与旧 pilot / full 的 manifest 和结果混用，目标目录已有其他协议时另选 `--output`。

交互版本：[stability.ipynb](stability.ipynb)。从仓库根目录安装
`python -m pip install -e '.[nn,notebooks]'`，再运行
`jupyter lab momentum_nn/stability.ipynb`，选择对应的 Python kernel。
notebook 默认以 FakeData 执行 16 个快速 trial，支持从头运行并显示两种扫描的
边界图、稳定比例、逐 seed 表格和单次训练轨迹。

在配置单元格设置 `PRESET="pilot"` 或 `"full"`，填写 `DATA_ROOT` 即可切换到
ImageNet。`RUN_TRAINING=False` 只生成清单并查看已有结果；`MAX_NEW_TRIALS`
限制本次新执行任务数。已完成结果自动复用，改变协议时更换 `RUN_NAME`。
默认结果保存到 `momentum_nn/results/generated/notebook_<preset>_v1/`。

在仓库根目录，使用 Python 3.10+。先安装与机器匹配的 PyTorch / torchvision，
再安装本项目的 NN 依赖：

```bash
python -m pip install -e '.[nn]'
```

无需下载 ImageNet 的流程检查（32×32 假数据；所有输出标记为 SMOKE TEST）：

```bash
python -m momentum_nn.stability_experiment sweep \
  --runs-root momentum_nn/results/generated/smoke \
  --fake-data --smoke-test --device cpu --workers 0 \
  --crop-size 32 --probe-size 8 --probe-batch 4 \
  --steps 4 --eval-every 1 --tail-evals 2 \
  --batches 2 --fixed-batch 2 --fixed-gap 0.25 --gaps 0.25 \
  --lrs 0.0001,100 --seeds 0 --max-micro-batch 1
```

先生成正式实验清单，查看实际开销；不加载数据、不执行训练：

```bash
python -m momentum_nn.stability_experiment sweep \
  --data-root /path/to/imagenet \
  --runs-root momentum_nn/results/generated/imagenet_stability \
  --dry-run
```

完整默认设置有 **11,250 个唯一 trial**，每次最多 2000 步，另有 probe 评估开销。
如需额外检查扫描范围，可显式运行一个小范围真实数据 pilot，使用独立输出目录。
pilot 保留正式 K 和判定阈值；下面是只扫描 batch 的 30-trial 示例：

```bash
python -m momentum_nn.stability_experiment sweep \
  --data-root /path/to/imagenet \
  --runs-root momentum_nn/results/generated/imagenet_pilot \
  --axis batch --batches 4,64 --lrs 0.001,0.01,0.1,1,8 --seeds 0
```

去掉完整清单命令的 `--dry-run` 会顺序执行全部 trial。也可按清单分发单个任务：

```bash
CUDA_VISIBLE_DEVICES=0 python -m momentum_nn.stability_experiment run \
  --runs-root momentum_nn/results/generated/imagenet_stability --job-index 0
```

`job-index` 从 0 开始，上界为 manifest 的 jobs 长度减一。每个任务只用一张 GPU；
`--max-micro-batch` 控制显存中的 micro-batch，梯度按实际样本数加权累积。
完整 global batch 先装入 CPU 内存，因此很大的 B 也需要足够的主机内存。
也可在相同 sweep 命令上添加 `--shard-count N --shard-index i` 分发互不重叠的任务。
先由一个进程生成 manifest，再启动 workers；同一 trial 不要同时启动两次。
`auto` 支持 CUDA、MPS、CPU；单设备路径已实现，跨设备 bitwise 一致性不作假设。
同一正式扫描尽量保持硬件、库版本和 worker 数一致。

每次 trial 完成后原子写入 summary；重跑会验证并跳过它。被中断的 trial 从同一 seed
重新开始，没有中途模型 checkpoint。配置或训练源码变化时要求新输出目录，防止误用旧缓存。
数据目录应保持只读且不变；保存文件名、标签、文件大小组成的 listing hash，
可以发现常见数据变化，但不替代逐字节内容校验。

所有任务结束后重新汇总：

```bash
python -m momentum_nn.stability_experiment summarize \
  --runs-root momentum_nn/results/generated/imagenet_stability
```

输出包含 `manifest.json`、每个 trial 的 `config.json` / `history.jsonl` /
`summary.json`、完整 `stability_grid.csv`、`boundaries.csv`、
`per_seed_boundaries.csv`，以及两种扫描的边界图和稳定比例图（PNG / PDF）。

## 给老板看的结论应如何组织

先报告两条经验边界及全部 seed 的稳定比例，回答两种 momentum 对 batch、gap 的
响应是否不同。再补充边界附近的 CE、gradient norm、parameter norm 轨迹，解释失败类型。
在得到实际结果前，不预设哪一种优化器必须更稳定。

正式写作前至少做两项敏感性检查：在边界附近用相同 seed 将 K 从 2000 延长至
4000 / 8000；检查 bounded 阈值改为 Ltail/L0 ≤ 1.5 或 3、growth ratio 改为
1.1 或 1.5 时，趋势是否保留。相邻学习率用更密网格复核；根据 seed 间差异增加重复次数。
更改协议时使用新的输出目录。原始 probe 日志允许重算尾部判定；已经因停止阈值
提前终止的 trial，不能靠放宽阈值补成完整运行，必须重新训练。

可以写：“我们在 ImageNet/ResNet-18 上测量了预先固定训练时长和判据下的经验稳定
学习率范围，以考察线性模型中观察到的 batch/momentum 现象能否延伸到非线性网络。”
是否出现预期趋势，要由真实数据结果决定。

## 本地验证记录（2026-09-08）

在 CPU、Python 3.11.15、PyTorch 2.14.0 / torchvision 0.29.0 下完成：

- 9 项自动检查：更新公式对照、非整除 micro-batch 累积、阻尼限制、扫描交点去重、
  有界与学习判据、边界截断/非单调/缺失数据、配置隔离和部分结果汇总。
- FakeData 上 6 次 ResNet-18 试跑，覆盖三种算法及低/高学习率；核对相同初始 loss、
  参数范数和 probe 图片；完成 CSV、PNG、PDF 输出及已完成 trial 的缓存复用。
- 16 张合成图片组成的 ImageFolder 上 2 次 Nesterov 试跑，覆盖独立 DataLoader
  worker、B=3 / micro-batch=2 的余数累积，以及有限 loss 停止阈值。

这些检查只验证代码路径。**尚未运行真实 ImageNet 扫描，也没有 NN stability 的科学结论。**

