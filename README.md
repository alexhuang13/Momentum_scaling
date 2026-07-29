# From Polyak to Nesterov: Momentum Scaling in Power-Law Kernel Regression

本仓库是论文 **“From Polyak to Nesterov: Landscape-Aware Momentum Methods May Improve Stability and Data Scaling”** 的精简复现实验代码。研究对象是具有幂律谱的高维核回归，比较 **SGD、Polyak heavy-ball momentum** 与 **Nesterov momentum** 在随机梯度下的稳定性、临界 batch size 和数据缩放规律。

> 仓库已将原先分离且大量重复的 CPU/GPU notebook 合并为一套 PyTorch 实现。`--device` 只决定计算设备，模型、随机过程、优化器更新和实验配置均共用同一份代码。

## 主要结论

设特征协方差谱和目标满足

\[
\lambda_j \asymp j^{-\beta}, \qquad
\lambda_j |\theta_j^\star|^2 \asymp j^{-(1+s\beta)},
\]

其中 \(\beta>1\) 是 capacity exponent，\(s>0\) 是 source exponent。数据预算记为 \(D=BK\)，其中 \(B\) 是 batch size，\(K\) 是迭代步数。

### 1. 随机稳定性边界

| 方法 | 临界学习率 |
|---|---|
| SGD | \(\eta_{\rm stab}\asymp 1\) |
| Polyak | \(\eta_{\rm stab}\asymp \min\{1,B(1-\rho)\}\) |
| Nesterov | \(\eta_{\rm stab}\asymp \min\{1,B^\beta(1-\rho)\}\) |

Nesterov 的 look-ahead 修正使稳定区域对 batch size 的增长更快，因此比 Polyak 更早达到确定性稳定上限。

### 2. 数据缩放

在有标签噪声时：

- SGD 的最优数据率为 \(D^{-s/(s+1)}\)；
- Polyak 保持相同的最优率，但能在更大的 batch size 范围内保持该速度；
- Nesterov 可达到更快的
  \[
  D^{-\frac{2s\beta}{2s\beta+\beta+1}+o(1)}.
  \]

![Batch-size phase diagram](assets/figures/rate_diagram.png)

## 论文中的核心数值结果

下图左侧验证 Polyak 与 Nesterov 不同的稳定性边界；中间展示 Polyak 在超过 SGD 临界 batch 后仍能保持最优缩放；右侧展示 Nesterov 在大 batch 设置下取得更快的数据缩放率。

| 稳定性边界 | Polyak 临界 batch | Nesterov 大 batch 加速 |
|---|---|---|
| ![Stability boundary](assets/figures/beta1_5s2_plot.png) | ![Polyak scaling](assets/figures/mom_large_batch.png) | ![Nesterov scaling](assets/figures/nag_acc.png) |

## 仓库结构

```text
.
├── README.md
├── pyproject.toml              # 可编辑安装与命令行入口
├── requirements.txt
├── src/momentum_scaling/
│   ├── core.py                 # 幂律问题、统一 CPU/GPU 后端、三种优化器
│   ├── experiments.py          # 稳定性与数据缩放实验
│   ├── plotting.py             # 从结果文件生成图（README 不包含画图代码）
│   └── cli.py                  # 统一命令行入口
├── tests/test_core.py          # 核心公式与更新规则测试
├── paper/                      # 论文 LaTeX、参考文献和论文原图
├── assets/figures/             # README 使用的关键 PNG 图片
└── results/                    # 运行时输出；默认不纳入版本控制
```

旧版 CPU/GPU notebook、重复画图脚本、缓存、临时二进制和重复 PDF 已移除。论文源码保留在 `paper/`，可直接核对定理、实验配置和原始图。

## 安装

需要 Python 3.10 或更高版本。建议创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`requirements.txt` 仅包含 NumPy、PyTorch 和 Matplotlib。若使用 NVIDIA GPU，请按照 PyTorch 官方说明安装与本机 CUDA 匹配的 PyTorch wheel，再执行 `python -m pip install -e .`。

检查安装：

```bash
python -m unittest discover -s tests -v
momentum-scaling --help
```

## 快速复现

快速配置用于检查环境、设备和完整数据流，通常只需较短时间，但**不应作为论文图的定量替代**：

```bash
momentum-scaling all --preset quick --device auto --output-dir results/quick
```

也可不安装命令行脚本，直接运行：

```bash
PYTHONPATH=src python -m momentum_scaling.cli all \
  --preset quick --device auto --output-dir results/quick
```

输出包括：

```text
results/quick/
├── stability.json
├── stability.npz
├── stability.png
├── stability.pdf
├── data_scaling.json
├── data_scaling.png
└── data_scaling.pdf
```

JSON 保存配置、设备、随机种子、各实验点均值和标准差；NPZ 额外保存稳定性扫描的完整布尔网格与逐 seed 边界，方便复查。

## 论文配置复现

### 稳定性边界

论文设置：\(d=4096\)、\(1-\rho=2^{-24}\)、\(B=2^0,\ldots,2^{12}\)、50 个学习率、2000 步、5 个随机种子。

```bash
momentum-scaling stability \
  --preset paper \
  --device cuda \
  --dtype float64 \
  --seed 0 \
  --output-dir results/stability
```

由于 \(1-\rho\) 极小，建议使用 `float64`；`float32` 可能将 \(\rho\) 舍入为 1。该扫描计算量较大，推荐 CUDA GPU。

### 数据缩放实验

论文设置：\(d=4096\)、\(s=0.3\)、\(\beta=4\)、\(\sigma=0.1\)、\(D\in\{2000,4000,\ldots,128000\}\)，每点 100 个独立重复。

```bash
momentum-scaling data-scaling \
  --preset paper \
  --device cuda \
  --dtype float64 \
  --seed 0 \
  --output-dir results/data_scaling
```

该命令同时运行论文附录中的两组配置：

1. **Polyak critical-batch experiment**：比较 \(B\asymp D^{s/(s+1)}\) 与 \(B\asymp D^{2s/(s+1)}\)；
2. **Nesterov large-batch experiment**：所有方法使用 Nesterov 理论指定的大 batch scaling，并对最后 1% 迭代的 excess risk 求平均。

已有结果只需重新绘图时：

```bash
momentum-scaling plot --output-dir results/data_scaling
```

## CPU、CUDA 与 Apple Silicon

```bash
# 自动选择：CUDA > MPS > CPU
momentum-scaling all --preset quick --device auto

# 强制 CPU
momentum-scaling all --preset quick --device cpu --dtype float64

# 指定 NVIDIA GPU
momentum-scaling all --preset quick --device cuda:0 --dtype float64

# Apple Silicon
momentum-scaling all --preset quick --device mps --dtype float32
```

MPS 对 `float64` 支持有限，因此 `--dtype auto` 在 MPS 上选择 `float32`，在 CPU/CUDA 上选择 `float64`。论文稳定性扫描仍建议使用支持双精度的 CPU 或 CUDA GPU。

## 可复现性说明

- 所有随机数由 `--seed` 控制，并为算法、实验点和重复试验使用确定性的 seed 偏移；
- CPU 与 GPU 共享同一数学实现，但不同硬件/BLAS/CUDA 版本可能产生微小浮点差异；
- 稳定性实验使用真实 Gaussian mini-batch，并分块计算梯度以限制显存/内存；
- 数据缩放实验按照论文附录，直接采样 curvature-aligned Gaussian gradient noise；
- 每次运行都会在终端记录 Python、PyTorch、device 和 dtype，并写入结果 JSON；
- `quick` 只做 smoke test；论文结论应使用 `paper` preset 和论文规定的 100 次重复。

## 优化器约定

本仓库严格使用论文中的更新：

\[
\begin{aligned}
\text{SGD: }&\theta_{k+1}=\theta_k-\eta\widehat g_k(\theta_k),\\
\text{Polyak: }&\theta_{k+1}=\theta_k-\eta\widehat g_k(\theta_k)
+\rho(\theta_k-\theta_{k-1}),\\
\text{Nesterov: }&\widetilde\theta_k=\theta_k+\rho(\theta_k-\theta_{k-1}),\\
&\theta_{k+1}=\widetilde\theta_k-\eta\widehat g_k(\widetilde\theta_k).
\end{aligned}
\]

这与某些深度学习框架中带特定学习率重参数化的 Nesterov 实现不完全相同，因此代码没有直接调用框架内置 optimizer。

## 论文源码

论文正文位于 `paper/main_arxiv.tex`，实验细节位于 `paper/appendix.tex` 的 **Experimental Details**。编译方式取决于本机 TeX 环境，典型命令为：

```bash
cd paper
latexmk -pdf main_arxiv.tex
```
