# Momentum on MNIST / CIFAR-10

Empirical companion to the synthetic scaling-law study in
`../linear_regression/linear_regression_scaling.ipynb`. It runs **SGD**,
**Heavy Ball (HB)**, and **Nesterov (NAG)** on real image classifiers, with the
model architectures specified below.

Two experiments:

* **A — baseline comparison** (`run_baseline.py`): fixed dataset, multi-epoch,
  the three optimizers under a **shared fixed learning rate** and `rho=0.9`,
  comparing train-loss / test-loss / test-accuracy curves.
* **B — compute-budget scaling** (`run_scaling.py`): fixed batch size `B`, sweep
  the number of steps `K`, plot final **test loss vs. D = B·K** on log-log axes
  and fit a slope — the real-network analogue of the notebook's *excess loss vs.
  D* curve.

## Setup

The three optimizers are **hand-written** (`experiments/optimizers.py`) to match
the update rules in the notebook exactly, rather than using
`torch.optim.SGD(momentum=..., nesterov=...)` (whose Nesterov form uses a
different LR convention):

```
SGD        theta_{t+1} = theta_t - eta*g(theta_t)
HeavyBall  theta_{t+1} = theta_t - eta*g(theta_t)           + rho*(theta_t - theta_{t-1})
Nesterov   theta_{t+1} = theta_t - eta*g(theta_t + rho*m_t) + rho*(theta_t - theta_{t-1})
```

Models (`experiments/models.py`):

| model      | dataset  | params (this repo) | target |
|------------|----------|--------------------|--------|
| `fc`       | MNIST    | 99,710             | ~99k   |
| `cnn`      | CIFAR-10 | 575,818 (`same` pad) | ~576k |
| `resnet32` | CIFAR-10 | 466,906            | ~467k  |

> **CNN padding caveat.** The reference spec says "no padding" but also quotes
> ~576k params. Those are inconsistent: valid (no) padding gives ~330k;
> `padding='same'` gives ~576k. The parameter count is the tighter spec, so
> `SmallCNN` defaults to `padding='same'`. Pass `padding='valid'` in
> `model_kwargs` for the literal wording (~330k).

## Experimental setting (models & tasks)

Two standard image-classification tasks; architectures reproduced by
`experiments/models.py`.

### Tasks and datasets

* **MNIST** — fully-connected network. Input: 784-dim image vectors. Output:
  10-class softmax.
* **CIFAR-10** — 3-layer CNN and ResNet-32. Output: 10-class softmax. Standard
  augmentation on the train split: random horizontal flip; random horizontal +
  vertical shift with range = 10% of image width/height (realised here as
  `RandomHorizontalFlip` + `RandomCrop(32, padding=4, reflect)`).

### 1. Fully-connected network (MNIST) — `fc`

* 3 hidden layers, each 100 ReLU neurons;
* dropout after each hidden layer, keep-prob 0.5 (`p_drop=0.5`);
* input dim 784; output layer 10 softmax neurons;
* ~99k trainable parameters.

The output softmax is folded into `nn.CrossEntropyLoss`, so the model returns
raw logits.

### 2. Convolutional network (CIFAR-10) — `cnn`

* 3 convolutional layers, 5×5 kernels;
* channels 64 → 64 → 128;
* each conv followed by a 2×2 max-pool, stride 2;
* then a fully-connected ReLU layer of size 128;
* output layer 10 softmax neurons;
* dropout after the FC layer, keep-prob 0.5;
* ~576k trainable parameters (see padding caveat above).

BatchNorm variant (`--use-bn`, "test-performance" setup): a BatchNorm layer
after each convolution, and the FC-phase dropout removed.

### 3. ResNet-32 (CIFAR-10) — `resnet32`

* 32 convolutional layers, 15 residual blocks;
* stage feature maps: 32×32×16 (5 blocks) → 16×16×32 (5 blocks) → 8×8×64
  (5 blocks);
* global average pooling after the residual blocks, then a 10-way linear head;
* ~467k trainable parameters.

BatchNorm is intrinsic to the residual blocks (a 32-layer plain net does not
train), so unlike `fc`/`cnn` this model always contains normalization.

## Install (on the GPU machine)

Python 3.10–3.12 recommended (torch wheels; 3.14 has no wheels yet).

```bash
python -m venv .venv && source .venv/bin/activate
# install the CUDA build of torch for your driver, see pytorch.org, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # numpy / matplotlib / pyyaml (+ pins)
```

## Run

Experiment A (baseline comparison):

```bash
python -m experiments.run_baseline --config configs/mnist_fc.yaml       --out results/mnist_fc_baseline.json
python -m experiments.run_baseline --config configs/cifar_cnn.yaml      --out results/cifar_cnn_baseline.json
python -m experiments.run_baseline --config configs/cifar_resnet32.yaml --out results/cifar_resnet32_baseline.json
```

Experiment B (D = B·K scaling):

```bash
python -m experiments.run_scaling --config configs/mnist_fc.yaml       --out results/mnist_fc_scaling.json
python -m experiments.run_scaling --config configs/cifar_cnn.yaml      --out results/cifar_cnn_scaling.json
python -m experiments.run_scaling --config configs/cifar_resnet32.yaml --out results/cifar_resnet32_scaling.json
```

Plot:

```bash
python -m experiments.plot --baseline results/mnist_fc_baseline.json --out results/mnist_fc_baseline.png
python -m experiments.plot --scaling  results/mnist_fc_scaling.json  --out results/mnist_fc_scaling.png
```

CLI flags override config values, e.g. `--lr 0.02 --epochs 40 --seeds 0`.

## What to expect / honesty notes

* **A** should show the momentum methods reaching a given train-loss/accuracy in
  fewer steps than SGD, with Nesterov ≲ HB ≲ SGD in final loss — consistent with
  the theory's smaller prefactor for momentum at equal asymptotic rate.
  Because the LR is *shared and fixed*, this isolates the effect of momentum; it
  is not a best-tuned-per-optimizer comparison.
* **B** is an **empirical** power law. On real data we don't know the spectral /
  source exponents `(s, beta)`, so — unlike the notebook — we do **not** scale
  `B, K, rho` with `D`; that scaling requires those exponents. So B measures how
  test loss falls with compute budget under a fixed recipe; it is *not* a
  reproduction of the exact `D^{-s/(s+1)}` rate.

## Layout

```
experiments/
  models.py        FC / CNN / ResNet-32 (setting.md)
  optimizers.py    SGD / HeavyBall / Nesterov (notebook-aligned)
  data.py          MNIST / CIFAR loaders, augmentation, deterministic subsets
  train.py         one training run + curve recording
  run_baseline.py  experiment A
  run_scaling.py   experiment B
  plot.py          figures
configs/           per-(model,dataset) hyperparameters
requirements.txt
```

Sanity-check parameter counts without a GPU:
`python -m experiments.models` (needs torch installed).
