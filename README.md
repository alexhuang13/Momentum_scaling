# Momentum Scaling

This repository studies scaling behavior for three first-order optimizers:

- **SGD**
- **Heavy Ball (HB)**
- **Nesterov acceleration (NAG)**

The project has two complementary parts:

1. `linear_regression/`: synthetic diagonal linear-regression experiments where the population excess loss is computed analytically.
2. `cifar_mnist/`: PyTorch experiments on MNIST and CIFAR-10 classifiers using the same optimizer update conventions.

## Repository layout

```text
.
├── README.md
├── linear_regression/
│   ├── D_scaling.ipynb      # synthetic excess-loss scaling vs data budget D
│   └── CBS.ipynb            # fixed-D batch-size sweep / critical-batch-size study
└── cifar_mnist/
    ├── README.md            # detailed image-experiment notes
    ├── requirements.txt
    ├── configs/
    │   ├── mnist_fc.yaml
    │   ├── cifar_cnn.yaml
    │   └── cifar_resnet32.yaml
    └── experiments/
        ├── data.py          # MNIST / CIFAR-10 dataloaders and augmentation
        ├── models.py        # FCNet, SmallCNN, ResNet-32
        ├── optimizers.py    # hand-written SGD / HB / NAG updates
        ├── train.py         # single-run training loop and metrics
        ├── run_baseline.py  # experiment A: optimizer comparison
        ├── run_scaling.py   # experiment B: D = B*K compute scaling
        └── plot.py          # plotting utilities
```

Generated data, caches, JSON logs, and figures are written under local `data/` or `results/` directories when experiments are run.

## Optimizer conventions

The optimizers in `cifar_mnist/experiments/optimizers.py` are implemented by hand so that the image experiments match the update rules used in the linear-regression notebooks.

```text
SGD        theta_{t+1} = theta_t - eta * g(theta_t)
HeavyBall  theta_{t+1} = theta_t - eta * g(theta_t) + rho * (theta_t - theta_{t-1})
Nesterov   theta_{t+1} = theta_t - eta * g(theta_t + rho * m_t) + rho * (theta_t - theta_{t-1})
```

This is intentional: `torch.optim.SGD(..., nesterov=True)` uses a different learning-rate convention for Nesterov momentum.

## `linear_regression/`: synthetic scaling experiments

The notebooks study a diagonal linear-regression model with power-law spectrum and source condition. Because the model is synthetic, the population excess loss can be evaluated exactly.

### `D_scaling.ipynb`

Sweeps the data/compute budget `D` and compares final excess loss for SGD, Heavy Ball, and Nesterov.

Main purpose:

- use the theory-prescribed scaling of batch size `B`, steps `K`, and momentum `rho` with `D`;
- plot final excess loss vs `D` on log-log axes;
- compare the measured slope with the predicted rate

```text
E[E_K] ~ D^{-s/(s+1)}.
```

### `CBS.ipynb`

Fixes the total budget `D` and sweeps batch size `B`, with `K = floor(D / B)`.

Main purpose:

- study the tradeoff between small-batch noise and large-batch step scarcity;
- compare SGD, Heavy Ball, and Nesterov across batch sizes;
- cache results in `linear_regression/results/` when enabled in the notebook.

### Running the notebooks

```bash
cd linear_regression
python -m pip install numpy matplotlib jupyter
jupyter notebook D_scaling.ipynb
jupyter notebook CBS.ipynb
```

## `cifar_mnist/`: MNIST / CIFAR-10 experiments

This directory ports the optimizer comparison to real image-classification tasks.

### Config matrix

| Config | Dataset | Model | Main setting |
|---|---|---|---|
| `configs/mnist_fc.yaml` | MNIST | fully-connected net | `lr=0.01`, `rho=0.9`, `epochs=30`, `batch_size=128` |
| `configs/cifar_cnn.yaml` | CIFAR-10 | 3-layer CNN | `lr=0.01`, `rho=0.9`, `epochs=60`, `batch_size=128` |
| `configs/cifar_resnet32.yaml` | CIFAR-10 | ResNet-32 | `lr=0.1`, `rho=0.9`, `epochs=80`, `batch_size=128` |

Models are defined in `experiments/models.py`. A quick parameter-count check is available with:

```bash
cd cifar_mnist
python -m experiments.models
```

### Experiment A: baseline optimizer comparison

`experiments/run_baseline.py` trains SGD, Heavy Ball, and Nesterov under a shared fixed learning rate and compares training loss, test loss, and test accuracy.

```bash
cd cifar_mnist
python -m experiments.run_baseline --config configs/mnist_fc.yaml       --out results/mnist_fc_baseline.json
python -m experiments.run_baseline --config configs/cifar_cnn.yaml      --out results/cifar_cnn_baseline.json
python -m experiments.run_baseline --config configs/cifar_resnet32.yaml --out results/cifar_resnet32_baseline.json
```

### Experiment B: compute-budget scaling

`experiments/run_scaling.py` fixes batch size `B`, sweeps the number of optimizer steps `K`, and uses

```text
D = B * K
```

as the processed-sample compute budget. The script records final test loss and fits a log-log slope for each optimizer.

```bash
cd cifar_mnist
python -m experiments.run_scaling --config configs/mnist_fc.yaml       --out results/mnist_fc_scaling.json
python -m experiments.run_scaling --config configs/cifar_cnn.yaml      --out results/cifar_cnn_scaling.json
python -m experiments.run_scaling --config configs/cifar_resnet32.yaml --out results/cifar_resnet32_scaling.json
```

### Plotting

```bash
cd cifar_mnist
python -m experiments.plot --baseline results/mnist_fc_baseline.json --out results/mnist_fc_baseline.png
python -m experiments.plot --scaling  results/mnist_fc_scaling.json  --out results/mnist_fc_scaling.png
```

## Installation for image experiments

From `cifar_mnist/`:

```bash
python -m venv .venv
source .venv/bin/activate

# Choose the PyTorch build for your machine from https://pytorch.org/get-started/locally/.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

For macOS CPU/MPS environments, use the platform-appropriate PyTorch install command from the PyTorch website instead of the CUDA example.

## Notes and caveats

- The linear-regression notebooks are the theory-controlled setting: the source/spectrum parameters are known, so the prescribed scaling with `D` can be tested directly.
- The MNIST/CIFAR-10 scaling experiment is empirical. On real data the source/spectrum exponents are unknown, so `B`, `K`, and `rho` are not scaled with `D` in the same theory-prescribed way.
- When running with a YAML config, the config file defines the main experimental recipe. To change a recipe, edit the corresponding file in `cifar_mnist/configs/` or run the script without `--config` and pass CLI values manually.
- See `cifar_mnist/README.md` for more details on architectures, augmentation, plotting, and expected behavior.
