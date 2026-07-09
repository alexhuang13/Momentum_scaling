"""Model architectures, implemented strictly per `setting.md`.

Three models:
  * FCNet     -- MNIST, ~99k params  (3 hidden layers x 100 ReLU, dropout 0.5)
  * SmallCNN  -- CIFAR-10, ~576k params (3 conv 64/64/128, 5x5, maxpool, FC-128)
  * ResNet32  -- CIFAR-10, ~467k params (15 residual blocks, 16/32/64 channels)

`setting.md` describes softmax-activated output layers; in PyTorch the softmax is
folded into `nn.CrossEntropyLoss`, so every model returns raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 2.1 Fully-connected network (MNIST)
# --------------------------------------------------------------------------- #
class FCNet(nn.Module):
    """784 -> [100 -> ReLU -> Dropout] x3 -> 10.

    Dropout keep-prob 0.5  =>  p_drop = 0.5.
    """

    def __init__(self, in_dim: int = 784, hidden: int = 100,
                 n_classes: int = 10, p_drop: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2.2 Convolutional network (CIFAR-10)
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """3 conv layers (64, 64, 128), 5x5 kernels, each followed by a 2x2 max-pool
    (stride 2); then FC-128 ReLU + dropout, then a 10-way output.

    NOTE ON PADDING / PARAMETER COUNT.  `setting.md` states "no padding" but also
    quotes ~576k trainable parameters.  These are inconsistent:
        * padding='valid' (no padding): 32 ->28 ->pool14 ->conv10 ->pool5 ->conv1
          gives a 128x1x1 feature -> FC input 128 -> ~330k params.
        * padding='same':               32 ->pool16 ->pool8 ->pool4
          gives a 128x4x4 feature -> FC input 2048 -> ~576k params. <-- matches.
    The ~576k figure only reproduces with 'same' padding, so that is the DEFAULT
    here (parameter count is the more precise spec).  Pass padding='valid' to get
    the literal "no padding" wording (and ~330k params) instead.

    Optionally (`use_bn=True`) inserts BatchNorm after each conv and removes the
    FC-phase dropout -- the "slightly modified 3-layer CNN" variant that
    `setting.md` mentions for one test-performance experiment.
    """

    def __init__(self, n_classes: int = 10, p_drop: float = 0.5,
                 use_bn: bool = False, padding: str = "same"):
        super().__init__()
        if padding not in ("same", "valid"):
            raise ValueError("padding must be 'same' or 'valid'")
        self.use_bn = use_bn
        pad = 2 if padding == "same" else 0   # 5x5 kernel -> pad 2 keeps size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=pad)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=pad)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=pad)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Infer the flattened feature width from a dummy 32x32x3 forward pass so
        # the linear head is correct for either padding mode.
        self.fc1 = nn.LazyLinear(128)
        self.fc_drop = None if use_bn else nn.Dropout(p_drop)
        self.fc2 = nn.Linear(128, n_classes)

    def _conv(self, x, conv, bn):
        x = conv(x)
        if self.use_bn:
            x = bn(x)
        return F.relu(x, inplace=True)

    def _features(self, x):
        x = self.pool(self._conv(x, self.conv1, getattr(self, "bn1", None)))
        x = self.pool(self._conv(x, self.conv2, getattr(self, "bn2", None)))
        x = self._conv(x, self.conv3, getattr(self, "bn3", None))
        # 'valid' path shrinks the 3rd conv output to 1x1 (no final pool);
        # 'same' path leaves a 4x4 map that we do NOT pool again either, so the
        # third max-pool of the description is only applied in the 'same' case.
        if x.shape[-1] > 1:
            x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self._features(x)
        x = F.relu(self.fc1(x), inplace=True)
        if self.fc_drop is not None:
            x = self.fc_drop(x)
        return self.fc2(x)


# --------------------------------------------------------------------------- #
# 2.3 ResNet-32 (CIFAR-10) -- the CIFAR-style ResNet of He et al. (2016)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """Standard CIFAR ResNet basic block (two 3x3 convs + identity/projection)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Projection shortcut when the spatial size / channel count changes.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet32(nn.Module):
    """CIFAR ResNet-32: an initial 3x3 conv + 3 stages of 5 basic blocks each.

    Depth = 6n + 2 with n = 5  =>  32 conv layers, 15 residual blocks.
    Feature-map stages: 32x32x16 -> 16x16x32 -> 8x8x64, then global avg-pool
    (`setting.md` says a 2x2 avg-pool with stride 2; on the 8x8 map that is
    equivalent to standard global pooling followed by the linear head).
    """

    def __init__(self, n_classes: int = 10, n: int = 5):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(16, n, stride=1)   # 32x32x16
        self.layer2 = self._make_stage(32, n, stride=2)   # 16x16x32
        self.layer3 = self._make_stage(64, n, stride=2)   # 8x8x64
        self.fc = nn.Linear(64, n_classes)

    def _make_stage(self, planes: int, n_blocks: int, stride: int):
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)   # 8x8 -> 1x1 (global average pool)
        out = torch.flatten(out, 1)
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_MODELS = {
    "fc": FCNet,
    "cnn": SmallCNN,
    "resnet32": ResNet32,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(_MODELS)}")
    return _MODELS[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick parameter-count sanity check against the sizes quoted in setting.md.
    for name, expect in [("fc", "~99k"), ("cnn", "~576k"), ("resnet32", "~467k")]:
        m = build_model(name)
        print(f"{name:10s} params = {count_params(m):>9,d}   (setting.md: {expect})")
