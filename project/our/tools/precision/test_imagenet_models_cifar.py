#!/usr/bin/env python3
"""
Real accuracy for ImageNet-architecture models (ResNet50/101/152, VGG16/19) on the
real CIFAR-10 dataset via transfer learning.

Method (scientifically valid, real data, no synthetic labels):
  1. Load ImageNet-pretrained backbone (frozen -> keeps its true activation ranges).
  2. Precompute backbone features on CIFAR-10 train set ONCE (fast).
  3. Train a linear head (10 classes) on those features.
  4. Attach head, evaluate quantized full model on the CIFAR-10 test set at each ell.

Because the backbone is frozen with pretrained weights, the activation magnitudes
that drive the fixed-point range requirement are exactly the ImageNet ones
(VGG max |act| ~186, ResNet ~12), so the precision cliff is faithful.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).parent))
from fixed_point import FixedPointConfig, convert_model_to_fixed_point

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = Path('./data')

# ImageNet preprocessing, CIFAR upsampled to 224 (standard transfer-learning recipe)
TF = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def cifar_loaders(train_bs=64, test_bs=64):
    train = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=TF)
    test = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=TF)
    return (DataLoader(train, batch_size=train_bs, shuffle=False, num_workers=4),
            DataLoader(test, batch_size=test_bs, shuffle=False, num_workers=4))


def split_backbone_head(name):
    """Return (backbone, feature_dim, rebuild_fn) where rebuild_fn(head)->full model."""
    if name == 'ResNet50':
        m = models.resnet50(weights='IMAGENET1K_V1')
    elif name == 'ResNet101':
        m = models.resnet101(weights='IMAGENET1K_V1')
    elif name == 'ResNet152':
        m = models.resnet152(weights='IMAGENET1K_V1')
    elif name == 'VGG16':
        m = models.vgg16(weights='IMAGENET1K_V1')
    elif name == 'VGG19':
        m = models.vgg19(weights='IMAGENET1K_V1')
    else:
        raise ValueError(name)

    if name.startswith('ResNet'):
        feat_dim = m.fc.in_features
        # backbone = everything up to (not including) fc; produces (B, feat_dim)
        backbone = nn.Sequential(*(list(m.children())[:-1]), nn.Flatten(1))

        def rebuild(head):
            m.fc = head
            return m
        return backbone, feat_dim, rebuild
    else:  # VGG
        feat_dim = m.classifier[6].in_features  # 4096
        # backbone = features + avgpool + flatten + classifier[:-1]
        backbone = nn.Sequential(
            m.features, m.avgpool, nn.Flatten(1),
            *list(m.classifier.children())[:-1]
        )

        def rebuild(head):
            m.classifier[6] = head
            return m
        return backbone, feat_dim, rebuild


@torch.no_grad()
def extract_features(backbone, loader):
    backbone.eval().to(DEVICE)
    feats, labels = [], []
    for x, y in loader:
        f = backbone(x.to(DEVICE)).cpu()
        feats.append(f)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_head(feat_dim, train_feats, train_labels, epochs=15):
    head = nn.Linear(feat_dim, 10).to(DEVICE)
    ds = TensorDataset(train_feats, train_labels)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    head.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
    return head.cpu()


@torch.no_grad()
def evaluate(model, loader):
    model.eval().to(DEVICE)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def run_model(name, train_loader, test_loader, bitwidths=(16, 20, 24, 28)):
    print(f"\n{'='*70}\n{name} (ImageNet-pretrained backbone, head fine-tuned on CIFAR-10)\n{'='*70}")
    t0 = time.time()

    backbone, feat_dim, rebuild = split_backbone_head(name)

    print("  Extracting frozen-backbone features on CIFAR-10 train set...")
    tr_feats, tr_labels = extract_features(backbone, train_loader)
    print(f"    features: {tuple(tr_feats.shape)}  ({time.time()-t0:.0f}s)")

    print("  Training linear head (15 epochs)...")
    head = train_head(feat_dim, tr_feats, tr_labels)

    model = rebuild(head)

    float_acc = evaluate(model, test_loader)
    print(f"  Float32 baseline (ell=32): {float_acc:.2f}%")
    results = [(name, 32, float_acc)]

    for ell in bitwidths:
        cfg = FixedPointConfig(total_bits=ell, frac_bits=16)
        qmodel = convert_model_to_fixed_point(model, cfg)
        acc = evaluate(qmodel, test_loader)
        print(f"  ell={ell}: {acc:.2f}%")
        results.append((name, ell, acc))
        del qmodel
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    del model, backbone, tr_feats
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print(f"  ({time.time()-t0:.0f}s total)")
    return results


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = cifar_loaders()

    all_rows = []
    for name in ['ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'VGG19']:
        try:
            all_rows += run_model(name, train_loader, test_loader)
        except Exception as e:
            print(f"  !! {name} failed: {e}")

    out = Path('results/e0_imagenet_models_cifar.csv')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        f.write('model,ell,accuracy_pct\n')
        for m, ell, acc in all_rows:
            f.write(f'{m},{ell},{acc:.2f}\n')
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
