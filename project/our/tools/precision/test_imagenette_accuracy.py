#!/usr/bin/env python3
"""
Test ImageNet-architecture models on real ImageNette validation set.
ImageNette is a 10-class subset of ImageNet preserving native 224x224 resolution
and the real ImageNet-1K label distribution.

This addresses the scientific concern: testing ResNet50/VGG on CIFAR-10 is a
task mismatch. ImageNette provides the proper evaluation on ImageNet data.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import sys, tarfile

sys.path.insert(0, str(Path(__file__).parent))
from fixed_point import FixedPointConfig, convert_model_to_fixed_point

# ImageNette wnid -> ImageNet-1K class index mapping
# From https://github.com/fastai/imagenette
IMAGENETTE_WNID_TO_IDX = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer spaniel
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = Path('./data')


def extract_imagenette(tarfile_path, extract_to):
    """Extract ImageNette tarball."""
    print(f"  Extracting {tarfile_path}...")
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def load_imagenette_val(batch_size=32):
    """Load ImageNette validation set with proper ImageNet preprocessing."""
    tar_path = DATA_DIR / 'imagenette2-160.tgz'
    val_dir = DATA_DIR / 'imagenette2-160' / 'val'

    # Extract if needed
    if not val_dir.exists():
        if not tar_path.exists():
            raise RuntimeError(f"Download ImageNette first: {tar_path} not found")
        extract_imagenette(tar_path, DATA_DIR)

    # Standard ImageNet preprocessing: Resize to 256, CenterCrop to 224
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(val_dir, transform=transform)

    # Build mapping from ImageFolder's actual class indices to ImageNet-1K indices.
    # ImageFolder assigns labels by sorted(dataset.classes) — build from the folders
    # that actually exist so a partial dataset still maps correctly.
    folder_to_imagenet_idx = {}
    for folder_idx, wnid in enumerate(dataset.classes):
        if wnid in IMAGENETTE_WNID_TO_IDX:
            folder_to_imagenet_idx[folder_idx] = IMAGENETTE_WNID_TO_IDX[wnid]
        else:
            print(f"  WARNING: folder {wnid} not in wnid map, skipping")

    print(f"Loaded ImageNette validation: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"  wnids: {dataset.classes}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, folder_to_imagenet_idx


def load_model(name):
    """Load pretrained ImageNet model (1000 classes, unmodified)."""
    if name == 'ResNet50':
        return models.resnet50(weights='IMAGENET1K_V1')
    elif name == 'ResNet101':
        return models.resnet101(weights='IMAGENET1K_V1')
    elif name == 'ResNet152':
        return models.resnet152(weights='IMAGENET1K_V1')
    elif name == 'VGG16':
        return models.vgg16(weights='IMAGENET1K_V1')
    elif name == 'VGG19':
        return models.vgg19(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate_top1(model, loader, folder_to_imagenet_idx):
    """
    Evaluate top-1 accuracy on ImageNette.

    ImageFolder gives labels 0-9 (alphabetical wnid order).
    Model outputs 1000-class logits.
    We map folder label -> ImageNet index, then check if model's top-1 matches.
    """
    model.eval().to(DEVICE)
    correct = total = 0

    for images, folder_labels in loader:
        images = images.to(DEVICE)
        logits = model(images)  # (B, 1000)

        # Map folder labels to ImageNet indices
        imagenet_labels = torch.tensor([folder_to_imagenet_idx[lbl.item()]
                                        for lbl in folder_labels], device=DEVICE)

        preds = logits.argmax(dim=1)
        correct += (preds == imagenet_labels).sum().item()
        total += images.size(0)

    return 100.0 * correct / total


def test_model(name, loader, folder_to_imagenet_idx, bitwidths=(16, 20, 24, 28)):
    """Test one model at multiple bitwidths."""
    print(f"\n{'='*80}\n{name} on ImageNette (real ImageNet distribution, 224x224)\n{'='*80}")

    model = load_model(name)

    # Float baseline
    print("  Evaluating float32 baseline...")
    float_acc = evaluate_top1(model, loader, folder_to_imagenet_idx)
    print(f"  Float32 (ell=32): {float_acc:.2f}%")
    results = [(name, 32, float_acc)]

    # Quantized models
    for ell in bitwidths:
        print(f"  Quantizing to ell={ell}...")
        config = FixedPointConfig(total_bits=ell, frac_bits=16)
        qmodel = convert_model_to_fixed_point(model, config)

        acc = evaluate_top1(qmodel, loader, folder_to_imagenet_idx)
        print(f"  ell={ell}: {acc:.2f}%")
        results.append((name, ell, acc))

        del qmodel
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return results


# This is a script helper, not a pytest fixture-based test function.
test_model.__test__ = False


def main():
    print(f"Device: {DEVICE}")
    print("="*80)
    print("ImageNette Accuracy Test (Real ImageNet Distribution)")
    print("="*80)

    loader, folder_to_imagenet_idx = load_imagenette_val(batch_size=32)

    all_results = []
    for model_name in ['ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'VGG19']:
        try:
            results = test_model(model_name, loader, folder_to_imagenet_idx)
            all_results.extend(results)
        except Exception as e:
            print(f"\n!! {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    out_path = Path('results/e0_imagenette_real.csv')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('model,ell,top1_accuracy_pct\n')
        for model, ell, acc in all_results:
            f.write(f'{model},{ell},{acc:.2f}\n')

    print(f"\n{'='*80}")
    print(f"Saved results -> {out_path}")
    print("="*80)


if __name__ == '__main__':
    main()
