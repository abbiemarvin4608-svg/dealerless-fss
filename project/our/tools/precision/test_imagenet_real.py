#!/usr/bin/env python3
"""
Comprehensive precision study with REAL ImageNet data.

Tests:
1. VGG-16/19 (no BatchNorm, large activation ranges)
2. ResNet50/101/152 (re-test with real images vs random)
3. Uses ImageNette (10-class ImageNet subset, real photos)

Goal: Eliminate random-input caveat and determine true ℓ requirements.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import tarfile

from fixed_point import FixedPointConfig, convert_model_to_fixed_point


# ImageNette wnid -> ImageNet index mapping
# From https://github.com/fastai/imagenette
IMAGENETTE_WNID_TO_IDX = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}


def extract_imagenette(tarfile_path, extract_to):
    """Extract ImageNette tarball."""
    print(f"Extracting {tarfile_path}...")
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def load_imagenette_val(data_dir, img_size=224):
    """Load ImageNette validation set with ImageNet preprocessing."""
    val_dir = Path(data_dir) / 'imagenette2-160' / 'val'

    if not val_dir.exists():
        raise RuntimeError(f"ImageNette val dir not found at {val_dir}. "
                           f"Extract imagenette2-160.tgz first.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(val_dir, transform=transform)

    # Remap class indices from ImageFolder's alphabetical order to ImageNet indices
    # ImageFolder assigns 0-9 to alphabetically sorted wnids
    # We need to map to real ImageNet indices
    wnids = sorted(IMAGENETTE_WNID_TO_IDX.keys())
    wnid_to_folder_idx = {w: i for i, w in enumerate(wnids)}

    print(f"Loaded ImageNette val: {len(dataset)} images, 10 classes")
    print(f"  Classes (wnid): {wnids}")

    return dataset, wnid_to_folder_idx


def measure_activation_ranges_real(model, dataloader, device='cpu', max_batches=50):
    """Measure activation ranges on real ImageNet images."""
    print(f"  Measuring activation ranges on real images (max {max_batches} batches)...")

    activations = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activations:
                activations[name] = {'min': float('inf'), 'max': float('-inf')}
            if isinstance(output, tuple):
                output = output[0]
            activations[name]['min'] = min(activations[name]['min'], output.min().item())
            activations[name]['max'] = max(activations[name]['max'], output.max().item())
        return fn

    # Register hooks on key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return activations


def analyze_activation_ranges(act_ranges, model_name):
    """Analyze activation ranges and determine minimum integer bits needed."""
    print(f"\n{'Layer':<50} {'Min':>10} {'Max':>10} {'Abs Max':>10}")
    print("-" * 82)

    max_abs = 0
    critical_layers = []

    for name, stats in sorted(act_ranges.items()):
        min_val = stats['min']
        max_val = stats['max']
        abs_max = max(abs(min_val), abs(max_val))

        # Print top layers by absolute magnitude
        if abs_max > 5:  # Threshold for interesting layers
            print(f"{name:<50} {min_val:>10.2f} {max_val:>10.2f} {abs_max:>10.2f}")
            max_abs = max(max_abs, abs_max)
            critical_layers.append((name, abs_max))

    # Compute required integer bits
    import math
    if max_abs > 0:
        min_int_bits = math.ceil(math.log2(max_abs)) + 1  # +1 for sign bit
    else:
        min_int_bits = 1

    print("\n" + "="*82)
    print(f"{model_name} Summary (REAL ImageNet data):")
    print(f"  Max absolute activation: {max_abs:.2f}")
    print(f"  Minimum integer bits required: {min_int_bits}")
    print(f"  With s=16 fractional bits: ℓ_min = {min_int_bits + 16}")
    print(f"  ℓ=20 (4 int bits) gives range: [-8, 8)")

    if max_abs > 8:
        print(f"  → CONCLUSION: ℓ=20 INSUFFICIENT (real activations exceed ±8)")
        print(f"  → Recommend ℓ ≥ {min_int_bits + 16} for faithful representation")
    else:
        print(f"  → ℓ=20 may be sufficient (real activations within ±8)")

    print("="*82)

    return max_abs, min_int_bits


def precision_sweep_real_data(model, model_name, dataloader, wnid_to_folder_idx,
                               bitwidths, frac_bits=16, device='cpu'):
    """
    Sweep bitwidths on real ImageNet data.

    Measures both:
    1. Top-1 agreement (quantized vs float32 outputs)
    2. Real accuracy (if labels are available)
    """
    print(f"\n{model_name}: Precision sweep on real ImageNet images")

    # Prepare ImageNet index mapping for accuracy computation
    folder_idx_to_imagenet_idx = {}
    for wnid, imagenet_idx in IMAGENETTE_WNID_TO_IDX.items():
        folder_idx = wnid_to_folder_idx[wnid]
        folder_idx_to_imagenet_idx[folder_idx] = imagenet_idx

    # Get float32 baseline outputs
    model.eval()
    float_outputs = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            out = model(inputs)
            float_outputs.append(out)
            # Remap labels from ImageFolder idx to ImageNet idx
            imagenet_labels = torch.tensor([folder_idx_to_imagenet_idx[l.item()] for l in labels])
            true_labels.append(imagenet_labels)

    # Float32 accuracy
    float_correct = 0
    float_total = 0
    for out, labels in zip(float_outputs, true_labels):
        preds = out.argmax(dim=1).cpu()
        float_correct += (preds == labels).sum().item()
        float_total += labels.size(0)
    float_acc = 100.0 * float_correct / float_total
    print(f"  Float32 accuracy: {float_acc:.2f}%")

    results = []
    results.append({
        'model': model_name,
        'ell': 'float32',
        'int_bits': '-',
        'frac_bits': '-',
        'top1_agreement_pct': 100.0,
        'real_accuracy_pct': float_acc
    })

    for ell in bitwidths:
        print(f"  Testing ℓ={ell}, s={frac_bits}...")
        config = FixedPointConfig(total_bits=ell, frac_bits=frac_bits)
        fp_model = convert_model_to_fixed_point(model, config)
        fp_model.to(device)
        fp_model.eval()

        # Measure output divergence and real accuracy
        total_agreement = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                fp_out = fp_model(inputs)
                float_out = float_outputs[i]
                imagenet_labels = true_labels[i]

                # Top-1 agreement
                fp_preds = fp_out.argmax(dim=1).cpu()
                float_preds = float_out.argmax(dim=1).cpu()
                total_agreement += (fp_preds == float_preds).sum().item()

                # Real accuracy
                total_correct += (fp_preds == imagenet_labels).sum().item()
                total_samples += labels.size(0)

        top1_agreement = 100.0 * total_agreement / total_samples
        real_accuracy = 100.0 * total_correct / total_samples

        results.append({
            'model': model_name,
            'ell': ell,
            'int_bits': ell - frac_bits,
            'frac_bits': frac_bits,
            'top1_agreement_pct': top1_agreement,
            'real_accuracy_pct': real_accuracy
        })

        print(f"    Top-1 agreement: {top1_agreement:.1f}%, Real accuracy: {real_accuracy:.1f}%")

    return results


def main():
    device = 'cpu'
    bitwidths = [16, 20, 24, 28, 32]
    frac_bits = 16
    data_dir = Path('data')

    print("="*82)
    print("Comprehensive Precision Study with REAL ImageNet Data")
    print("="*82)
    print("Dataset: ImageNette (10-class ImageNet subset)")
    print("Models: VGG-16/19 (no BN), ResNet50/101/152")
    print("="*82)

    # Extract ImageNette if needed
    tar_path = data_dir / 'imagenette2-160.tgz'
    extract_dir = data_dir / 'imagenette2-160'

    if not extract_dir.exists():
        if not tar_path.exists():
            print(f"ERROR: {tar_path} not found. Download it first:")
            print("  wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz")
            return
        extract_imagenette(tar_path, data_dir)

    # Load dataset
    dataset, wnid_to_folder_idx = load_imagenette_val(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # Models to test: VGG first (no BN, largest activations expected)
    models_to_test = [
        ('VGG16', models.vgg16(pretrained=True)),
        ('VGG19', models.vgg19(pretrained=True)),
        ('ResNet50', models.resnet50(pretrained=True)),
        ('ResNet101', models.resnet101(pretrained=True)),
        ('ResNet152', models.resnet152(pretrained=True)),
    ]

    all_results = []

    for model_name, model in models_to_test:
        print(f"\n{'='*82}")
        print(f"Testing: {model_name}")
        print(f"{'='*82}")

        model.to(device)
        model.eval()

        # Step 1: Measure activation ranges on real data
        act_ranges = measure_activation_ranges_real(model, dataloader, device, max_batches=50)
        max_abs, min_int_bits = analyze_activation_ranges(act_ranges, model_name)

        # Step 2: Precision sweep
        sweep_results = precision_sweep_real_data(
            model, model_name, dataloader, wnid_to_folder_idx,
            bitwidths, frac_bits, device
        )
        all_results.extend(sweep_results)

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print()

    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    output_path = Path('results/e0_imagenet_real.csv')
    df.to_csv(output_path, index=False)

    print("\n" + "="*82)
    print("Results Summary")
    print("="*82)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
