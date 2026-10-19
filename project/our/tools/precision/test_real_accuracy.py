#!/usr/bin/env python3
"""
Test real top-1 accuracy on actual datasets (CIFAR-10, MNIST) with fixed-point quantization.
Measures the true precision cliff, not just activation ranges.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent to path for fixed_point module
sys.path.insert(0, str(Path(__file__).parent))
from fixed_point import FixedPointConfig, convert_model_to_fixed_point

def load_cifar10(batch_size=128):
    """Load CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader

def load_mnist(batch_size=128):
    """Load MNIST test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testset, testloader

def create_mnist_cnn():
    """Simple CNN for MNIST (LeNet-style)"""
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.maxpool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return MNISTNet()

def train_mnist_model(device='cpu', epochs=3):
    """Quick train a MNIST model (or load if exists)"""
    model_path = Path('./data/mnist_cnn.pt')

    if model_path.exists():
        print("  Loading cached MNIST model...")
        model = create_mnist_cnn()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print("  Training MNIST model (3 epochs, quick baseline)...")
    model = create_mnist_cnn().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'    Epoch {epoch+1}, batch {i+1}: loss {running_loss/100:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), model_path)
    print(f"  Saved to {model_path}")
    return model

def create_resnet20_cifar():
    """ResNet20 for CIFAR-10"""
    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super().__init__()
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = torch.nn.functional.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    return ResNet(BasicBlock, [3, 3, 3])

def train_resnet20_cifar(device='cpu', epochs=50):
    """Train ResNet20 on CIFAR-10 (or load if exists)"""
    model_path = Path('./data/resnet20_cifar10.pt')

    if model_path.exists():
        print("  Loading cached ResNet20-CIFAR10 model...")
        model = create_resnet20_cifar()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print("  Training ResNet20 on CIFAR-10 (50 epochs)...")
    model = create_resnet20_cifar().to(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}: loss {running_loss/len(trainloader):.3f}')

    torch.save(model.state_dict(), model_path)
    print(f"  Saved to {model_path}")
    return model

def evaluate_accuracy(model, dataloader, device='cpu'):
    """Evaluate top-1 accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total

def test_model_at_bitwidths(model_name, model, dataloader, device='cpu', test_bitwidths=[16, 20, 24, 28, 32]):
    """Test a model at different bitwidths"""
    print(f"\n{model_name}:")

    # Float baseline
    float_acc = evaluate_accuracy(model, dataloader, device)
    print(f"  Float32 baseline: {float_acc:.2f}%")

    results = [(32, float_acc)]  # Treat float as ell=32 reference

    for ell in test_bitwidths:
        if ell == 32:
            continue  # Already have float baseline

        config = FixedPointConfig(total_bits=ell, frac_bits=16)
        quant_model = convert_model_to_fixed_point(model, config)
        quant_model.to(device)

        acc = evaluate_accuracy(quant_model, dataloader, device)
        print(f"  ℓ={ell}: {acc:.2f}%")
        results.append((ell, acc))

    return results


# This is a script helper, not a pytest fixture-based test function.
test_model_at_bitwidths.__test__ = False

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("="*80)
    print("Real Dataset Accuracy Test")
    print("="*80)

    all_results = []

    # 1. MNIST
    print("\n[1/6] MNIST CNN")
    mnist_dataset, mnist_loader = load_mnist()
    mnist_model = train_mnist_model(device)
    mnist_results = test_model_at_bitwidths('MNIST-CNN', mnist_model, mnist_loader, device)
    for ell, acc in mnist_results:
        all_results.append(('MNIST-CNN', ell, acc))

    # 2. ResNet20 on CIFAR-10
    print("\n[2/6] ResNet20-CIFAR10")
    cifar_loader = load_cifar10()
    resnet20_model = train_resnet20_cifar(device)
    resnet20_results = test_model_at_bitwidths('ResNet20-CIFAR10', resnet20_model, cifar_loader, device)
    for ell, acc in resnet20_results:
        all_results.append(('ResNet20-CIFAR10', ell, acc))

    # 3-5. ImageNet models on CIFAR-10 (transfer learning test)
    # Use pretrained ImageNet models, test on CIFAR-10 with proper transforms
    print("\n[3/6] ResNet50-ImageNet (tested on CIFAR-10 with ImageNet normalization)")

    # CIFAR-10 with ImageNet preprocessing (resize to 224x224)
    transform_imagenet = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar_imagenet = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_imagenet
    )
    cifar_imagenet_loader = DataLoader(cifar_imagenet, batch_size=64, shuffle=False, num_workers=2)

    # Note: These are 1000-class ImageNet models tested on 10-class CIFAR
    # The accuracy will be low, but the RELATIVE degradation with ℓ is what matters

    print("  Loading ResNet50 (ImageNet pretrained)...")
    resnet50 = models.resnet50(weights='IMAGENET1K_V1')
    # Map 1000 classes to 10 via simple linear projection (just for relative comparison)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)
    nn.init.xavier_uniform_(resnet50.fc.weight)
    resnet50_results = test_model_at_bitwidths('ResNet50-ImageNet', resnet50, cifar_imagenet_loader, device, [16, 20, 24, 28])
    for ell, acc in resnet50_results:
        all_results.append(('ResNet50-ImageNet', ell, acc))

    print("\n[4/6] ResNet101-ImageNet")
    print("  Loading ResNet101 (ImageNet pretrained)...")
    resnet101 = models.resnet101(weights='IMAGENET1K_V1')
    resnet101.fc = nn.Linear(resnet101.fc.in_features, 10)
    nn.init.xavier_uniform_(resnet101.fc.weight)
    resnet101_results = test_model_at_bitwidths('ResNet101-ImageNet', resnet101, cifar_imagenet_loader, device, [16, 20, 24, 28])
    for ell, acc in resnet101_results:
        all_results.append(('ResNet101-ImageNet', ell, acc))

    print("\n[5/6] ResNet152-ImageNet")
    print("  Loading ResNet152 (ImageNet pretrained)...")
    resnet152 = models.resnet152(weights='IMAGENET1K_V1')
    resnet152.fc = nn.Linear(resnet152.fc.in_features, 10)
    nn.init.xavier_uniform_(resnet152.fc.weight)
    resnet152_results = test_model_at_bitwidths('ResNet152-ImageNet', resnet152, cifar_imagenet_loader, device, [16, 20, 24, 28])
    for ell, acc in resnet152_results:
        all_results.append(('ResNet152-ImageNet', ell, acc))

    print("\n[6/6] VGG16-ImageNet")
    print("  Loading VGG16 (ImageNet pretrained)...")
    vgg16 = models.vgg16(weights='IMAGENET1K_V1')
    vgg16.classifier[6] = nn.Linear(4096, 10)
    nn.init.xavier_uniform_(vgg16.classifier[6].weight)
    vgg16_results = test_model_at_bitwidths('VGG16-ImageNet', vgg16, cifar_imagenet_loader, device, [16, 20, 24, 28])
    for ell, acc in vgg16_results:
        all_results.append(('VGG16-ImageNet', ell, acc))

    # Save results
    print("\n" + "="*80)
    print("Saving results to results/e0_real_accuracy.csv")
    output_path = Path('results/e0_real_accuracy.csv')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('model,ell,accuracy_pct\n')
        for model_name, ell, acc in all_results:
            f.write(f'{model_name},{ell},{acc:.2f}\n')

    print("Done!")
    print("="*80)

if __name__ == '__main__':
    main()
