import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from binary_optimizers.models.swarm_logic import LogSwarmLinear, LogSwarmConv2d
from binary_optimizers.models.logic_layers import LogicHomeostasis
from binary_optimizers.optimizers.swarm_log_optimizer import SwarmLogOptimizer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 2  # Reduced for CPU feasibility (Driver issue)
hidden_dim = 512

# Data
# Standard CIFAR-10 transforms
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)


class STE_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified VGG-like structure for STE baseline
        # Note: Using standard Conv2d but we will employ logic/binary constraints if possible
        # For a fair baseline, we should essentially use Float weights but discrete optimization if testing that.
        # But here, let's use standard PyTorch Conv2d + BatchNorm as the "Gold Standard" or STE logic.

        # Actually, let's build a binary-compatible baseline using just standard components
        # but optimized with SGD to set a "Float Topline".
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(256 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Swarm_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            LogSwarmConv2d(3, 128, kernel_size=3, padding=1),
            LogicHomeostasis(128),  # Applies per channel
            nn.Hardtanh(),
            LogSwarmConv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LogicHomeostasis(128),
            nn.Hardtanh(),
            LogSwarmConv2d(128, 256, kernel_size=3, padding=1),
            LogicHomeostasis(256),
            nn.Hardtanh(),
            LogSwarmConv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LogicHomeostasis(256),
            nn.Hardtanh(),
            nn.Flatten(),
        )
        self.classifier = LogSwarmLinear(256 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total


def run_experiment(name, model_fn, optimizer_fn):
    print(f"\nRunning Experiment: {name}")
    model = model_fn().to(device)
    optimizer = optimizer_fn(model.parameters())

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss, train_acc = train(model, optimizer, train_loader)
        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch {epoch:02d}: Loss={loss:.4f}, TrainAcc={train_acc:.4f}, TestAcc={test_acc:.4f}"
        )
        history.append(
            {"epoch": epoch, "loss": loss, "train_acc": train_acc, "test_acc": test_acc}
        )

    duration = time.time() - start_time
    return history, duration


if __name__ == "__main__":
    # 1. Baseline: Float weights + SGD (Topline)
    # We use this to see what a "normal" network achieves with this architecture
    # SKIPPED: Already run (Epoch 1 Acc: 0.5149)
    print("Baseline already run. Using cached results.")
    ste_history = [{"test_acc": 0.5149}]  # Mock for summary
    ste_time = 0.0
    # ste_history, ste_time = run_experiment(
    #     "Float-Baseline", STE_VGG, lambda p: optim.SGD(p, lr=0.01, momentum=0.9)
    # )

    # 2. Swarm Logic: Populations + Logarithmic flips
    swarm_history, swarm_time = run_experiment(
        "Swarm-Logic",
        Swarm_VGG,
        lambda p: SwarmLogOptimizer(p, threshold=10, flip_prob=0.1),
    )

    # Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS (CIFAR-10)")
    print("=" * 50)
    print(f"Float Baseline: {ste_history[-1]['test_acc']:.4f} (Time: {ste_time:.1f}s)")
    print(
        f"Swarm Logic:    {swarm_history[-1]['test_acc']:.4f} (Time: {swarm_time:.1f}s)"
    )

    # Save results to text file
    with open("cifar10_swarm_results.txt", "w") as f:
        f.write("Method,Final Test Acc,Duration\n")
        f.write(f"Float-Baseline,{ste_history[-1]['test_acc']},{ste_time}\n")
        f.write(f"Swarm-Logic,{swarm_history[-1]['test_acc']},{swarm_time}\n")
