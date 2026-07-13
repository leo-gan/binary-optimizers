import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from binary_optimizers.models.logic_layers import IntegerLogicLinear, LogicHomeostasis
from binary_optimizers.optimizers.logic_optimizer import IntegerVotingOptimizer
from binary_optimizers.models.bit_layers import BitLinearSTE
from binary_optimizers.optimizers.voting import VotingOptimizer
from binary_optimizers.models.swarm_logic import LogSwarmLinear
from binary_optimizers.optimizers.swarm_log_optimizer import SwarmLogOptimizer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 20  # Increased for convergence
hidden_dim = 512

# Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def create_ste_model():
    return nn.Sequential(
        nn.Flatten(),
        BitLinearSTE(28 * 28, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        BitLinearSTE(hidden_dim, 10),
        nn.BatchNorm1d(10),
    ).to(device)


def create_logic_model():
    return nn.Sequential(
        nn.Flatten(),
        IntegerLogicLinear(28 * 28, hidden_dim),
        LogicHomeostasis(hidden_dim),
        nn.ReLU(),
        IntegerLogicLinear(hidden_dim, 10),
        LogicHomeostasis(10),
    ).to(device)


def create_swarm_model():
    return nn.Sequential(
        nn.Flatten(),
        LogSwarmLinear(28 * 28, hidden_dim),
        LogicHomeostasis(hidden_dim),
        nn.ReLU(),
        LogSwarmLinear(hidden_dim, 10),
        LogicHomeostasis(10),
    ).to(device)


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
    model = model_fn()
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
    # 1. Baseline: STE + SGD
    ste_history, ste_time = run_experiment(
        "STE-Baseline", create_ste_model, lambda p: optim.SGD(p, lr=0.1, momentum=0.9)
    )

    # 2. Voting: Float-based Voting
    voting_history, voting_time = run_experiment(
        "Float-Voting", create_ste_model, lambda p: VotingOptimizer(p, lr=0.1)
    )

    # 3. Pure Logic: Integer Voting + Homeostasis
    logic_history, logic_time = run_experiment(
        "Pure-Logic",
        create_logic_model,
        lambda p: IntegerVotingOptimizer(p, threshold=15),
    )

    print(
        f"Pure Logic:   {logic_history[-1]['test_acc']:.4f} (Time: {logic_time:.1f}s)"
    )

    # 4. Swarm Logic: Populations + Logarithmic flips
    swarm_history, swarm_time = run_experiment(
        "Swarm-Logic",
        create_swarm_model,
        lambda p: SwarmLogOptimizer(p, threshold=5, flip_prob=0.1),
    )

    # Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"STE Baseline: {ste_history[-1]['test_acc']:.4f} (Time: {ste_time:.1f}s)")
    print(
        f"Float Voting: {voting_history[-1]['test_acc']:.4f} (Time: {voting_time:.1f}s)"
    )
    print(
        f"Pure Logic:   {logic_history[-1]['test_acc']:.4f} (Time: {logic_time:.1f}s)"
    )
    print(
        f"Swarm Logic:  {swarm_history[-1]['test_acc']:.4f} (Time: {swarm_time:.1f}s)"
    )

    # Save results to text file
    with open("logic_benchmark_results.txt", "w") as f:
        f.write("Method,Final Test Acc,Duration\n")
        f.write(f"STE-Baseline,{ste_history[-1]['test_acc']},{ste_time}\n")
        f.write(f"Float-Voting,{voting_history[-1]['test_acc']},{voting_time}\n")
        f.write(f"Pure-Logic,{logic_history[-1]['test_acc']},{logic_time}\n")
        f.write(f"Swarm-Logic,{swarm_history[-1]['test_acc']},{swarm_time}\n")
