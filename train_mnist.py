import torch
import torch.nn as nn
import json
import numpy as np
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================
# 1. Grown Network MNIST Classifier
# ============================================
class GrownNetworkMNIST(nn.Module):
    def __init__(self, network_json_path, input_size=784, num_classes=10):
        super().__init__()
        
        with open(network_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract neurons
        neuron_ids = [n['id'] for n in data['nodes'] if n['type'] == 'neuron']
        n_neurons = len(neuron_ids)
        
        print(f"\n🧠 Loaded from grown network:")
        print(f"   Number of neurons: {n_neurons}")
        print(f"   Total nodes: {len(data['nodes'])}")
        print(f"   Number of synaptic connections: {sum(1 for e in data['edges'] if e['relation']=='synapse')}")
        
        # Build connection matrix
        self.weights = torch.zeros(n_neurons, n_neurons)
        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        
        synapse_count = 0
        for edge in data['edges']:
            if edge['relation'] == 'synapse':
                u, v = edge['source'], edge['target']
                if u in id_to_idx and v in id_to_idx:
                    self.weights[id_to_idx[u], id_to_idx[v]] = edge['weight']
                    synapse_count += 1
        
        # Normalize weights
        row_sums = self.weights.sum(dim=1, keepdim=True)
        self.weights = self.weights / (row_sums + 1e-8)
        
        # Input projection layer (784 → n_neurons)
        self.input_proj = nn.Linear(input_size, n_neurons, bias=False)
        
        # Output layer (n_neurons → 10)
        self.output = nn.Linear(n_neurons, num_classes, bias=False)
        
        # Fix the grown synaptic weights
        self.register_buffer('fixed_weights', self.weights)
        self.fixed_weights.requires_grad = False
        
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.input_proj(x))
        x = x @ self.fixed_weights.T  # Pass through grown synaptic network
        x = self.output(x)
        return x


# ============================================
# 2. Training Functions
# ============================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


# ============================================
# Save Function
# ============================================
def save_trained_network(model, optimizer, epoch, best_acc, history, save_path):
    """Save complete training results"""
    
    # 1. Full checkpoint (resumable training)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'history': history
    }
    torch.save(checkpoint, f"{save_path}.pth")
    
    # 2. Model weights only (for deployment/inference)
    torch.save(model.state_dict(), f"{save_path}_weights.pth")
    
    # 3. Metadata
    metadata = {
        'network_source': 'Grown from mouse astrocyte GRN',
        'architecture': {
            'input_size': 784,
            'neurons': model.fixed_weights.shape[0],
            'synapses': int((model.fixed_weights > 0).sum().item()),
            'num_classes': 10
        },
        'performance': {
            'best_accuracy': best_acc,
            'epochs_trained': epoch,
            'untrained_accuracy': history['test_acc'][0] if history else None,
            'final_train_acc': history['train_acc'][-1] if history else None,
            'final_test_acc': history['test_acc'][-1] if history else None,
            'training_improvement': best_acc - history['test_acc'][0] if history else None
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_path}_metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 4. Synapse matrix
    np.save(f"{save_path}_synapses.npy", model.fixed_weights.cpu().numpy())
    
    # 5. Projection layer weights
    np.savez(
        f"{save_path}_projections.npz",
        input_proj=model.input_proj.weight.data.cpu().numpy(),
        output=model.output.weight.data.cpu().numpy()
    )
    
    print(f"\n💾 Training results saved:")
    print(f"   {save_path}.pth (full checkpoint)")
    print(f"   {save_path}_weights.pth (weights only)")
    print(f"   {save_path}_metadata.json (metadata)")
    print(f"   {save_path}_synapses.npy (synapse matrix)")
    print(f"   {save_path}_projections.npz (projection layers)")


def load_trained_network(network_json_path, weights_path, device='cpu'):
    """Load a trained network (for inference)"""
    model = GrownNetworkMNIST(network_json_path).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"📂 Loaded trained network: {weights_path}")
    return model


# ============================================
# 3. Main Program
# ============================================
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check network file
    network_path = os.path.join(script_dir, "grown_neural_network.json")
    if not os.path.exists(network_path):
        print(f"❌ Network file not found: {network_path}")
        print("   Please run the growth script first to generate the network")
        exit(1)
    
    # Load model
    model = GrownNetworkMNIST(network_path).to(device)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    print("\n📦 Checking MNIST data...")
    data_dir = os.path.join(script_dir, "mnist_data")
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    print(f"   Training set: {len(train_dataset)} images")
    print(f"   Test set: {len(test_dataset)} images")
    
    # Data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # ============================================
    # 4. Evaluate untrained state
    # ============================================
    print("\n" + "=" * 50)
    print("📊 Untrained State Evaluation")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"   Test accuracy: {test_acc:.2f}%")
    
    # Random baseline
    random_model = GrownNetworkMNIST(network_path).to(device)
    random_model.input_proj.weight.data = torch.randn_like(random_model.input_proj.weight)
    random_model.output.weight.data = torch.randn_like(random_model.output.weight)
    _, random_acc = evaluate(random_model, test_loader, criterion, device)
    print(f"   Random weight baseline: {random_acc:.2f}%")
    
    if test_acc > random_acc:
        print(f"   ✅ Grown network outperforms random network by {test_acc - random_acc:.2f}%")
    else:
        print(f"   ⚠️ Grown network shows no advantage")
    
    # ============================================
    # 5. Training
    # ============================================
    print("\n" + "=" * 50)
    print("🚀 Starting Training")
    
    epochs = 10
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
    
    # ============================================
    # 6. Final Results
    # ============================================
    print("\n" + "=" * 50)
    print("🎯 Final Results")
    print(f"   Best test accuracy: {best_acc:.2f}%")
    print(f"   Untrained accuracy: {history['test_acc'][0]:.2f}%")
    print(f"   Training improvement: {best_acc - history['test_acc'][0]:.2f}%")
    
    # Save training results (using relative paths)
    save_path = os.path.join(script_dir, "trained_grown_network")
    save_trained_network(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        best_acc=best_acc,
        history=history,
        save_path=save_path
    )
    
    # Save results JSON
    results_path = os.path.join(script_dir, "mnist_results.json")
    results = {
        'network_source': network_path,
        'best_accuracy': best_acc,
        'untrained_accuracy': history['test_acc'][0],
        'random_baseline': random_acc,
        'history': history
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")