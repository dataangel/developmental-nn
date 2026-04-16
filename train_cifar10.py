import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os
import numpy as np
from datetime import datetime

# ============================================
# 1. Exactly the same structure as GrownNetworkMNIST (no alignment layer)
# ============================================
class GrownNetworkCIFAR10(nn.Module):
    """Exactly the same as GrownNetworkMNIST, only input dimension changed to 3072 (32*32*3)"""
    def __init__(self, network_json_path, input_size=3072, num_classes=10):
        super().__init__()
        
        with open(network_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract neurons (exactly the same as MNIST version)
        neuron_ids = [n['id'] for n in data['nodes'] if n['type'] == 'neuron']
        n_neurons = len(neuron_ids)
        
        print(f"\n🧠 Loading biomimetic network (native adaptability test):")
        print(f"   Number of neurons: {n_neurons}")
        print(f"   Number of synaptic connections: {sum(1 for e in data['edges'] if e['relation']=='synapse')}")
        print(f"   Input dimension: {input_size} (32×32×3)")
        
        # Build synapse matrix (exactly the same as MNIST)
        weights = torch.zeros(n_neurons, n_neurons)
        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        
        for edge in data['edges']:
            if edge['relation'] == 'synapse':
                u, v = edge['source'], edge['target']
                if u in id_to_idx and v in id_to_idx:
                    weights[id_to_idx[u], id_to_idx[v]] = edge['weight']
        
        row_sums = weights.sum(dim=1, keepdim=True)
        weights = weights / (row_sums + 1e-8)
        
        self.register_buffer('fixed_weights', weights)
        self.n_neurons = n_neurons
        
        # Input projection layer (dimension increased: 3072 → n_neurons)
        self.input_proj = nn.Linear(input_size, n_neurons, bias=False)
        
        # Output layer (exactly the same as MNIST)
        self.output = nn.Linear(n_neurons, num_classes, bias=False)
        
        self.fixed_weights.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {trainable:,}")
        print(f"   ⚠️ No alignment layer, directly flatten 32×32×3=3072 dimensional input")
        
    def forward(self, x):
        # Directly flatten, no shape adaptation
        x = x.view(x.size(0), -1)           # [B, 3, 32, 32] → [B, 3072]
        x = torch.relu(self.input_proj(x))  # Input projection
        x = x @ self.fixed_weights.T        # Through synaptic network
        x = self.output(x)                  # Classification
        return x


# ============================================
# 2. Training functions (exactly the same as MNIST version)
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
# 3. Control experiment: Random projection layer
# ============================================
def test_random_projection(network_json_path, test_loader, device):
    """Test accuracy with random projection layer"""
    model = GrownNetworkCIFAR10(network_json_path)
    
    # Randomize projection layers
    model.input_proj.weight.data = torch.randn_like(model.input_proj.weight)
    model.output.weight.data = torch.randn_like(model.output.weight)
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100. * correct / total


# ============================================
# 4. Main program
# ============================================
if __name__ == "__main__":
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    NETWORK_JSON = os.path.join(script_dir, "grown_neural_network.json")
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print("=" * 60)
    print("🧪 Native Adaptability Test: No alignment layer, directly flatten CIFAR-10")
    print("=" * 60)
    
    if not os.path.exists(NETWORK_JSON):
        print(f"❌ Network structure file not found: {NETWORK_JSON}")
        exit(1)
    
    # CIFAR-10 data (no size modification)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("\n📦 Loading CIFAR-10 (keeping original 32×32×3)...")
    data_dir = os.path.join(script_dir, "cifar10_data")
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = GrownNetworkCIFAR10(NETWORK_JSON).to(device)
    
    # Test random baseline
    print("\n" + "=" * 60)
    print("📊 Random Projection Layer Baseline Test")
    random_acc = test_random_projection(NETWORK_JSON, test_loader, device)
    print(f"   Random projection accuracy: {random_acc:.2f}% (CIFAR-10 random baseline ~10%)")
    
    # Test untrained state
    print("\n" + "=" * 60)
    print("📊 Untrained State Evaluation (Native Adaptability)")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"   Test accuracy: {test_acc:.2f}%")
    
    
    # Training
    print("\n" + "=" * 60)
    print("🚀 Starting Training")
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 Native Adaptability Test Results")
    print(f"   Random baseline: {random_acc:.2f}%")
    print(f"   Untrained accuracy: {history['test_acc'][0]:.2f}%")
    print(f"   Best after training: {best_acc:.2f}%")
    print(f"   Training improvement: {best_acc - history['test_acc'][0]:.2f}%")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("📋 Conclusion")
    if history['test_acc'][0] > random_acc + 5:
        print("   ✅ Biomimetic network has innate native adaptability to CIFAR-10!")
        print("      Works directly after flattening, no alignment layer needed")
    else:
        print("   ⚠️ Biomimetic network shows no obvious innate adaptability to CIFAR-10")
        print("      Suggestion: Add alignment layer (resize to 28×28, compress channels)")
    
    # Save results
    results = {
        'test_type': 'native_adaptability',
        'input_size': '32x32x3 = 3072',
        'alignment_layer': 'None',
        'random_baseline': random_acc,
        'untrained_accuracy': history['test_acc'][0],
        'best_accuracy': best_acc,
        'history': history
    }
    
    results_path = os.path.join(script_dir, f"cifar10_native_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # ============================================
    # Save model weights
    # ============================================
    save_path = os.path.join(script_dir, f"cifar10_native_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 1. Save full checkpoint (resumable training)
    checkpoint = {
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'history': history
    }
    torch.save(checkpoint, f"{save_path}.pth")

    # 2. Save model weights only (for inference)
    torch.save(model.state_dict(), f"{save_path}_weights.pth")

    # 3. Save metadata
    metadata = {
        'network_source': NETWORK_JSON,
        'architecture': {
            'input_size': 3072,
            'neurons': model.n_neurons,
            'num_classes': 10
        },
        'performance': {
            'best_accuracy': best_acc,
            'untrained_accuracy': history['test_acc'][0],
            'random_baseline': random_acc,
            'epochs': EPOCHS
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'weight_decay': 1e-4,
        }
    }
    
    metadata_path = os.path.join(script_dir, f"{save_path}_metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Model saved:")
    print(f"   {save_path}.pth (full checkpoint)")
    print(f"   {save_path}_weights.pth (weights only)")
    print(f"   {metadata_path} (metadata)")