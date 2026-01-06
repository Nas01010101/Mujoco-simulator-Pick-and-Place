import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ml.dataset import DemoDataset

class BCPolicy(nn.Module):
    """
    Behavior Cloning policy: MLP mapping observations to actions.
    Architecture: 7D obs -> [256, 256, 256] -> 4D action
    """
    def __init__(self, obs_dim=7, action_dim=4, hidden_size=256):
        super(BCPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
    
    def forward(self, obs):
        return self.network(obs)


def train_bc(data_path="ml/data/demos.npz", 
             model_path="ml/bc_policy.pth",
             stats_path="ml/data/norm_stats.pkl",
             plot_path="assets/plots/training_loss.png",
             batch_size=64,
             learning_rate=1e-3,
             num_epochs=100,
             device='cpu'):
    """
    Train a behavior cloning policy using expert demonstrations.
    
    Args:
        data_path: Path to demonstration data
        model_path: Where to save trained model
        stats_path: Path to normalization statistics
        plot_path: Where to save training loss plot
        batch_size: Training batch size
        learning_rate: Adam learning rate
        num_epochs: Number of training epochs
        device: 'cpu' or 'cuda'
    """
    
    # Create output directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = DemoDataset(
        data_path=data_path,
        normalize=True,
        stats_path=stats_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for macOS compatibility
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = BCPolicy(obs_dim=7, action_dim=4, hidden_size=256)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            # Forward pass
            predicted_actions = model(obs_batch)
            loss = criterion(predicted_actions, action_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'obs_dim': 7,
        'action_dim': 4,
        'hidden_size': 256,
        'stats_path': stats_path
    }, model_path)
    
    # Plot training loss
    print(f"Saving training loss plot to {plot_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Behavior Cloning Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"Plot saved to: {plot_path}")
    
    return model, loss_history


if __name__ == "__main__":
    # Check if MPS (Apple Silicon GPU) is available
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    train_bc(
        data_path="ml/data/demos.npz",
        model_path="ml/bc_policy.pth",
        stats_path="ml/data/norm_stats.pkl",
        plot_path="assets/plots/training_loss.png",
        batch_size=64,
        learning_rate=5e-4,  # Reduced for more stable training
        num_epochs=200,      # Increased epochs
        device=device
    )
