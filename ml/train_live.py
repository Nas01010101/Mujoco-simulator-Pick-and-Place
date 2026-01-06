"""
Live Training Module
Provides streaming training with real-time loss updates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from typing import Generator, Dict, Any

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_bc import BCPolicy


def load_demos(path: str = "ml/data/demos.npz"):
    """Load demonstration data."""
    data = np.load(path)
    return data['obs'], data['actions']


def load_norm_stats(path: str = "ml/data/norm_stats.pkl"):
    """Load normalization statistics."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_bc_live(
    learning_rate: float = 0.0005,
    batch_size: int = 64,
    num_epochs: int = 100,
    hidden_size: int = 256
) -> Generator[Dict[str, Any], None, None]:
    """
    Train BC policy with live updates.
    
    Yields dictionaries with training progress:
        - epoch: current epoch
        - loss: current loss value
        - progress: percentage complete (0-100)
        - done: whether training is complete
    """
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    obs, acts = load_demos()
    norm_stats = load_norm_stats()
    
    # Normalize
    obs_norm = (obs - norm_stats['obs_mean']) / (norm_stats['obs_std'] + 1e-8)
    acts_norm = (acts - norm_stats['action_mean']) / (norm_stats['action_std'] + 1e-8)
    
    # Tensors
    obs_tensor = torch.FloatTensor(obs_norm).to(device)
    acts_tensor = torch.FloatTensor(acts_norm).to(device)
    
    # Model
    model = BCPolicy(obs_dim=7, action_dim=4, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    n_samples = len(obs_tensor)
    losses = []
    
    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        obs_shuffled = obs_tensor[perm]
        acts_shuffled = acts_tensor[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_obs = obs_shuffled[i:i+batch_size]
            batch_acts = acts_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_acts = model(batch_obs)
            loss = criterion(pred_acts, batch_acts)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Yield progress
        yield {
            'epoch': epoch + 1,
            'total_epochs': num_epochs,
            'loss': avg_loss,
            'losses': losses.copy(),
            'progress': (epoch + 1) / num_epochs * 100,
            'done': False
        }
    
    # Save model
    torch.save(model.state_dict(), 'ml/bc_policy.pth')
    
    # Final yield
    yield {
        'epoch': num_epochs,
        'total_epochs': num_epochs,
        'loss': losses[-1],
        'losses': losses,
        'progress': 100,
        'done': True,
        'model_path': 'ml/bc_policy.pth'
    }


if __name__ == "__main__":
    # Test live training
    print("Testing live training...")
    for update in train_bc_live(num_epochs=10):
        print(f"Epoch {update['epoch']}/{update['total_epochs']}: Loss={update['loss']:.4f}")
    print("Done!")
