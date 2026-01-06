"""
Simplified Diffusion Policy Implementation
Denoising diffusion for action prediction.

Algorithm:
1. Add noise to expert actions over T timesteps (forward diffusion)
2. Train denoising network to predict noise at each timestep
3. At inference, start from noise and iteratively denoise to get action
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
from typing import Generator, Dict, Any
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings using sinusoidal functions."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingNetwork(nn.Module):
    """Network that predicts noise given noisy action, state, and timestep."""
    def __init__(self, state_dim=7, action_dim=4, hidden_size=256, time_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )
    
    def forward(self, noisy_action, state, timestep):
        t_emb = self.time_mlp(timestep)
        x = torch.cat([noisy_action, state, t_emb], dim=-1)
        return self.network(x)


class NoiseScheduler:
    """Simple linear noise scheduler for diffusion."""
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x, t, noise):
        """Add noise to x at timestep t."""
        device = x.device
        # Move scheduler tensors to correct device and index with CPU tensor
        t_cpu = t.cpu()
        sqrt_alpha = self.sqrt_alphas_cumprod[t_cpu].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(device)
        
        if sqrt_alpha.dim() == 0:
            sqrt_alpha = sqrt_alpha.unsqueeze(0)
        if sqrt_one_minus_alpha.dim() == 0:
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(0)
        
        return sqrt_alpha[:, None] * x + sqrt_one_minus_alpha[:, None] * noise


def train_diffusion_live(
    learning_rate: float = 0.0005,
    batch_size: int = 64,
    num_epochs: int = 100,
    hidden_size: int = 256,
    num_timesteps: int = 50
) -> Generator[Dict[str, Any], None, None]:
    """
    Train Diffusion Policy with live updates.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = np.load("ml/data/demos.npz")
    obs = data['obs']
    acts = data['actions']
    
    with open("ml/data/norm_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    
    obs_norm = (obs - norm_stats['obs_mean']) / (norm_stats['obs_std'] + 1e-8)
    acts_norm = (acts - norm_stats['action_mean']) / (norm_stats['action_std'] + 1e-8)
    
    obs_t = torch.FloatTensor(obs_norm).to(device)
    acts_t = torch.FloatTensor(acts_norm).to(device)
    
    # Model and scheduler
    model = DenoisingNetwork(state_dim=7, action_dim=4, hidden_size=hidden_size).to(device)
    scheduler = NoiseScheduler(num_timesteps=num_timesteps)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    n_samples = len(obs_t)
    losses = []
    
    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        obs_shuffled = obs_t[perm]
        acts_shuffled = acts_t[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_obs = obs_shuffled[i:i+batch_size]
            batch_acts = acts_shuffled[i:i+batch_size]
            batch_size_actual = len(batch_obs)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (batch_size_actual,), device=device)
            
            # Generate noise and add to actions
            noise = torch.randn_like(batch_acts)
            noisy_acts = scheduler.add_noise(batch_acts, t, noise)
            
            # Predict noise
            optimizer.zero_grad()
            pred_noise = model(noisy_acts, batch_obs, t.float())
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        yield {
            'epoch': epoch + 1,
            'total_epochs': num_epochs,
            'loss': avg_loss,
            'losses': losses.copy(),
            'progress': (epoch + 1) / num_epochs * 100,
            'done': False
        }
    
    # Save model
    torch.save(model.state_dict(), 'ml/diffusion_policy.pth')
    
    yield {
        'epoch': num_epochs,
        'total_epochs': num_epochs,
        'loss': losses[-1],
        'losses': losses,
        'progress': 100,
        'done': True,
        'model_path': 'ml/diffusion_policy.pth'
    }


if __name__ == "__main__":
    print("Testing Diffusion Policy training...")
    for update in train_diffusion_live(num_epochs=10):
        print(f"Epoch {update['epoch']}/{update['total_epochs']} | Loss={update['loss']:.4f}")
    print("Done!")
