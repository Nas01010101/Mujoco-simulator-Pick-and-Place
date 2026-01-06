"""
GAIL (Generative Adversarial Imitation Learning) - Simplified Version
Adversarial training with a discriminator to learn from expert demonstrations.

This is a simplified BC-GAIL variant that:
1. Uses a discriminator to score state-action pairs (expert vs policy)
2. Training alternates between generator (policy) and discriminator updates
3. Policy tries to produce actions that fool the discriminator
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
from typing import Generator, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_bc import BCPolicy
from sim.make_scene import PickPlaceEnv
from sim.expert import ScriptedExpert


class Discriminator(nn.Module):
    """Discriminator network that classifies state-action pairs as expert or policy."""
    def __init__(self, state_dim=7, action_dim=4, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


def train_gail_live(
    learning_rate: float = 0.0005,
    batch_size: int = 64,
    num_epochs: int = 100,
    hidden_size: int = 256,
    d_steps: int = 3  # Discriminator steps per generator step
) -> Generator[Dict[str, Any], None, None]:
    """
    Train policy using simplified GAIL with live updates.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load expert data
    expert_data = np.load("ml/data/demos.npz")
    expert_obs = expert_data['obs']
    expert_acts = expert_data['actions']
    
    with open("ml/data/norm_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Normalize expert data
    expert_obs_norm = (expert_obs - norm_stats['obs_mean']) / (norm_stats['obs_std'] + 1e-8)
    expert_acts_norm = (expert_acts - norm_stats['action_mean']) / (norm_stats['action_std'] + 1e-8)
    
    expert_obs_t = torch.FloatTensor(expert_obs_norm).to(device)
    expert_acts_t = torch.FloatTensor(expert_acts_norm).to(device)
    
    # Initialize models
    policy = BCPolicy(obs_dim=7, action_dim=4, hidden_size=hidden_size).to(device)
    discriminator = Discriminator(state_dim=7, action_dim=4, hidden_size=hidden_size).to(device)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    bce_loss = nn.BCELoss()
    n_expert = len(expert_obs_t)
    
    losses_d = []
    losses_g = []
    
    for epoch in range(num_epochs):
        # Shuffle expert data
        perm = torch.randperm(n_expert)
        expert_obs_shuffled = expert_obs_t[perm]
        expert_acts_shuffled = expert_acts_t[perm]
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0
        
        for i in range(0, min(n_expert, 1000), batch_size):  # Limit batches per epoch
            batch_expert_obs = expert_obs_shuffled[i:i+batch_size]
            batch_expert_acts = expert_acts_shuffled[i:i+batch_size]
            batch_size_actual = len(batch_expert_obs)
            
            # Generate policy actions for same states
            with torch.no_grad():
                policy.eval()
                policy_acts = policy(batch_expert_obs)
            
            # ===== Train Discriminator =====
            for _ in range(d_steps):
                discriminator.train()
                disc_optimizer.zero_grad()
                
                # Expert: label = 1 (real)
                expert_preds = discriminator(batch_expert_obs, batch_expert_acts)
                expert_labels = torch.ones(batch_size_actual, 1).to(device)
                
                # Policy: label = 0 (fake)
                policy_preds = discriminator(batch_expert_obs, policy_acts.detach())
                policy_labels = torch.zeros(batch_size_actual, 1).to(device)
                
                d_loss = bce_loss(expert_preds, expert_labels) + bce_loss(policy_preds, policy_labels)
                d_loss.backward()
                disc_optimizer.step()
            
            # ===== Train Policy (Generator) =====
            policy.train()
            policy_optimizer.zero_grad()
            
            # Generate new actions
            gen_acts = policy(batch_expert_obs)
            
            # Try to fool discriminator (want discriminator to output 1)
            disc_output = discriminator(batch_expert_obs, gen_acts)
            g_loss = bce_loss(disc_output, torch.ones(batch_size_actual, 1).to(device))
            g_loss.backward()
            policy_optimizer.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1
        
        avg_d_loss = epoch_d_loss / max(n_batches, 1)
        avg_g_loss = epoch_g_loss / max(n_batches, 1)
        losses_d.append(avg_d_loss)
        losses_g.append(avg_g_loss)
        
        yield {
            'epoch': epoch + 1,
            'total_epochs': num_epochs,
            'disc_loss': avg_d_loss,
            'gen_loss': avg_g_loss,
            'losses_d': losses_d.copy(),
            'losses_g': losses_g.copy(),
            'progress': (epoch + 1) / num_epochs * 100,
            'done': False
        }
    
    # Save model
    torch.save(policy.state_dict(), 'ml/gail_policy.pth')
    
    yield {
        'epoch': num_epochs,
        'total_epochs': num_epochs,
        'disc_loss': losses_d[-1],
        'gen_loss': losses_g[-1],
        'losses_d': losses_d,
        'losses_g': losses_g,
        'progress': 100,
        'done': True,
        'model_path': 'ml/gail_policy.pth'
    }


if __name__ == "__main__":
    print("Testing GAIL training...")
    for update in train_gail_live(num_epochs=10):
        print(f"Epoch {update['epoch']}/{update['total_epochs']} | "
              f"D_loss={update['disc_loss']:.4f} | G_loss={update['gen_loss']:.4f}")
    print("Done!")
