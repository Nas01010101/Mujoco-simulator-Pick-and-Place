"""
DAgger (Dataset Aggregation) Implementation
Interactive imitation learning that queries expert during policy rollouts.

Algorithm:
1. Train initial policy on expert demos (like BC)
2. Roll out policy, query expert for corrections at each state
3. Add corrected data to dataset
4. Retrain policy
5. Repeat for N iterations
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
from typing import Generator, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_bc import BCPolicy
from sim.make_scene import PickPlaceEnv
from sim.expert import ScriptedExpert


def train_dagger_live(
    learning_rate: float = 0.0005,
    batch_size: int = 64,
    num_iterations: int = 5,
    epochs_per_iter: int = 20,
    rollouts_per_iter: int = 10,
    hidden_size: int = 256
) -> Generator[Dict[str, Any], None, None]:
    """
    Train policy using DAgger with live updates.
    
    Yields dictionaries with training progress:
        - iteration: current DAgger iteration
        - epoch: current epoch within iteration
        - loss: current loss value
        - dataset_size: current dataset size
        - progress: percentage complete (0-100)
        - done: whether training is complete
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load initial expert demos
    initial_data = np.load("ml/data/demos.npz")
    obs_data = initial_data['obs'].tolist()
    acts_data = initial_data['actions'].tolist()
    
    # Load normalization stats
    with open("ml/data/norm_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Initialize model
    model = BCPolicy(obs_dim=7, action_dim=4, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Create environment and expert
    env = PickPlaceEnv()
    expert = ScriptedExpert()
    
    all_losses = []
    total_steps = num_iterations * epochs_per_iter
    current_step = 0
    
    for iteration in range(num_iterations):
        # ==== AGGREGATION PHASE ====
        # Roll out current policy and query expert for corrections
        for rollout in range(rollouts_per_iter):
            obs = env.reset()
            expert.reset()
            
            for step in range(150):  # Max steps per episode
                # Get policy action (for execution)
                obs_norm = (obs - norm_stats['obs_mean']) / (norm_stats['obs_std'] + 1e-8)
                obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    model.eval()
                    act_norm = model(obs_tensor).cpu().numpy()[0]
                
                # Query expert for the "correct" action at this state
                expert_action, done = expert.get_action(obs)
                
                # Add (state, expert_action) to dataset
                obs_data.append(obs.tolist())
                acts_data.append(expert_action.tolist())
                
                # Execute policy action (not expert action - this is key to DAgger)
                act = act_norm * (norm_stats['action_std'] + 1e-8) + norm_stats['action_mean']
                obs = env.step(act)
                
                if done:
                    break
        
        # ==== TRAINING PHASE ====
        # Retrain on aggregated dataset
        obs_array = np.array(obs_data)
        acts_array = np.array(acts_data)
        
        obs_norm = (obs_array - norm_stats['obs_mean']) / (norm_stats['obs_std'] + 1e-8)
        acts_norm = (acts_array - norm_stats['action_mean']) / (norm_stats['action_std'] + 1e-8)
        
        obs_tensor = torch.FloatTensor(obs_norm).to(device)
        acts_tensor = torch.FloatTensor(acts_norm).to(device)
        
        n_samples = len(obs_tensor)
        
        for epoch in range(epochs_per_iter):
            model.train()
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
            all_losses.append(avg_loss)
            current_step += 1
            
            yield {
                'iteration': iteration + 1,
                'total_iterations': num_iterations,
                'epoch': epoch + 1,
                'epochs_per_iter': epochs_per_iter,
                'loss': avg_loss,
                'losses': all_losses.copy(),
                'dataset_size': len(obs_data),
                'progress': current_step / total_steps * 100,
                'done': False
            }
    
    # Save model
    torch.save(model.state_dict(), 'ml/dagger_policy.pth')
    
    yield {
        'iteration': num_iterations,
        'total_iterations': num_iterations,
        'epoch': epochs_per_iter,
        'epochs_per_iter': epochs_per_iter,
        'loss': all_losses[-1],
        'losses': all_losses,
        'dataset_size': len(obs_data),
        'progress': 100,
        'done': True,
        'model_path': 'ml/dagger_policy.pth'
    }


if __name__ == "__main__":
    print("Testing DAgger training...")
    for update in train_dagger_live(num_iterations=2, epochs_per_iter=5, rollouts_per_iter=3):
        if update['epoch'] == 1 or update['done']:
            print(f"Iter {update['iteration']}/{update['total_iterations']} "
                  f"Epoch {update['epoch']}/{update['epochs_per_iter']} "
                  f"Loss={update['loss']:.4f} "
                  f"Dataset={update['dataset_size']}")
    print("Done!")
