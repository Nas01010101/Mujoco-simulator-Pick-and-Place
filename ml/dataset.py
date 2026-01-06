import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle

class DemoDataset(Dataset):
    """
    Dataset for behavior cloning from expert demonstrations.
    Loads (observation, action) pairs and normalizes observations.
    """
    def __init__(self, data_path, normalize=True, stats_path=None):
        """
        Args:
            data_path: Path to .npz file containing 'obs' and 'actions' arrays
            normalize: Whether to normalize observations
            stats_path: Path to save/load normalization statistics
        """
        # Load data
        data = np.load(data_path)
        self.observations = data['obs'].astype(np.float32)
        self.actions = data['actions'].astype(np.float32)
        
        assert len(self.observations) == len(self.actions), \
            f"Obs and actions length mismatch: {len(self.observations)} vs {len(self.actions)}"
        
        print(f"Loaded {len(self.observations)} transitions from {data_path}")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        
        # Compute or load normalization statistics
        self.normalize = normalize
        self.obs_mean = None
        self.obs_std = None
        
        if normalize:
            if stats_path and os.path.exists(stats_path):
                # Load existing stats
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                    self.obs_mean = stats['obs_mean']
                    self.obs_std = stats['obs_std']
                    self.action_mean = stats['action_mean']
                    self.action_std = stats['action_std']
                print(f"Loaded normalization stats from {stats_path}")
            else:
                # Compute new stats
                self.obs_mean = np.mean(self.observations, axis=0)
                self.obs_std = np.std(self.observations, axis=0) + 1e-8  # Avoid division by zero
                
                self.action_mean = np.mean(self.actions, axis=0)
                self.action_std = np.std(self.actions, axis=0) + 1e-8
                
                # Save stats if path provided
                if stats_path:
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    with open(stats_path, 'wb') as f:
                        pickle.dump({
                            'obs_mean': self.obs_mean,
                            'obs_std': self.obs_std,
                            'action_mean': self.action_mean,
                            'action_std': self.action_std
                        }, f)
                    print(f"Saved normalization stats to {stats_path}")
            
            # Normalize observations
            self.observations = (self.observations - self.obs_mean) / self.obs_std
            # Normalize actions
            self.actions = (self.actions - self.action_mean) / self.action_std
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.observations[idx]),
            torch.from_numpy(self.actions[idx])
        )
    
    def get_normalization_stats(self):
        """Returns normalization statistics for use during inference"""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std
        }


if __name__ == "__main__":
    # Test dataset loading
    dataset = DemoDataset(
        data_path="ml/data/demos.npz",
        normalize=True,
        stats_path="ml/data/norm_stats.pkl"
    )
    
    print(f"\nDataset size: {len(dataset)}")
    obs, action = dataset[0]
    print(f"Sample observation: {obs}")
    print(f"Sample action: {action}")
