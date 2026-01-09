import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
from sim.make_scene import PickPlaceEnv
from ml.train_bc import BCPolicy

def evaluate_policy(model_path="ml/bc_policy.pth",
                    num_episodes=50,
                    max_steps=300,
                    device='cpu',
                    verbose=True):
    """
    Evaluate a trained BC policy by running rollouts.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        device: 'cpu', 'cuda', or 'mps'
        verbose: Whether to print per-episode results
    
    Returns:
        dict: Evaluation statistics
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint dict and raw state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        model = BCPolicy(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_size=checkpoint['hidden_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        stats_path = checkpoint.get('stats_path', 'ml/data/norm_stats.pkl')
    else:
        # Legacy format: raw state_dict - infer hidden size from weights
        state_dict = checkpoint
        hidden_size = state_dict['network.0.weight'].shape[0]
        model = BCPolicy(obs_dim=7, action_dim=4, hidden_size=hidden_size)
        model.load_state_dict(state_dict)
        stats_path = 'ml/data/norm_stats.pkl'
    
    model = model.to(device)
    model.eval()
    
    # Load normalization stats
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            obs_mean = stats['obs_mean']
            obs_std = stats['obs_std']
            action_mean = stats['action_mean'] if 'action_mean' in stats else None
            action_std = stats['action_std'] if 'action_std' in stats else None
        print(f"Loaded normalization stats from {stats_path}")
    else:
        print("Warning: No normalization stats found, using raw observations")
        obs_mean = None
        obs_std = None
        action_mean = None
        action_std = None
    
    # Initialize environment
    env = PickPlaceEnv()
    
    # Run evaluation
    print(f"\nEvaluating policy over {num_episodes} episodes...")
    
    successes = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Normalize observation
            if obs_mean is not None:
                obs_normalized = (obs - obs_mean) / obs_std
            else:
                obs_normalized = obs
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_normalized).float().to(device)
                action_tensor = model(obs_tensor)
                action = action_tensor.cpu().numpy()
            
            # Denormalize action
            if action_mean is not None:
                action = action * action_std + action_mean
            
            # Step environment
            obs = env.step(action)
            steps += 1
        
        # Check success: cube must be INSIDE the bin
        # Bin is at (0.2, -0.1, 0.35), floor at z~0.35, walls to z~0.39
        # Cube is "inside" if: XY within bin bounds AND z > bin floor
        cube_pos = obs[3:6]
        bin_xy = np.array([0.2, -0.1])  # env.box_pos[:2]
        bin_floor_z = 0.35
        
        dist_xy = np.linalg.norm(cube_pos[:2] - bin_xy)
        inside_xy = dist_xy < 0.06  # Bin inner radius ~0.05
        above_floor = cube_pos[2] > bin_floor_z
        
        success = inside_xy and above_floor
        
        successes.append(success)
        episode_lengths.append(steps)
        
        if verbose:
            status = "SUCCESS" if success else "FAIL"
            reason = ""
            if not success:
                if not inside_xy:
                    reason = f" (cube xy dist={dist_xy:.3f}, need <0.06)"
                elif not above_floor:
                    reason = f" (cube z={cube_pos[2]:.3f}, need >0.35)"
            print(f"Episode {episode+1}/{num_episodes}: {status}{reason}")
    
    # Compute statistics
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Success Rate:     {success_rate*100:.1f}% ({sum(successes)}/{num_episodes})")
    print(f"Avg Episode Len:  {avg_length:.1f} ± {std_length:.1f} steps")
    print(f"Max Episode Len:  {max(episode_lengths)} steps")
    print(f"Min Episode Len:  {min(episode_lengths)} steps")
    print("="*60)
    
    return {
        'success_rate': success_rate,
        'successes': successes,
        'episode_lengths': episode_lengths,
        'avg_length': avg_length,
        'std_length': std_length
    }


if __name__ == "__main__":
    # Check device availability
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    results = evaluate_policy(
        model_path="ml/bc_policy.pth",
        num_episodes=50,
        max_steps=1200,
        device=device,
        verbose=True
    )
