import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
import argparse
import mujoco
import imageio
from sim.make_scene import PickPlaceEnv
from ml.train_bc import BCPolicy

def render_frame(model, data, camera_name='main_cam', width=640, height=480):
    """Render a frame from MuJoCo simulation"""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    frame = renderer.render()
    return frame


def rollout_policy(model_path="ml/bc_policy.pth",
                   num_episodes=5,
                   max_steps=1200,
                   save_dir="assets/videos",
                   device='cpu',
                   width=640,
                   height=480):
    """
    Run policy rollouts and save videos.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        save_dir: Directory to save video files
        device: 'cpu', 'cuda', or 'mps'
        width: Video width in pixels
        height: Video height in pixels
    """
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = BCPolicy(
        obs_dim=checkpoint['obs_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_size=checkpoint['hidden_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load normalization stats
    stats_path = checkpoint['stats_path']
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
    
    # Run episodes with video recording
    print(f"\nRecording {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        frames = []
        steps = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}:")
        
        while steps < max_steps:
            # Render frame
            frame = render_frame(env.model, env.data, width=width, height=height)
            frames.append(frame)
            
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
        cube_pos = obs[3:6]
        bin_xy = np.array([0.2, -0.1])
        bin_floor_z = 0.35
        
        dist_xy = np.linalg.norm(cube_pos[:2] - bin_xy)
        inside_xy = dist_xy < 0.06
        above_floor = cube_pos[2] > bin_floor_z
        success = inside_xy and above_floor
        status = "success" if success else "fail"
        
        # Save video
        video_path = os.path.join(save_dir, f"episode_{episode+1:03d}_{status}.mp4")
        
        print(f"  Status: {status.upper()}")
        print(f"  Steps: {steps}")
        print(f"  Final cube position: {cube_pos}")
        print(f"  Saving video to {video_path}...")
        
        imageio.mimsave(video_path, frames, fps=30)
        print(f"  Video saved ({len(frames)} frames)")
    
    print(f"\n✓ All videos saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollout trained BC policy with video recording')
    parser.add_argument('--model-path', type=str, default='ml/bc_policy.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=5,
                        help='Number of episodes to record')
    parser.add_argument('--max-steps', type=int, default=1200,
                        help='Maximum steps per episode')
    parser.add_argument('--save-dir', type=str, default='assets/videos',
                        help='Directory to save videos')
    parser.add_argument('--width', type=int, default=640,
                        help='Video width in pixels')
    parser.add_argument('--height', type=int, default=480,
                        help='Video height in pixels')
    
    args = parser.parse_args()
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    rollout_policy(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        device=device,
        width=args.width,
        height=args.height
    )
