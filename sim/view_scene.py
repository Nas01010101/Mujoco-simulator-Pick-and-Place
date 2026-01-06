"""
Interactive viewer for the UR5e pick-and-place environment.
Opens MuJoCo's native viewer for real-time visualization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
from sim.make_scene import PickPlaceEnv

def run_viewer(use_policy=False, model_path="ml/bc_policy.pth"):
    """
    Open interactive MuJoCo viewer.
    
    Args:
        use_policy: If True, run the trained BC policy
        model_path: Path to trained model (if use_policy=True)
    """
    env = PickPlaceEnv()
    obs = env.reset()
    
    print("=" * 50)
    print("UR5e Pick-and-Place Viewer")
    print("=" * 50)
    print("\nControls:")
    print("  - Mouse drag: Rotate camera")
    print("  - Scroll: Zoom")
    print("  - Space: Pause/unpause")
    print("  - ESC or close window: Exit")
    print("=" * 50)
    
    # Load policy if requested
    policy = None
    obs_mean = None
    obs_std = None
    action_mean = None
    action_std = None
    
    if use_policy:
        import torch
        import pickle
        from ml.train_bc import BCPolicy
        
        checkpoint = torch.load(model_path, map_location='cpu')
        policy = BCPolicy(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_size=checkpoint['hidden_size']
        )
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        
        stats_path = checkpoint['stats_path']
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                obs_mean = stats['obs_mean']
                obs_std = stats['obs_std']
                action_mean = stats.get('action_mean')
                action_std = stats.get('action_std')
        
        print(f"\nLoaded policy from {model_path}")
        print("Running trained BC policy...")
    else:
        print("\nNo policy loaded - showing static scene")
        print("Run with --policy flag to see the trained agent")
    
    step_count = [0]  # Use list to allow modification in nested function
    
    def controller(model, data):
        if policy is not None:
            import torch
            
            # Get observation
            grip_pos = env.data.site_xpos[env.grip_site_id].copy()
            cube_pos = env.data.site_xpos[env.cube_site_id].copy()
            grip_state = 1.0 if env.gripping else 0.0
            current_obs = np.concatenate([grip_pos, cube_pos, [grip_state]]).astype(np.float32)
            
            # Normalize observation
            if obs_mean is not None:
                obs_norm = (current_obs - obs_mean) / obs_std
            else:
                obs_norm = current_obs
            
            # Get action
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_norm).float()
                action = policy(obs_tensor).numpy()
            
            # Denormalize action
            if action_mean is not None:
                action = action * action_std + action_mean
            
            # Apply action
            env.step(action)
            step_count[0] += 1
            
            # Reset every 200 steps
            if step_count[0] >= 200:
                env.reset()
                step_count[0] = 0
    
    # Launch viewer (blocking mode works on macOS)
    mujoco.viewer.launch(env.model, env.data, show_left_ui=True, show_right_ui=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive UR5e viewer")
    parser.add_argument('--policy', action='store_true', help='Run trained BC policy')
    parser.add_argument('--model-path', type=str, default='ml/bc_policy.pth', help='Path to model')
    args = parser.parse_args()
    
    run_viewer(use_policy=args.policy, model_path=args.model_path)
