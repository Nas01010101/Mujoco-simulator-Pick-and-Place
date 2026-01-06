import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim.make_scene import PickPlaceEnv
from sim.expert import ScriptedExpert

def collect_data(num_episodes=100, save_path="demos.npz"):
    env = PickPlaceEnv()
    expert = ScriptedExpert()
    
    all_obs = []
    all_actions = []
    
    print(f"Collecting {num_episodes} episodes...")
    
    success_count = 0
    episode_idx = 0
    
    while success_count < num_episodes:
        obs = env.reset()
        expert.reset()
        
        episode_obs = []
        episode_actions = []
        
        steps = 0
        max_steps = 150  # Reduced - UR5e is faster
        
        while steps < max_steps:
            action, expert_done = expert.get_action(obs)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            obs = env.step(action)
            steps += 1

            if expert_done:
                break
                
        # Check success: cube near bin region
        cube_pos = obs[3:6]
        bin_pos = env.box_pos  # [0.35, 0.0, 0.35]
        
        # Cube should be close to bin XY position
        in_bin_xy = np.linalg.norm(cube_pos[:2] - bin_pos[:2]) < 0.1
        
        success = in_bin_xy
        
        episode_idx += 1
        if episode_idx % 10 == 0 or success:
            print(f"Episode {episode_idx}: Steps={steps}, Success={success}, CubePos=[{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        
        if success:
            success_count += 1
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            
    # Save
    np.savez(save_path, obs=np.array(all_obs), actions=np.array(all_actions))
    print(f"\nSaved {len(all_obs)} transitions from {success_count} successful episodes to {save_path}")

if __name__ == "__main__":
    os.makedirs("ml/data", exist_ok=True)
    collect_data(num_episodes=100, save_path="ml/data/demos.npz")
