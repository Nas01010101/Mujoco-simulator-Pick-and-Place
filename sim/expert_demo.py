"""
Generate demo videos using the EXPERT policy (not learned policy).
This creates clean, successful demonstrations for the project showcase.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import imageio
from sim.make_scene import PickPlaceEnv
from sim.expert import ScriptedExpert


def render_frame(model, data, camera_name='main_cam', width=640, height=480):
    """Render a frame from MuJoCo simulation"""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    frame = renderer.render()
    return frame


def generate_expert_demos(num_episodes=3, max_steps=400, save_dir="assets/videos",
                          width=640, height=480, camera='main_cam'):
    """
    Generate demo videos using expert policy.
    
    Args:
        num_episodes: Number of demo episodes to record
        max_steps: Maximum steps per episode  
        save_dir: Directory to save videos
        width: Video width
        height: Video height
        camera: Camera to use ('main_cam', 'front_cam', or 'overview_cam')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    env = PickPlaceEnv()
    expert = ScriptedExpert()
    
    print(f"Generating {num_episodes} expert demo videos...")
    print(f"Using camera: {camera}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        expert.reset()
        frames = []
        done = False
        steps = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}:")
        
        while not done and steps < max_steps:
            # Render frame
            frame = render_frame(env.model, env.data, camera_name=camera, 
                               width=width, height=height)
            frames.append(frame)
            
            # Get expert action
            action, done = expert.get_action(obs)
            
            # Step environment
            obs = env.step(action)
            steps += 1
        
        # Check final state
        cube_pos = obs[3:6]
        bin_xy = np.array([0.2, -0.1])
        dist_xy = np.linalg.norm(cube_pos[:2] - bin_xy)
        
        # Save video (neutral naming, no success/fail)
        video_path = os.path.join(save_dir, f"demo_{episode+1:02d}.mp4")
        
        print(f"  Steps: {steps}")
        print(f"  Final cube position: {cube_pos}")
        print(f"  Distance to bin: {dist_xy:.3f}")
        print(f"  Saving to {video_path}...")
        
        imageio.mimsave(video_path, frames, fps=30)
        print(f"  Saved ({len(frames)} frames)")
    
    print(f"\n✓ All demos saved to {save_dir}/")


if __name__ == "__main__":
    generate_expert_demos(
        num_episodes=3,
        max_steps=400,
        camera='main_cam'  # Options: main_cam, front_cam, overview_cam
    )
