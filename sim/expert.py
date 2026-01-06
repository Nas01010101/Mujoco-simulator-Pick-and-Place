import numpy as np

class ScriptedExpert:
    """
    Expert policy for UR5e pick-and-place task.
    Uses Cartesian control via velocity-based IK.
    """
    def __init__(self):
        # Container position - bin is at (0.2, -0.1, 0.35) with top at z~0.43
        self.box_xy = np.array([0.2, -0.1])
        self.reset()
        
    def reset(self):
        self.stage = 0
        self.counter = 0
        
    def get_action(self, obs):
        grip_pos = obs[0:3]
        cube_pos = obs[3:6]
        
        # Heights - tuned for IK workspace limits
        hover_z = 0.45      # Reachable hover height
        grasp_z = 0.32      # At cube level (cube is at ~0.31)
        drop_z = 0.40       # Just above bin (bin at z=0.35, walls to z~0.39)
        
        grip_cmd = -1.0
        done = False
        
        if self.stage == 0:  # Move above cube
            target_x = cube_pos[0]
            target_y = cube_pos[1]
            target_z = hover_z
            
            dist_xy = np.linalg.norm(grip_pos[:2] - cube_pos[:2])
            if dist_xy < 0.04 and abs(grip_pos[2] - hover_z) < 0.05:
                self.stage = 1
                self.counter = 0
                
        elif self.stage == 1:  # Lower to cube
            target_x = cube_pos[0]
            target_y = cube_pos[1]
            target_z = grasp_z
            
            dist = np.linalg.norm(grip_pos - cube_pos)
            if dist < 0.05:
                self.stage = 2
                self.counter = 0
                
        elif self.stage == 2:  # Grip
            target_x = cube_pos[0]
            target_y = cube_pos[1]
            target_z = grasp_z
            grip_cmd = 1.0
            
            self.counter += 1
            if self.counter > 15:
                self.stage = 3
                self.counter = 0
                
        elif self.stage == 3:  # Lift
            target_x = grip_pos[0]
            target_y = grip_pos[1]
            target_z = hover_z
            grip_cmd = 1.0
            
            if grip_pos[2] > 0.42:
                self.stage = 4
                self.counter = 0
                
        elif self.stage == 4:  # Move to bin
            target_x = self.box_xy[0]
            target_y = self.box_xy[1]
            target_z = hover_z
            grip_cmd = 1.0
            
            dist_xy = np.linalg.norm(grip_pos[:2] - self.box_xy)
            if dist_xy < 0.03:
                self.stage = 5
                self.counter = 0
                
        elif self.stage == 5:  # Lower toward bin
            target_x = self.box_xy[0]
            target_y = self.box_xy[1]
            target_z = drop_z
            grip_cmd = 1.0
            
            if grip_pos[2] < 0.42:
                self.stage = 6
                self.counter = 0
                
        elif self.stage == 6:  # Release
            target_x = self.box_xy[0]
            target_y = self.box_xy[1]
            target_z = drop_z
            grip_cmd = -1.0
            
            self.counter += 1
            if self.counter > 15:
                self.stage = 7
                
        else:  # Done - retreat up
            target_x = self.box_xy[0]
            target_y = self.box_xy[1]
            target_z = hover_z
            grip_cmd = -1.0
            done = True
            
        return np.array([target_x, target_y, target_z, grip_cmd], dtype=np.float32), done
