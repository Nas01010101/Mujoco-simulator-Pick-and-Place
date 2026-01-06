"""
UR5e Pick-and-Place Environment using MuJoCo Menagerie.
Uses velocity-based IK control for end-effector positioning.
"""
import mujoco
import numpy as np
from robot_descriptions import ur5e_mj_description
import os

def build_scene_xml():
    """Create MJCF XML with UR5e robot positioned on a pedestal to reach the table."""
    ur5e_path = ur5e_mj_description.MJCF_PATH
    ur5e_dir = os.path.dirname(ur5e_path)
    
    return f'''
<mujoco model="ur5e_pickplace">
    <compiler angle="radian" meshdir="{ur5e_dir}/assets" autolimits="true"/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
    
    <default>
        <joint damping="20" armature="0.1"/>
        <geom contype="1" conaffinity="1" friction="1 0.5 0.01"/>
        <default class="ur5e">
            <material specular="0.5" shininess="0.25"/>
            <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
            <default class="size3">
                <default class="size3_limited">
                    <joint range="-3.1415 3.1415"/>
                </default>
            </default>
            <default class="size1"/>
            <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2"/>
            </default>
            <default class="collision">
                <geom type="capsule" group="3" contype="0" conaffinity="0"/>
            </default>
        </default>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.65" rgb2="0.1 0.1 0.15" width="512" height="512"/>
        <texture type="2d" name="grid" builtin="checker" rgb1="0.3 0.3 0.35" rgb2="0.4 0.4 0.45" width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.15"/>
        <material name="table_mat" rgba="0.45 0.35 0.25 1" specular="0.3" shininess="0.3"/>
        <material name="pedestal_mat" rgba="0.3 0.3 0.35 1" specular="0.5" shininess="0.5"/>
        <material name="cube_mat" rgba="0.9 0.3 0.2 1" specular="0.6" shininess="0.7"/>
        <material name="bin_mat" rgba="0.25 0.6 0.35 1" specular="0.4" shininess="0.5"/>
        <material name="gripper_mat" rgba="0.2 0.2 0.25 1" specular="0.8" shininess="0.8"/>
        
        <!-- UR5e meshes -->
        <material class="ur5e" name="black" rgba="0.033 0.033 0.033 1"/>
        <material class="ur5e" name="jointgray" rgba="0.278 0.278 0.278 1"/>
        <material class="ur5e" name="linkgray" rgba="0.82 0.82 0.82 1"/>
        <material class="ur5e" name="urblue" rgba="0.49 0.678 0.8 1"/>
        
        <mesh file="base_0.obj"/>
        <mesh file="base_1.obj"/>
        <mesh file="shoulder_0.obj"/>
        <mesh file="shoulder_1.obj"/>
        <mesh file="shoulder_2.obj"/>
        <mesh file="upperarm_0.obj"/>
        <mesh file="upperarm_1.obj"/>
        <mesh file="upperarm_2.obj"/>
        <mesh file="upperarm_3.obj"/>
        <mesh file="forearm_0.obj"/>
        <mesh file="forearm_1.obj"/>
        <mesh file="forearm_2.obj"/>
        <mesh file="forearm_3.obj"/>
        <mesh file="wrist1_0.obj"/>
        <mesh file="wrist1_1.obj"/>
        <mesh file="wrist1_2.obj"/>
        <mesh file="wrist2_0.obj"/>
        <mesh file="wrist2_1.obj"/>
        <mesh file="wrist2_2.obj"/>
        <mesh file="wrist3.obj"/>
    </asset>

    <worldbody>
        <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9" specular="0.5 0.5 0.5" castshadow="true"/>
        <light pos="-0.3 0.3 0.8" dir="0.2 -0.2 -1" diffuse="0.5 0.5 0.5"/>
        
        <!-- Ground -->
        <geom name="floor" type="plane" size="1 1 0.1" material="grid_mat"/>
        
        <!-- Robot Pedestal -->
        <body name="pedestal" pos="0 -0.3 0.2">
            <geom type="box" size="0.1 0.1 0.2" material="pedestal_mat"/>
        </body>
        
        <!-- Table (in front of robot) -->
        <body name="table" pos="0.35 0.15 0.25">
            <geom type="box" size="0.25 0.25 0.02" material="table_mat"/>
            <geom type="box" size="0.02 0.02 0.25" pos="0.22 0.22 -0.25" material="table_mat"/>
            <geom type="box" size="0.02 0.02 0.25" pos="-0.22 0.22 -0.25" material="table_mat"/>
            <geom type="box" size="0.02 0.02 0.25" pos="0.22 -0.22 -0.25" material="table_mat"/>
            <geom type="box" size="0.02 0.02 0.25" pos="-0.22 -0.22 -0.25" material="table_mat"/>
        </body>
        
        <!-- UR5e Robot on pedestal -->
        <body name="base" pos="0 -0.3 0.4" quat="0.7071 0 0 0.7071" childclass="ur5e">
            <inertial mass="4.0" pos="0 0 0" diaginertia="0.00443333156 0.00443333156 0.0072"/>
            <geom mesh="base_0" material="black" class="visual"/>
            <geom mesh="base_1" material="jointgray" class="visual"/>
            <body name="shoulder_link" pos="0 0 0.163">
                <inertial mass="3.7" pos="0 0 0" diaginertia="0.0102675 0.0102675 0.00666"/>
                <joint name="shoulder_pan_joint" class="size3" axis="0 0 1"/>
                <geom mesh="shoulder_0" material="urblue" class="visual"/>
                <geom mesh="shoulder_1" material="black" class="visual"/>
                <geom mesh="shoulder_2" material="jointgray" class="visual"/>
                <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
                    <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>
                    <joint name="shoulder_lift_joint" class="size3"/>
                    <geom mesh="upperarm_0" material="linkgray" class="visual"/>
                    <geom mesh="upperarm_1" material="black" class="visual"/>
                    <geom mesh="upperarm_2" material="jointgray" class="visual"/>
                    <geom mesh="upperarm_3" material="urblue" class="visual"/>
                    <body name="forearm_link" pos="0 -0.131 0.425">
                        <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0311796 0.0311796 0.004095"/>
                        <joint name="elbow_joint" class="size3_limited"/>
                        <geom mesh="forearm_0" material="urblue" class="visual"/>
                        <geom mesh="forearm_1" material="linkgray" class="visual"/>
                        <geom mesh="forearm_2" material="black" class="visual"/>
                        <geom mesh="forearm_3" material="jointgray" class="visual"/>
                        <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
                            <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
                            <joint name="wrist_1_joint" class="size1"/>
                            <geom mesh="wrist1_0" material="black" class="visual"/>
                            <geom mesh="wrist1_1" material="urblue" class="visual"/>
                            <geom mesh="wrist1_2" material="jointgray" class="visual"/>
                            <body name="wrist_2_link" pos="0 0.127 0">
                                <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
                                <joint name="wrist_2_joint" axis="0 0 1" class="size1"/>
                                <geom mesh="wrist2_0" material="black" class="visual"/>
                                <geom mesh="wrist2_1" material="urblue" class="visual"/>
                                <geom mesh="wrist2_2" material="jointgray" class="visual"/>
                                <body name="wrist_3_link" pos="0 0 0.1">
                                    <inertial mass="0.1889" pos="0 0.0771683 0" quat="1 0 0 1" diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                                    <joint name="wrist_3_joint" class="size1"/>
                                    <geom material="linkgray" mesh="wrist3" class="visual"/>
                                    
                                    <!-- Simple magnetic gripper -->
                                    <body name="gripper" pos="0 0.1 0" quat="-1 1 0 0">
                                        <geom type="cylinder" size="0.02 0.025" material="gripper_mat"/>
                                        <geom type="sphere" pos="0 0 0.04" size="0.015" material="gripper_mat"/>
                                        <site name="grip_site" pos="0 0 0.05" size="0.008"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Cube to pick up -->
        <body name="cube" pos="0.35 0.2 0.3">
            <joint name="cube_joint" type="free" damping="0"/>
            <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.03" material="cube_mat"/>
            <site name="cube_site" pos="0 0 0" size="0.004"/>
        </body>
        
        <!-- Target Bin - positioned closer to robot on a raised platform -->
        <body name="bin" pos="0.2 -0.1 0.35">
            <!-- Platform base -->
            <geom type="cylinder" size="0.08 0.04" pos="0 0 -0.04" material="table_mat"/>
            <!-- Bin bottom -->
            <geom type="box" size="0.05 0.05 0.004" pos="0 0 0.004" material="bin_mat"/>
            <!-- Bin walls (taller) -->
            <geom type="box" size="0.05 0.004 0.04" pos="0 0.046 0.04" material="bin_mat"/>
            <geom type="box" size="0.05 0.004 0.04" pos="0 -0.046 0.04" material="bin_mat"/>
            <geom type="box" size="0.004 0.05 0.04" pos="0.046 0 0.04" material="bin_mat"/>
            <geom type="box" size="0.004 0.05 0.04" pos="-0.046 0 0.04" material="bin_mat"/>
            <site name="bin_site" pos="0 0 0.05" size="0.008"/>
        </body>
        
        <camera name="main_cam" pos="0.8 -0.5 0.8" xyaxes="0.6 0.8 0 -0.35 0.25 0.9"/>
        <camera name="front_cam" pos="0.35 -0.9 0.5" xyaxes="1 0 0 0 0.5 0.87"/>
    </worldbody>
    
    <!-- Magnetic grip for cube -->
    <equality>
        <weld name="grip_weld" body1="gripper" body2="cube" relpose="0 0 0.07 1 0 0 0" active="false"/>
    </equality>

    <!-- UR5e actuators - using velocity control for smoother motion -->
    <actuator>
        <velocity name="shoulder_pan_v" joint="shoulder_pan_joint" kv="100"/>
        <velocity name="shoulder_lift_v" joint="shoulder_lift_joint" kv="100"/>
        <velocity name="elbow_v" joint="elbow_joint" kv="100"/>
        <velocity name="wrist_1_v" joint="wrist_1_joint" kv="50"/>
        <velocity name="wrist_2_v" joint="wrist_2_joint" kv="50"/>
        <velocity name="wrist_3_v" joint="wrist_3_joint" kv="50"/>
    </actuator>
</mujoco>
'''


class PickPlaceEnv:
    """UR5e pick-and-place environment with Jacobian-based velocity control."""
    
    def __init__(self):
        xml = build_scene_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        
        # Cache IDs
        self.grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self.cube_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.bin_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_site")
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        
        self.grip_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld")
        
        # Joint IDs
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        self.gripping = False
        
        # Bin position (for expert policy) 
        # Bin position (for expert policy) - inside the raised bin
        self.box_pos = np.array([0.2, -0.1, 0.40])
        
        # Table height
        self.table_z = 0.27
        
    def _compute_jacobian(self):
        """Compute end-effector Jacobian for the 6 arm joints."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.grip_site_id)
        return jacp[:, :6]
    
    def _velocity_control(self, target_pos, gain=5.0):
        """Compute joint velocities to move gripper toward target."""
        mujoco.mj_forward(self.model, self.data)
        
        current_pos = self.data.site_xpos[self.grip_site_id].copy()
        error = target_pos - current_pos
        
        # Clamp max velocity
        max_vel = 0.3
        error = np.clip(error, -max_vel, max_vel)
        
        # Compute Jacobian
        jac = self._compute_jacobian()
        
        # Damped least squares IK
        damping = 0.05
        jac_T = jac.T
        joint_vel = gain * jac_T @ np.linalg.solve(jac @ jac_T + damping**2 * np.eye(3), error)
        
        # Set velocity actuator controls
        self.data.ctrl[:6] = joint_vel
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # Good initial arm configuration - arm curved over the table
        init_qpos = [0.0, -1.2, 1.8, -2.2, -1.57, 0.0]
        for i, jid in enumerate(self.joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr] = init_qpos[i]
        
        # Randomize cube position on table
        cube_x = 0.35 + np.random.uniform(-0.08, 0.08)
        cube_y = 0.2 + np.random.uniform(-0.08, 0.08)
        cube_z = self.table_z + 0.02 + 0.02  # Table height + half cube
        
        cube_qpos_adr = self.model.jnt_qposadr[self.cube_joint_id]
        self.data.qpos[cube_qpos_adr:cube_qpos_adr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_adr+3:cube_qpos_adr+7] = [1, 0, 0, 0]
        
        # Disable grip
        self.data.eq_active[self.grip_weld_id] = 0
        self.gripping = False
        
        mujoco.mj_forward(self.model, self.data)
        
        # Move gripper above cube
        target = np.array([cube_x, cube_y, cube_z + 0.12])
        for _ in range(200):
            self._velocity_control(target, gain=8.0)
            mujoco.mj_step(self.model, self.data)
        
        # Zero velocities
        self.data.ctrl[:6] = 0
        self.data.qvel[:6] = 0
        
        # Reset cube position (in case it moved)
        self.data.qpos[cube_qpos_adr:cube_qpos_adr+3] = [cube_x, cube_y, cube_z]
        self.data.qvel[cube_qpos_adr:cube_qpos_adr+6] = 0
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()
    
    def get_obs(self):
        """Get 7D observation: gripper pos (3), cube pos (3), grip state (1)."""
        grip_pos = self.data.site_xpos[self.grip_site_id].copy()
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        grip_state = 1.0 if self.gripping else 0.0
        
        return np.concatenate([grip_pos, cube_pos, [grip_state]]).astype(np.float32)
    
    def set_grip(self, grip_on):
        """Toggle magnetic grip based on proximity."""
        if grip_on and not self.gripping:
            grip_pos = self.data.site_xpos[self.grip_site_id]
            cube_pos = self.data.site_xpos[self.cube_site_id]
            dist = np.linalg.norm(grip_pos - cube_pos)
            if dist < 0.06:
                self.data.eq_active[self.grip_weld_id] = 1
                self.gripping = True
        elif not grip_on and self.gripping:
            self.data.eq_active[self.grip_weld_id] = 0
            self.gripping = False
    
    def step(self, action, n_substeps=8):
        """Execute action: [target_x, target_y, target_z, grip_cmd]"""
        target_x, target_y, target_z, grip_cmd = action
        
        # Clip to workspace
        target_pos = np.array([
            np.clip(target_x, 0.1, 0.6),
            np.clip(target_y, -0.2, 0.4),
            np.clip(target_z, 0.28, 0.6)
        ])
        
        self.set_grip(grip_cmd > 0)
        
        for _ in range(n_substeps):
            self._velocity_control(target_pos, gain=6.0)
            mujoco.mj_step(self.model, self.data)
        
        return self.get_obs()


if __name__ == "__main__":
    env = PickPlaceEnv()
    obs = env.reset()
    print("UR5e Pick-Place Environment Initialized!")
    print(f"Observation shape: {obs.shape}")
    print(f"Grip position: {obs[:3]}")
    print(f"Cube position: {obs[3:6]}")
    print(f"Grip state: {obs[6]}")
    print(f"Bin position: {env.box_pos}")
    
    # Test movement toward cube
    print("\nTesting movement to cube...")
    cube_pos = obs[3:6].copy()
    target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.05]
    for i in range(20):
        obs = env.step([target[0], target[1], target[2], -1])
    print(f"After 20 steps: Grip position: {obs[:3]}")
    print(f"Distance to target: {np.linalg.norm(obs[:3] - target):.4f}")
