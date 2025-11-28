"""
Gymnasium environment wrapper for Franka pick-and-place task in Isaac Sim.

This module provides a Gymnasium-compatible environment for training RL agents
on a pick-and-place task using the Franka Panda robot in NVIDIA Isaac Sim.

Usage:
    from franka_rl_env import FrankaPickPlaceEnv
    env = FrankaPickPlaceEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

Compatible with Stable-Baselines3 and other Gymnasium-compatible RL libraries.
"""

from isaacsim import SimulationApp

# Module-level globals (set after SimulationApp is created)
# This follows the same pattern as franka_keyboard_control.py
simulation_app = None
World = None
ArticulationAction = None
Franka = None
KinematicsSolver = None
euler_angles_to_quat = None
quat_to_euler_angles = None

import numpy as np

# Import gymnasium
import gymnasium
from gymnasium import spaces

# Import reusable components from franka_keyboard_control
from franka_keyboard_control import (
    ObservationBuilder,
    RewardComputer,
    SceneManager,
)


class FrankaPickPlaceEnv(gymnasium.Env):
    """Gymnasium environment for Franka Panda pick-and-place task in Isaac Sim.

    This environment wraps the Isaac Sim simulation of a Franka Panda robot
    performing a pick-and-place task. The robot must grasp a cube and place
    it at a goal location.

    Observation Space (23D Box):
        [0:7]   - Joint positions (7 arm joints, radians)
        [7:10]  - End-effector position (x, y, z in meters)
        [10:14] - End-effector orientation (quaternion w, x, y, z)
        [14:15] - Gripper width (meters)
        [15:18] - Cube position (x, y, z in meters)
        [18:21] - Goal position (x, y, z in meters)
        [21:22] - Cube grasped flag (0.0 or 1.0)
        [22:23] - Distance to cube (meters)

    Action Space (7D Box, continuous [-1, 1]):
        [0:3]   - End-effector velocity (dx, dy, dz)
        [3:6]   - End-effector angular velocity (droll, dpitch, dyaw)
        [6]     - Gripper command (-1 = close, +1 = open)

    Attributes:
        reward_mode: 'dense' for shaped rewards, 'sparse' for task completion only
        max_episode_steps: Maximum steps before episode truncation
    """

    metadata = {'render_modes': ['human']}

    # Action scaling factors (from continuous [-1, 1] to physical units)
    DEFAULT_ACTION_SCALE = {
        'position': 0.02,   # meters per action unit
        'rotation': 0.1,    # radians per action unit
        'gripper': 0.01,    # meters per action unit
    }

    # Home position (neutral pose) - 9 DOF: 7 arm joints + 2 gripper fingers
    HOME_JOINT_POSITIONS = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04])

    # Workspace bounds for out-of-bounds checking
    DEFAULT_WORKSPACE_BOUNDS = {
        'x': [0.3, 0.6],
        'y': [-0.3, 0.3],
        'z': [0.02, 0.20]
    }

    def __init__(
        self,
        reward_mode: str = 'dense',
        max_episode_steps: int = 500,
        action_scale: dict = None,
        workspace_bounds: dict = None,
        render_mode: str = 'human',
        headless: bool = False,
    ):
        """Initialize the Franka pick-and-place environment.

        Args:
            reward_mode: 'dense' for shaped rewards, 'sparse' for task completion only
            max_episode_steps: Maximum steps before episode truncation
            action_scale: Dict with 'position', 'rotation', 'gripper' scaling factors
            workspace_bounds: Dict with 'x', 'y', 'z' bounds for workspace
            render_mode: Render mode ('human' for always rendering)
            headless: If True, run simulation without GUI (faster for training)
        """
        super().__init__()

        # Store configuration
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale or self.DEFAULT_ACTION_SCALE.copy()
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS.copy()
        self.render_mode = render_mode
        self.headless = headless

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Initialize Isaac Sim
        self._init_isaac_sim()

        # Initialize helper components
        self.obs_builder = ObservationBuilder()
        self.reward_computer = RewardComputer(mode=reward_mode)
        self.scene_manager = SceneManager(self.world, workspace_bounds=self.workspace_bounds)

        # Spawn initial scene objects
        self.scene_manager.spawn_cube()
        self.scene_manager.spawn_goal_marker()

        # Initialize tracking state
        self._step_count = 0
        self._cube_grasped = False
        self._prev_grasped = False
        self._prev_obs = None

        # End-effector target (position + orientation as euler angles)
        self._ee_target_position = np.array([0.3, 0.0, 0.4])
        self._ee_target_euler = np.array([np.pi, 0.0, 0.0])  # [roll, pitch, yaw]
        self._gripper_target = 0.04  # gripper width in meters

    def _init_isaac_sim(self):
        """Initialize Isaac Sim simulation environment."""
        global simulation_app, World, ArticulationAction, Franka, KinematicsSolver
        global euler_angles_to_quat, quat_to_euler_angles

        # Create SimulationApp if not already created
        if simulation_app is None:
            simulation_app = SimulationApp({"headless": self.headless})
        self.simulation_app = simulation_app

        # Import Isaac Sim modules after SimulationApp is created
        if World is None:
            from isaacsim.core.api import World as _World
            from isaacsim.core.utils.types import ArticulationAction as _ArticulationAction
            from isaacsim.robot.manipulators.examples.franka import Franka as _Franka
            from isaacsim.robot.manipulators.examples.franka import KinematicsSolver as _KinematicsSolver
            from isaacsim.core.utils.rotations import euler_angles_to_quat as _euler_angles_to_quat
            from isaacsim.core.utils.rotations import quat_to_euler_angles as _quat_to_euler_angles

            World = _World
            ArticulationAction = _ArticulationAction
            Franka = _Franka
            KinematicsSolver = _KinematicsSolver
            euler_angles_to_quat = _euler_angles_to_quat
            quat_to_euler_angles = _quat_to_euler_angles

        # Create world and robot
        self.world = World(stage_units_in_meters=1.0)
        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="franka_rl")
        )
        self.world.scene.add_default_ground_plane()

        # Reset world to initialize physics
        self.world.reset()

        # Initialize IK solver
        self.ik_solver = KinematicsSolver(self.franka)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation (23D numpy array)
            info: Information dictionary
        """
        super().reset(seed=seed)

        # Reset robot to home position
        self.franka.set_joint_positions(self.HOME_JOINT_POSITIONS)

        # Reset scene with new random cube and goal positions
        self.scene_manager.reset_scene()

        # Step simulation to settle physics
        for _ in range(10):
            self.world.step(render=True)

        # Reset tracking state
        self._step_count = 0
        self._cube_grasped = False
        self._prev_grasped = False

        # Reset end-effector targets from current pose
        ee_pos, ee_quat = self.franka.end_effector.get_world_pose()
        self._ee_target_position = np.array(ee_pos)
        self._ee_target_euler = quat_to_euler_angles(ee_quat)
        self._gripper_target = 0.04

        # Build initial observation
        observation = self._build_observation()
        self._prev_obs = observation.copy()

        # Build info dictionary
        cube_pos, _ = self.scene_manager.get_cube_pose()
        goal_pos = self.scene_manager.get_goal_position()

        info = {
            'cube_position': np.array(cube_pos) if cube_pos is not None else np.zeros(3),
            'goal_position': np.array(goal_pos) if goal_pos is not None else np.zeros(3),
            'is_success': False,
        }

        return observation, info

    def step(self, action):
        """Execute one environment step.

        Args:
            action: 7D numpy array with values in [-1, 1]

        Returns:
            observation: Next observation (23D numpy array)
            reward: Scalar reward
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was cut short (time limit)
            info: Information dictionary
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Scale action from [-1, 1] to physical units
        scaled_action = self._scale_action(action)

        # Update end-effector target
        self._ee_target_position += scaled_action[0:3]
        self._ee_target_euler += scaled_action[3:6]
        self._gripper_target += scaled_action[6]
        self._gripper_target = np.clip(self._gripper_target, 0.0, 0.08)  # Franka gripper limits

        # Compute IK and apply to robot
        ik_success = self._apply_ik_control()

        # Apply gripper control
        self._apply_gripper_control()

        # Step simulation
        self.world.step(render=True)

        # Increment step counter
        self._step_count += 1

        # Check grasp state
        ee_pos, _ = self.franka.end_effector.get_world_pose()
        gripper_width = self.franka.get_joint_positions()[7] * 2  # Both fingers
        self._prev_grasped = self._cube_grasped
        self._cube_grasped = self.scene_manager.check_grasp(ee_pos, gripper_width)

        # Build next observation
        next_obs = self._build_observation()

        # Build info dict for reward computation
        task_complete = self.scene_manager.check_task_complete()
        info = {
            'cube_grasped': self._cube_grasped,
            'just_grasped': self._cube_grasped and not self._prev_grasped,
            'cube_dropped': not self._cube_grasped and self._prev_grasped,
            'task_complete': task_complete,
            'ik_success': ik_success,
            'is_success': task_complete,
        }

        # Compute reward
        reward = self.reward_computer.compute(self._prev_obs, action, next_obs, info)

        # Check termination and truncation
        terminated = self._check_termination(info)
        truncated = self._check_truncation()

        # Update previous observation
        self._prev_obs = next_obs.copy()

        return next_obs, reward, terminated, truncated, info

    def _scale_action(self, action):
        """Scale action from [-1, 1] to physical units.

        Args:
            action: 7D numpy array with values in [-1, 1]

        Returns:
            Scaled action with physical units
        """
        scaled = np.zeros(7, dtype=np.float32)
        scaled[0:3] = action[0:3] * self.action_scale['position']
        scaled[3:6] = action[3:6] * self.action_scale['rotation']
        scaled[6] = action[6] * self.action_scale['gripper']
        return scaled

    def _apply_ik_control(self):
        """Apply inverse kinematics control to reach end-effector target.

        Returns:
            True if IK solution was found and applied, False otherwise
        """
        # Convert euler angles to quaternion
        target_quat = euler_angles_to_quat(self._ee_target_euler)

        # Compute IK
        actions, success = self.ik_solver.compute_inverse_kinematics(
            target_position=self._ee_target_position,
            target_orientation=target_quat
        )

        if success:
            # Apply the IK solution to arm joints (indices 0-6)
            self.franka.get_articulation_controller().apply_action(actions)

        return success

    def _apply_gripper_control(self):
        """Apply gripper control based on target width."""
        current_positions = self.franka.get_joint_positions()
        new_positions = current_positions.copy()
        # Set both gripper finger positions (indices 7 and 8)
        finger_pos = self._gripper_target / 2.0  # Each finger is half the width
        new_positions[7] = finger_pos
        new_positions[8] = finger_pos
        self.franka.set_joint_positions(new_positions)

    def _build_observation(self):
        """Build observation vector from current state.

        Returns:
            23D observation vector as float32 numpy array
        """
        # Get robot state
        joint_positions = self.franka.get_joint_positions()[:7]  # Arm joints only
        ee_pos, ee_quat = self.franka.end_effector.get_world_pose()
        gripper_width = self.franka.get_joint_positions()[7] * 2  # Both fingers

        # Get scene state
        cube_pos, _ = self.scene_manager.get_cube_pose()
        goal_pos = self.scene_manager.get_goal_position()

        # Handle None values
        if cube_pos is None:
            cube_pos = np.zeros(3)
        if goal_pos is None:
            goal_pos = np.zeros(3)

        return self.obs_builder.build(
            joint_positions=joint_positions,
            ee_position=ee_pos,
            ee_orientation=ee_quat,
            gripper_width=gripper_width,
            cube_position=cube_pos,
            goal_position=goal_pos,
            cube_grasped=self._cube_grasped
        )

    def _check_termination(self, info):
        """Check if episode should terminate (success or failure).

        Args:
            info: Info dictionary from step

        Returns:
            True if episode should terminate, False otherwise
        """
        # SUCCESS: Task completed (cube at goal)
        if info.get('task_complete', False):
            return True

        # FAILURE: Cube dropped after grasping
        if info.get('cube_dropped', False):
            return True

        # FAILURE: End-effector out of workspace bounds
        ee_pos, _ = self.franka.end_effector.get_world_pose()
        if self._is_out_of_bounds(ee_pos):
            return True

        # FAILURE: Cube fell off table (z < 0)
        cube_pos, _ = self.scene_manager.get_cube_pose()
        if cube_pos is not None and cube_pos[2] < 0:
            return True

        return False

    def _check_truncation(self):
        """Check if episode should be truncated (time limit).

        Returns:
            True if episode should be truncated, False otherwise
        """
        return self._step_count >= self.max_episode_steps

    def _is_out_of_bounds(self, position):
        """Check if position is outside workspace bounds with margin.

        Args:
            position: [x, y, z] position

        Returns:
            True if position is out of bounds, False otherwise
        """
        margin = 0.15  # Allow some margin beyond workspace
        bounds = self.workspace_bounds

        return (
            position[0] < bounds['x'][0] - margin or
            position[0] > bounds['x'][1] + margin or
            position[1] < bounds['y'][0] - margin or
            position[1] > bounds['y'][1] + margin or
            position[2] < 0.0 or
            position[2] > bounds['z'][1] + 0.3
        )

    def close(self):
        """Clean up resources."""
        if self.simulation_app is not None:
            self.simulation_app.close()
