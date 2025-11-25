"""
Test Suite for Franka RL Environment

This test suite covers:
- Space definitions (observation and action spaces)
- Reset functionality
- Step functionality and action scaling
- Termination and truncation conditions
- Integration with existing helper classes

Run with: ./run_tests.sh -v src/test_franka_rl_env.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys

# ============================================================================
# MOCK GYMNASIUM MODULE (before importing franka_rl_env)
# ============================================================================
# Mock gymnasium since it may not be installed in Isaac Sim's Python environment

class MockBox:
    """Mock gymnasium.spaces.Box."""
    def __init__(self, low, high, shape, dtype):
        if np.isscalar(low):
            self.low = np.full(shape, low, dtype=dtype)
        else:
            self.low = np.array(low, dtype=dtype)
        if np.isscalar(high):
            self.high = np.full(shape, high, dtype=dtype)
        else:
            self.high = np.array(high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        # Handle -inf and inf bounds
        low = np.where(np.isinf(self.low), -1e6, self.low)
        high = np.where(np.isinf(self.high), 1e6, self.high)
        return np.random.uniform(low, high).astype(self.dtype)

    def contains(self, x):
        if not hasattr(x, 'shape'):
            return False
        return x.shape == self.shape


class MockSpaces:
    """Mock gymnasium.spaces module."""
    Box = MockBox


class MockEnv:
    """Mock gymnasium.Env base class."""
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.metadata = {}
        self._np_random = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        return None, {}

    def step(self, action):
        return None, 0.0, False, False, {}

    def close(self):
        pass


# Create a proper module-like mock for gymnasium
import types
mock_gymnasium = types.ModuleType('gymnasium')
mock_gymnasium.Env = MockEnv
mock_gymnasium.spaces = types.ModuleType('gymnasium.spaces')
mock_gymnasium.spaces.Box = MockBox

sys.modules['gymnasium'] = mock_gymnasium
sys.modules['gymnasium.spaces'] = mock_gymnasium.spaces

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing franka_rl_env)
# ============================================================================
# These mocks must be set up BEFORE importing franka_rl_env
# because the module imports isaacsim packages at the top level

# Create mock modules for isaacsim
mock_isaacsim = MagicMock()
mock_isaacsim_core = MagicMock()
mock_isaacsim_core_api = MagicMock()
mock_isaacsim_core_api_objects = MagicMock()
mock_isaacsim_core_utils = MagicMock()
mock_isaacsim_core_utils_types = MagicMock()
mock_isaacsim_core_utils_rotations = MagicMock()
mock_isaacsim_robot = MagicMock()
mock_isaacsim_robot_manipulators = MagicMock()
mock_isaacsim_robot_manipulators_examples = MagicMock()
mock_isaacsim_robot_manipulators_examples_franka = MagicMock()

# Mock SimulationApp class
mock_simulation_app_class = MagicMock()
mock_isaacsim.SimulationApp = mock_simulation_app_class


# Mock ArticulationAction class to properly store joint_positions
class MockArticulationAction:
    """Mock ArticulationAction that stores joint_positions."""
    def __init__(self, joint_positions=None, **kwargs):
        self.joint_positions = joint_positions if joint_positions is not None else np.array([])


mock_isaacsim_core_utils_types.ArticulationAction = MockArticulationAction

# Mock rotation utilities
mock_isaacsim_core_utils_rotations.euler_angles_to_quat = Mock(
    return_value=np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion
)
mock_isaacsim_core_utils_rotations.quat_to_euler_angles = Mock(
    return_value=np.array([np.pi, 0.0, 0.0])
)

# Register all mocks in sys.modules before any imports
sys.modules['isaacsim'] = mock_isaacsim
sys.modules['isaacsim.core'] = mock_isaacsim_core
sys.modules['isaacsim.core.api'] = mock_isaacsim_core_api
sys.modules['isaacsim.core.api.objects'] = mock_isaacsim_core_api_objects
sys.modules['isaacsim.core.utils'] = mock_isaacsim_core_utils
sys.modules['isaacsim.core.utils.types'] = mock_isaacsim_core_utils_types
sys.modules['isaacsim.core.utils.rotations'] = mock_isaacsim_core_utils_rotations
sys.modules['isaacsim.robot'] = mock_isaacsim_robot
sys.modules['isaacsim.robot.manipulators'] = mock_isaacsim_robot_manipulators
sys.modules['isaacsim.robot.manipulators.examples'] = mock_isaacsim_robot_manipulators_examples
sys.modules['isaacsim.robot.manipulators.examples.franka'] = mock_isaacsim_robot_manipulators_examples_franka


# ============================================================================
# MOCK FIXTURES - Isaac Sim Dependencies
# ============================================================================

@pytest.fixture
def mock_simulation_app():
    """Mock SimulationApp to avoid launching Isaac Sim."""
    with patch('franka_rl_env.SimulationApp') as mock_app:
        mock_instance = Mock()
        mock_instance.is_running.return_value = True
        mock_instance.close = Mock()
        mock_app.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_world():
    """Mock World class for simulation control."""
    world = Mock()
    world.is_playing.return_value = True
    world.is_stopped.return_value = False
    world.step = Mock()
    world.reset = Mock()
    world.scene = Mock()
    world.scene.add = Mock(side_effect=lambda x: x)
    world.scene.add_default_ground_plane = Mock()
    return world


@pytest.fixture
def mock_franka():
    """Mock Franka robot with typical joint positions."""
    franka = Mock()
    # Typical home position: [0, -1, 0, -2.2, 0, 2.4, 0.8, 0.04, 0.04]
    franka.get_joint_positions.return_value = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04])
    franka.set_joint_positions = Mock()

    # Mock end effector pose
    franka.end_effector = Mock()
    franka.end_effector.get_world_pose.return_value = (
        np.array([0.4, 0.0, 0.3]),  # position in workspace
        np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (identity)
    )

    # Mock articulation controller
    controller = Mock()
    controller.apply_action = Mock()
    franka.get_articulation_controller.return_value = controller

    return franka


@pytest.fixture
def mock_ik_solver():
    """Mock KinematicsSolver for IK computation."""
    solver = Mock()
    # Default: IK always succeeds
    mock_actions = Mock()
    solver.compute_inverse_kinematics.return_value = (mock_actions, True)
    return solver


@pytest.fixture
def mock_scene_manager():
    """Mock SceneManager for scene object management."""
    scene_manager = Mock()
    scene_manager.spawn_cube = Mock(return_value=Mock())
    scene_manager.spawn_goal_marker = Mock(return_value=np.array([0.5, 0.1, 0.05]))
    scene_manager.get_cube_pose = Mock(return_value=(np.array([0.4, 0.0, 0.05]), np.array([0, 0, 0, 1])))
    scene_manager.get_goal_position = Mock(return_value=np.array([0.5, 0.1, 0.05]))
    scene_manager.reset_scene = Mock(return_value=([0.4, 0.0, 0.05], [0.5, 0.1, 0.05]))
    scene_manager.check_grasp = Mock(return_value=False)
    scene_manager.check_task_complete = Mock(return_value=False)
    return scene_manager


@pytest.fixture
def mock_env(mock_simulation_app, mock_world, mock_franka, mock_ik_solver, mock_scene_manager):
    """Create a FrankaPickPlaceEnv instance with all mocks."""
    with patch('franka_rl_env.World', return_value=mock_world), \
         patch('franka_rl_env.Franka', return_value=mock_franka), \
         patch('franka_rl_env.KinematicsSolver', return_value=mock_ik_solver), \
         patch('franka_rl_env.SceneManager', return_value=mock_scene_manager):

        # Import the module and set the global variables to mocks
        import franka_rl_env
        franka_rl_env.simulation_app = mock_simulation_app

        # Set the Isaac Sim global functions to mocked versions
        franka_rl_env.World = Mock(return_value=mock_world)
        franka_rl_env.ArticulationAction = MockArticulationAction
        franka_rl_env.Franka = Mock(return_value=mock_franka)
        franka_rl_env.KinematicsSolver = Mock(return_value=mock_ik_solver)
        franka_rl_env.euler_angles_to_quat = mock_isaacsim_core_utils_rotations.euler_angles_to_quat
        franka_rl_env.quat_to_euler_angles = mock_isaacsim_core_utils_rotations.quat_to_euler_angles

        from franka_rl_env import FrankaPickPlaceEnv
        env = FrankaPickPlaceEnv()
        env.world = mock_world
        env.franka = mock_franka
        env.ik_solver = mock_ik_solver
        env.simulation_app = mock_simulation_app
        env.scene_manager = mock_scene_manager

        yield env


# ============================================================================
# TEST SUITE 1: Space Definitions
# ============================================================================

class TestSpaceDefinitions:
    """Tests for observation and action space definitions."""

    def test_observation_space_shape(self, mock_env):
        """Test observation space has correct shape (23D)."""
        assert mock_env.observation_space.shape == (23,)

    def test_observation_space_dtype(self, mock_env):
        """Test observation space has correct dtype."""
        assert mock_env.observation_space.dtype == np.float32

    def test_action_space_shape(self, mock_env):
        """Test action space has correct shape (7D)."""
        assert mock_env.action_space.shape == (7,)

    def test_action_space_bounds(self, mock_env):
        """Test action space bounds are [-1, 1]."""
        assert np.all(mock_env.action_space.low == -1.0)
        assert np.all(mock_env.action_space.high == 1.0)

    def test_action_space_dtype(self, mock_env):
        """Test action space has correct dtype."""
        assert mock_env.action_space.dtype == np.float32

    def test_sample_observation_in_space(self, mock_env):
        """Test sampled observations are in observation space."""
        for _ in range(10):
            obs = mock_env.observation_space.sample()
            assert mock_env.observation_space.contains(obs)

    def test_sample_action_in_space(self, mock_env):
        """Test sampled actions are in action space."""
        for _ in range(10):
            action = mock_env.action_space.sample()
            assert mock_env.action_space.contains(action)


# ============================================================================
# TEST SUITE 2: Reset Functionality
# ============================================================================

class TestResetFunctionality:
    """Tests for environment reset."""

    def test_reset_returns_tuple(self, mock_env):
        """Test reset returns (observation, info) tuple."""
        result = mock_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self, mock_env):
        """Test reset returns observation with correct shape."""
        obs, info = mock_env.reset()
        assert obs.shape == (23,)

    def test_reset_observation_dtype(self, mock_env):
        """Test reset returns observation with correct dtype."""
        obs, info = mock_env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_dict_keys(self, mock_env):
        """Test reset returns info dict with expected keys."""
        obs, info = mock_env.reset()
        assert isinstance(info, dict)
        assert 'cube_position' in info
        assert 'goal_position' in info
        assert 'is_success' in info

    def test_reset_clears_step_count(self, mock_env):
        """Test reset clears the step counter."""
        mock_env._step_count = 100
        mock_env.reset()
        assert mock_env._step_count == 0

    def test_reset_clears_grasp_state(self, mock_env):
        """Test reset clears grasp state."""
        mock_env._cube_grasped = True
        mock_env._prev_grasped = True
        mock_env.reset()
        assert mock_env._cube_grasped == False
        assert mock_env._prev_grasped == False

    def test_reset_with_seed(self, mock_env):
        """Test reset accepts seed parameter."""
        obs1, _ = mock_env.reset(seed=42)
        assert obs1 is not None


# ============================================================================
# TEST SUITE 3: Step Functionality
# ============================================================================

class TestStepFunctionality:
    """Tests for environment step."""

    def test_step_returns_tuple(self, mock_env):
        """Test step returns 5-tuple (obs, reward, terminated, truncated, info)."""
        mock_env.reset()
        action = np.zeros(7)
        result = mock_env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observation_shape(self, mock_env):
        """Test step returns observation with correct shape."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert obs.shape == (23,)

    def test_step_observation_dtype(self, mock_env):
        """Test step returns observation with correct dtype."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert obs.dtype == np.float32

    def test_step_reward_is_scalar(self, mock_env):
        """Test step returns scalar reward."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert isinstance(reward, (int, float))

    def test_step_terminated_is_bool(self, mock_env):
        """Test step returns bool for terminated."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, mock_env):
        """Test step returns bool for truncated."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, mock_env):
        """Test step returns info as dict."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert isinstance(info, dict)

    def test_step_increments_counter(self, mock_env):
        """Test step increments step counter."""
        mock_env.reset()
        assert mock_env._step_count == 0
        mock_env.step(np.zeros(7))
        assert mock_env._step_count == 1
        mock_env.step(np.zeros(7))
        assert mock_env._step_count == 2

    def test_step_clips_action(self, mock_env):
        """Test step clips actions to [-1, 1]."""
        mock_env.reset()
        # Action outside bounds should be clipped
        action = np.array([2.0, -2.0, 1.5, -1.5, 0.5, -0.5, 0.0])
        obs, reward, terminated, truncated, info = mock_env.step(action)
        # Should not raise error
        assert obs is not None

    def test_step_info_has_required_keys(self, mock_env):
        """Test step info dict has required keys."""
        mock_env.reset()
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert 'cube_grasped' in info
        assert 'task_complete' in info
        assert 'is_success' in info


# ============================================================================
# TEST SUITE 4: Action Scaling
# ============================================================================

class TestActionScaling:
    """Tests for action scaling from [-1, 1] to physical units."""

    def test_scale_action_position(self, mock_env):
        """Test position action scaling."""
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        scaled = mock_env._scale_action(action)
        assert np.isclose(scaled[0], mock_env.action_scale['position'])
        assert np.isclose(scaled[1], 0.0)
        assert np.isclose(scaled[2], 0.0)

    def test_scale_action_rotation(self, mock_env):
        """Test rotation action scaling."""
        action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        scaled = mock_env._scale_action(action)
        assert np.isclose(scaled[3], mock_env.action_scale['rotation'])
        assert np.isclose(scaled[4], 0.0)
        assert np.isclose(scaled[5], 0.0)

    def test_scale_action_gripper(self, mock_env):
        """Test gripper action scaling."""
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        scaled = mock_env._scale_action(action)
        assert np.isclose(scaled[6], mock_env.action_scale['gripper'])

    def test_scale_action_negative(self, mock_env):
        """Test negative action scaling."""
        action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        scaled = mock_env._scale_action(action)
        assert np.isclose(scaled[0], -mock_env.action_scale['position'])
        assert np.isclose(scaled[3], -mock_env.action_scale['rotation'])
        assert np.isclose(scaled[6], -mock_env.action_scale['gripper'])

    def test_scale_action_zero(self, mock_env):
        """Test zero action scaling."""
        action = np.zeros(7)
        scaled = mock_env._scale_action(action)
        assert np.allclose(scaled, 0.0)

    def test_default_action_scale_values(self, mock_env):
        """Test default action scale values match constants."""
        assert np.isclose(mock_env.action_scale['position'], 0.02)
        assert np.isclose(mock_env.action_scale['rotation'], 0.1)
        assert np.isclose(mock_env.action_scale['gripper'], 0.01)


# ============================================================================
# TEST SUITE 5: Termination Conditions
# ============================================================================

class TestTerminationConditions:
    """Tests for episode termination conditions."""

    def test_termination_on_task_complete(self, mock_env):
        """Test episode terminates on task completion."""
        info = {'task_complete': True}
        assert mock_env._check_termination(info) == True

    def test_termination_on_cube_dropped(self, mock_env):
        """Test episode terminates when cube is dropped."""
        info = {'cube_dropped': True}
        assert mock_env._check_termination(info) == True

    def test_no_termination_normal_state(self, mock_env):
        """Test episode does not terminate in normal state."""
        info = {'task_complete': False, 'cube_dropped': False}
        # EE in workspace, cube on table
        assert mock_env._check_termination(info) == False

    def test_termination_ee_out_of_bounds(self, mock_env):
        """Test episode terminates when EE is out of bounds."""
        # Move EE way out of bounds
        mock_env.franka.end_effector.get_world_pose.return_value = (
            np.array([10.0, 0.0, 0.0]),  # Far outside workspace
            np.array([0.0, 0.0, 0.0, 1.0])
        )
        info = {'task_complete': False, 'cube_dropped': False}
        assert mock_env._check_termination(info) == True


# ============================================================================
# TEST SUITE 6: Truncation Conditions
# ============================================================================

class TestTruncationConditions:
    """Tests for episode truncation conditions."""

    def test_truncation_at_max_steps(self, mock_env):
        """Test episode truncates at max_episode_steps."""
        mock_env._step_count = mock_env.max_episode_steps
        assert mock_env._check_truncation() == True

    def test_no_truncation_before_max_steps(self, mock_env):
        """Test episode does not truncate before max_episode_steps."""
        mock_env._step_count = mock_env.max_episode_steps - 1
        assert mock_env._check_truncation() == False

    def test_no_truncation_at_zero_steps(self, mock_env):
        """Test episode does not truncate at zero steps."""
        mock_env._step_count = 0
        assert mock_env._check_truncation() == False

    def test_custom_max_episode_steps(self, mock_simulation_app, mock_world, mock_franka, mock_ik_solver):
        """Test environment respects custom max_episode_steps."""
        with patch('franka_rl_env.World', return_value=mock_world), \
             patch('franka_rl_env.Franka', return_value=mock_franka), \
             patch('franka_rl_env.KinematicsSolver', return_value=mock_ik_solver):

            import franka_rl_env
            franka_rl_env.simulation_app = mock_simulation_app
            franka_rl_env.World = Mock(return_value=mock_world)
            franka_rl_env.ArticulationAction = MockArticulationAction
            franka_rl_env.Franka = Mock(return_value=mock_franka)
            franka_rl_env.KinematicsSolver = Mock(return_value=mock_ik_solver)
            franka_rl_env.euler_angles_to_quat = mock_isaacsim_core_utils_rotations.euler_angles_to_quat
            franka_rl_env.quat_to_euler_angles = mock_isaacsim_core_utils_rotations.quat_to_euler_angles

            from franka_rl_env import FrankaPickPlaceEnv
            env = FrankaPickPlaceEnv(max_episode_steps=100)
            env.world = mock_world
            env.franka = mock_franka
            env.ik_solver = mock_ik_solver

            assert env.max_episode_steps == 100
            env._step_count = 100
            assert env._check_truncation() == True


# ============================================================================
# TEST SUITE 7: Out of Bounds Checking
# ============================================================================

class TestOutOfBoundsChecking:
    """Tests for workspace bounds checking."""

    def test_position_in_bounds(self, mock_env):
        """Test position within bounds is not out of bounds."""
        # Center of workspace
        pos = np.array([0.45, 0.0, 0.1])
        assert mock_env._is_out_of_bounds(pos) == False

    def test_position_x_too_low(self, mock_env):
        """Test position with x too low is out of bounds."""
        pos = np.array([0.0, 0.0, 0.1])  # x < 0.3 - margin
        assert mock_env._is_out_of_bounds(pos) == True

    def test_position_x_too_high(self, mock_env):
        """Test position with x too high is out of bounds."""
        pos = np.array([1.0, 0.0, 0.1])  # x > 0.6 + margin
        assert mock_env._is_out_of_bounds(pos) == True

    def test_position_y_too_low(self, mock_env):
        """Test position with y too low is out of bounds."""
        pos = np.array([0.45, -0.6, 0.1])  # y < -0.3 - margin
        assert mock_env._is_out_of_bounds(pos) == True

    def test_position_y_too_high(self, mock_env):
        """Test position with y too high is out of bounds."""
        pos = np.array([0.45, 0.6, 0.1])  # y > 0.3 + margin
        assert mock_env._is_out_of_bounds(pos) == True

    def test_position_z_too_low(self, mock_env):
        """Test position with z below 0 is out of bounds."""
        pos = np.array([0.45, 0.0, -0.1])  # z < 0
        assert mock_env._is_out_of_bounds(pos) == True

    def test_position_z_too_high(self, mock_env):
        """Test position with z too high is out of bounds."""
        pos = np.array([0.45, 0.0, 1.0])  # z > 0.2 + 0.3
        assert mock_env._is_out_of_bounds(pos) == True


# ============================================================================
# TEST SUITE 8: Reward Modes
# ============================================================================

class TestRewardModes:
    """Tests for different reward modes."""

    def test_default_reward_mode_is_dense(self, mock_env):
        """Test default reward mode is dense."""
        assert mock_env.reward_mode == 'dense'

    def test_sparse_reward_mode(self, mock_simulation_app, mock_world, mock_franka, mock_ik_solver):
        """Test environment can be created with sparse reward mode."""
        with patch('franka_rl_env.World', return_value=mock_world), \
             patch('franka_rl_env.Franka', return_value=mock_franka), \
             patch('franka_rl_env.KinematicsSolver', return_value=mock_ik_solver):

            import franka_rl_env
            franka_rl_env.simulation_app = mock_simulation_app
            franka_rl_env.World = Mock(return_value=mock_world)
            franka_rl_env.ArticulationAction = MockArticulationAction
            franka_rl_env.Franka = Mock(return_value=mock_franka)
            franka_rl_env.KinematicsSolver = Mock(return_value=mock_ik_solver)
            franka_rl_env.euler_angles_to_quat = mock_isaacsim_core_utils_rotations.euler_angles_to_quat
            franka_rl_env.quat_to_euler_angles = mock_isaacsim_core_utils_rotations.quat_to_euler_angles

            from franka_rl_env import FrankaPickPlaceEnv
            env = FrankaPickPlaceEnv(reward_mode='sparse')
            env.world = mock_world
            env.franka = mock_franka
            env.ik_solver = mock_ik_solver

            assert env.reward_mode == 'sparse'


# ============================================================================
# TEST SUITE 9: Close Functionality
# ============================================================================

class TestCloseFunctionality:
    """Tests for environment cleanup."""

    def test_close_calls_simulation_app_close(self, mock_env):
        """Test close calls simulation_app.close()."""
        mock_env.close()
        mock_env.simulation_app.close.assert_called_once()


# ============================================================================
# TEST SUITE 10: Helper Class Integration
# ============================================================================

class TestHelperClassIntegration:
    """Tests for integration with helper classes."""

    def test_observation_builder_initialized(self, mock_env):
        """Test ObservationBuilder is initialized."""
        assert mock_env.obs_builder is not None
        assert mock_env.obs_builder.obs_dim == 23

    def test_reward_computer_initialized(self, mock_env):
        """Test RewardComputer is initialized."""
        assert mock_env.reward_computer is not None

    def test_scene_manager_initialized(self, mock_env):
        """Test SceneManager is initialized."""
        assert mock_env.scene_manager is not None


# ============================================================================
# TEST SUITE 11: Gymnasium API Compliance
# ============================================================================

class TestGymnasiumAPICompliance:
    """Tests for Gymnasium API compliance."""

    def test_metadata_exists(self, mock_env):
        """Test metadata attribute exists."""
        assert hasattr(mock_env, 'metadata')
        # Check class-level metadata (defined on the class, not instance)
        from franka_rl_env import FrankaPickPlaceEnv
        assert hasattr(FrankaPickPlaceEnv, 'metadata')
        assert 'render_modes' in FrankaPickPlaceEnv.metadata

    def test_observation_space_exists(self, mock_env):
        """Test observation_space attribute exists."""
        assert hasattr(mock_env, 'observation_space')

    def test_action_space_exists(self, mock_env):
        """Test action_space attribute exists."""
        assert hasattr(mock_env, 'action_space')

    def test_reset_method_exists(self, mock_env):
        """Test reset method exists."""
        assert hasattr(mock_env, 'reset')
        assert callable(mock_env.reset)

    def test_step_method_exists(self, mock_env):
        """Test step method exists."""
        assert hasattr(mock_env, 'step')
        assert callable(mock_env.step)

    def test_close_method_exists(self, mock_env):
        """Test close method exists."""
        assert hasattr(mock_env, 'close')
        assert callable(mock_env.close)


# ============================================================================
# TEST SUITE 12: Episode Loop
# ============================================================================

class TestEpisodeLoop:
    """Tests for complete episode execution."""

    def test_basic_episode_loop(self, mock_env):
        """Test running a basic episode loop."""
        obs, info = mock_env.reset()
        assert obs is not None

        done = False
        steps = 0
        max_steps = 10

        while not done and steps < max_steps:
            action = mock_env.action_space.sample()
            obs, reward, terminated, truncated, info = mock_env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0
        assert obs.shape == (23,)

    def test_multiple_episodes(self, mock_env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = mock_env.reset()
            assert mock_env._step_count == 0

            for step in range(5):
                action = mock_env.action_space.sample()
                obs, reward, terminated, truncated, info = mock_env.step(action)
                if terminated or truncated:
                    break

            assert mock_env._step_count > 0
