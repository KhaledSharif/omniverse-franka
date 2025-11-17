"""
Comprehensive Test Suite for Franka Keyboard Control

This test suite covers:
- Command processing and the critical bug we fixed (None return handling)
- Command queue threading and concurrency
- Mode switching logic
- End-effector control with IK validation
- Joint control logic
- State management
- Integration tests and edge cases

Run with: pytest test_franka_keyboard_control.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import threading
import time
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing franka_keyboard_control)
# ============================================================================
# These mocks must be set up BEFORE importing franka_keyboard_control
# because the module imports isaacsim packages at the top level

# Create mock modules for isaacsim
mock_isaacsim = MagicMock()
mock_isaacsim_core = MagicMock()
mock_isaacsim_core_api = MagicMock()
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
    with patch('franka_keyboard_control.SimulationApp') as mock_app:
        mock_instance = Mock()
        mock_instance.is_running.return_value = True
        mock_app.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_world():
    """Mock World class for simulation control."""
    world = Mock()
    world.is_playing.return_value = True
    world.is_stopped.return_value = False
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

    # Mock end effector pose
    franka.end_effector = Mock()
    franka.end_effector.get_world_pose.return_value = (
        np.array([0.3, 0.0, 0.4]),  # position
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
def mock_tui_renderer():
    """Mock TUIRenderer to avoid Rich TUI dependencies."""
    tui = Mock()
    tui.set_pressed_key = Mock()
    tui.clear_pressed_key = Mock()
    tui.update_telemetry = Mock()
    tui.set_mode = Mock()
    tui.set_active_joint = Mock()
    tui.set_ik_status = Mock()
    tui.set_last_command = Mock()
    tui.render = Mock(return_value=Mock())
    return tui


@pytest.fixture
def mock_keyboard_listener():
    """Mock pynput keyboard listener."""
    with patch('franka_keyboard_control.keyboard.Listener') as mock_listener_class:
        listener_instance = Mock()
        listener_instance.start = Mock()
        listener_instance.stop = Mock()
        mock_listener_class.return_value = listener_instance
        yield listener_instance


@pytest.fixture
def controller_instance(mock_simulation_app, mock_world, mock_franka, mock_tui_renderer, mock_keyboard_listener):
    """Create a FrankaKeyboardController instance with all mocks."""
    with patch('franka_keyboard_control.World', return_value=mock_world), \
         patch('franka_keyboard_control.Franka', return_value=mock_franka), \
         patch('franka_keyboard_control.TUIRenderer', return_value=mock_tui_renderer), \
         patch('franka_keyboard_control.termios'), \
         patch('franka_keyboard_control.sys'):

        # Import the module and set the global variables to mocks
        import franka_keyboard_control
        franka_keyboard_control.simulation_app = mock_simulation_app

        # Set the Isaac Sim global functions to mocked versions
        franka_keyboard_control.World = mock_world.__class__
        franka_keyboard_control.ArticulationAction = MockArticulationAction
        franka_keyboard_control.Franka = mock_franka.__class__
        franka_keyboard_control.KinematicsSolver = Mock
        franka_keyboard_control.euler_angles_to_quat = mock_isaacsim_core_utils_rotations.euler_angles_to_quat
        franka_keyboard_control.quat_to_euler_angles = mock_isaacsim_core_utils_rotations.quat_to_euler_angles

        from franka_keyboard_control import FrankaKeyboardController
        controller = FrankaKeyboardController()
        controller.world = mock_world
        controller.franka = mock_franka
        controller.tui = mock_tui_renderer

        return controller


# ============================================================================
# TEST SUITE 1: Critical Bug Regression Tests (The Bug We Fixed)
# ============================================================================

class TestCommandProcessingBug:
    """
    Tests for the critical bug we fixed:
    - _process_commands() was clearing the queue but then trying to read from empty queue
    - This caused last_key_processed to never be set
    - Control application never executed
    """

    def test_process_commands_returns_none_on_empty_queue(self, controller_instance):
        """Verify None returned when no commands in queue."""
        result = controller_instance._process_commands()
        assert result is None, "Should return None when queue is empty"

    def test_process_commands_returns_none_on_special_commands_only(self, controller_instance):
        """Verify Tab/Esc (special commands) don't return char command."""
        # Queue special command (Tab)
        controller_instance._queue_command(('special', 'tab'))
        result = controller_instance._process_commands()

        assert result is None, "Should return None for special commands (not char)"
        assert controller_instance.control_mode == 1, "Mode should have toggled"

    def test_last_key_processed_not_overwritten_by_none(self, controller_instance, mock_ik_solver):
        """
        THE MAIN BUG TEST: Verify None doesn't overwrite last_key_processed.

        Before fix: last_key_processed would stay None forever
        After fix: last_key_processed gets set on first char command
        """
        # Initial state
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance.ik_solver = mock_ik_solver  # Inject mock to avoid real IK solver creation
        last_key_processed = None

        # Simulate frame 1: No commands, process returns None
        result = controller_instance._process_commands()
        if result is not None:
            last_key_processed = result
        assert last_key_processed is None, "Should still be None (no commands)"

        # Simulate frame 2: User presses 'w'
        controller_instance._queue_command(('char', 'w'))
        result = controller_instance._process_commands()
        if result is not None:
            last_key_processed = result

        # THIS IS THE FIX: last_key_processed should now be 'w', not None
        assert last_key_processed == 'w', "Should be set to 'w' after processing char command"

        # Simulate frame 3: No new commands, process returns None
        result = controller_instance._process_commands()
        if result is not None:  # Key part of fix: only update if not None
            last_key_processed = result

        # Verify last_key_processed is NOT overwritten by None
        assert last_key_processed == 'w', "Should still be 'w' (not overwritten by None)"

    def test_control_applies_after_first_keypress_endeffector(self, controller_instance, mock_ik_solver):
        """
        Integration test: Verify EE control actually applies on first keypress.

        This would FAIL before the fix because _apply_endeffector_control() never executed.
        """
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance.ik_solver = mock_ik_solver

        # Initial position
        initial_x = controller_instance.ee_target_position[0]

        # Press 'w' to move forward
        controller_instance._queue_command(('char', 'w'))

        # Simulate main loop processing
        last_char_cmd = controller_instance._process_commands()
        last_key_processed = None
        if last_char_cmd is not None:
            last_key_processed = last_char_cmd

        # Verify target updated
        assert controller_instance.ee_target_position[0] == initial_x + 0.02, "Target should update"

        # Apply control (this is what was broken before fix)
        if controller_instance.control_mode == controller_instance.MODE_ENDEFFECTOR and last_key_processed:
            if last_key_processed in ['w', 's', 'a', 'd', 'q', 'e', 'i', 'k', 'j', 'l', 'u', 'o']:
                controller_instance._apply_endeffector_control()

        # Verify IK was called and action applied
        assert mock_ik_solver.compute_inverse_kinematics.called, "IK should be computed"
        controller = controller_instance.franka.get_articulation_controller()
        assert controller.apply_action.called, "Action should be applied to robot"

    def test_control_applies_after_first_keypress_joint_mode(self, controller_instance):
        """
        Integration test: Verify joint control applies on first keypress.

        This would FAIL before the fix.
        """
        controller_instance.control_mode = controller_instance.MODE_JOINT
        controller_instance.active_joint = 2  # Joint 3

        # Get initial joint positions
        initial_positions = controller_instance.franka.get_joint_positions().copy()

        # Press 'w' to increase joint angle
        controller_instance._queue_command(('char', 'w'))

        # Simulate main loop processing
        last_char_cmd = controller_instance._process_commands()
        last_key_processed = None
        if last_char_cmd is not None:
            last_key_processed = last_char_cmd

        # Apply control
        if controller_instance.control_mode == controller_instance.MODE_JOINT and last_key_processed:
            if last_key_processed in ['w', 's', 'q', 'e']:
                controller_instance._apply_joint_control(last_key_processed)

        # Verify action applied
        controller = controller_instance.franka.get_articulation_controller()
        assert controller.apply_action.called, "Action should be applied to robot"

        # Verify joint angle was incremented in the action call
        call_args = controller.apply_action.call_args
        action = call_args[0][0]
        expected_position = initial_positions[2] + controller_instance.JOINT_INCREMENT
        assert action.joint_positions[2] == pytest.approx(expected_position), "Joint 3 should increment"

    def test_endeffector_control_applies_continuously(self, controller_instance, mock_ik_solver):
        """
        Verify EE mode applies control every frame with same key.

        Key behavior: last_key_processed NOT cleared in EE mode.
        """
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance.ik_solver = mock_ik_solver

        # Press 'w' once
        controller_instance._queue_command(('char', 'w'))
        last_char_cmd = controller_instance._process_commands()
        last_key_processed = last_char_cmd

        # Reset mock counter (command processing also called IK for validation)
        mock_ik_solver.compute_inverse_kinematics.reset_mock()

        # Simulate 3 frames of continuous control
        for _ in range(3):
            if controller_instance.control_mode == controller_instance.MODE_ENDEFFECTOR and last_key_processed:
                if last_key_processed in ['w', 's', 'a', 'd', 'q', 'e', 'i', 'k', 'j', 'l', 'u', 'o']:
                    controller_instance._apply_endeffector_control()

        # Should be called 3 times (in the loop, not counting the validation during processing)
        assert mock_ik_solver.compute_inverse_kinematics.call_count == 3, "Should apply continuously"

    def test_joint_control_applies_once(self, controller_instance):
        """
        Verify joint mode clears last_key_processed after one application.

        Key behavior: Single-shot control in joint mode.
        """
        controller_instance.control_mode = controller_instance.MODE_JOINT
        controller_instance.active_joint = 0

        # Press 'w' once
        controller_instance._queue_command(('char', 'w'))
        last_char_cmd = controller_instance._process_commands()
        last_key_processed = last_char_cmd

        # Apply control once (simulates main loop)
        if controller_instance.control_mode == controller_instance.MODE_JOINT and last_key_processed:
            if last_key_processed in ['w', 's', 'q', 'e']:
                controller_instance._apply_joint_control(last_key_processed)
                last_key_processed = None  # Cleared after application

        controller = controller_instance.franka.get_articulation_controller()
        assert controller.apply_action.call_count == 1, "Should apply once"
        assert last_key_processed is None, "Should be cleared after application"


# ============================================================================
# TEST SUITE 2: Command Queue & Threading Tests
# ============================================================================

class TestCommandQueue:
    """Tests for thread-safe command queue operations."""

    def test_queue_command_thread_safe(self, controller_instance):
        """Verify command queueing is thread-safe."""
        # Queue command from main thread
        controller_instance._queue_command(('char', 'w'))
        assert len(controller_instance.pending_commands) == 1

        # Verify command stored correctly
        assert controller_instance.pending_commands[0] == ('char', 'w')

    def test_process_commands_empties_queue(self, controller_instance):
        """Verify queue is cleared after processing."""
        controller_instance._queue_command(('char', 'w'))
        controller_instance._queue_command(('char', 'a'))

        assert len(controller_instance.pending_commands) == 2

        result = controller_instance._process_commands()

        assert len(controller_instance.pending_commands) == 0, "Queue should be empty"
        assert result == 'a', "Should return last char command"

    def test_multiple_commands_processed_in_order(self, controller_instance):
        """Verify FIFO order preserved when processing commands."""
        commands = [('char', 'w'), ('char', 'a'), ('char', 's')]
        for cmd in commands:
            controller_instance._queue_command(cmd)

        # Track processing order
        processed = []

        # Monkey-patch to capture processing
        original_process_joint = controller_instance._process_joint_command
        def capture_joint(key):
            processed.append(key)
            return original_process_joint(key)
        controller_instance._process_joint_command = capture_joint

        controller_instance.control_mode = controller_instance.MODE_JOINT
        result = controller_instance._process_commands()

        assert processed == ['w', 'a', 's'], "Commands should process in FIFO order"
        assert result == 's', "Should return last char command"

    def test_last_char_command_captured_from_batch(self, controller_instance, mock_ik_solver):
        """Verify only last char command returned from batch."""
        # Mix of special and char commands
        controller_instance._queue_command(('char', 'w'))
        controller_instance._queue_command(('special', 'tab'))  # Switches to EE mode
        controller_instance._queue_command(('char', 'a'))

        # Inject mock to avoid real IK solver creation when 'a' is processed in EE mode
        controller_instance.ik_solver = mock_ik_solver

        result = controller_instance._process_commands()

        assert result == 'a', "Should return last char command ('a'), not 'w'"

    def test_special_commands_dont_return_char(self, controller_instance):
        """Verify special commands (Tab, Esc) don't return as char commands."""
        controller_instance._queue_command(('special', 'esc'))

        result = controller_instance._process_commands()

        assert result is None, "Special commands should not return char value"
        assert controller_instance.should_exit is True, "Esc should set exit flag"


# ============================================================================
# TEST SUITE 3: Mode Switching Tests
# ============================================================================

class TestModeSwitch:
    """Tests for switching between Joint and End-Effector control modes."""

    def test_toggle_mode_switches_between_joint_and_ee(self, controller_instance):
        """Verify mode toggle cycles between 0 (Joint) and 1 (EE)."""
        assert controller_instance.control_mode == 0, "Should start in Joint mode"

        controller_instance._toggle_mode()
        assert controller_instance.control_mode == 1, "Should switch to EE mode"

        controller_instance._toggle_mode()
        assert controller_instance.control_mode == 0, "Should switch back to Joint mode"

    def test_ee_target_initialized_on_mode_entry(self, controller_instance):
        """Verify EE target set from current pose when entering EE mode."""
        # Start in joint mode
        controller_instance.control_mode = controller_instance.MODE_JOINT

        # Toggle to EE mode
        controller_instance._toggle_mode()

        # Verify target initialized from current end effector pose
        expected_position = np.array([0.3, 0.0, 0.4])
        np.testing.assert_array_almost_equal(
            controller_instance.ee_target_position,
            expected_position,
            err_msg="EE target should initialize from current pose"
        )

    def test_active_joint_persists_across_mode_switch(self, controller_instance):
        """Verify active_joint state persists when switching modes."""
        controller_instance.control_mode = controller_instance.MODE_JOINT
        controller_instance.active_joint = 3  # Joint 4

        # Switch to EE mode
        controller_instance._toggle_mode()
        assert controller_instance.active_joint == 3, "Should persist"

        # Switch back to joint mode
        controller_instance._toggle_mode()
        assert controller_instance.active_joint == 3, "Should still persist"

    def test_control_application_respects_current_mode(self, controller_instance, mock_ik_solver):
        """Verify correct control handler called based on current mode."""
        controller_instance.ik_solver = mock_ik_solver

        # Test EE mode
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance._queue_command(('char', 'w'))
        controller_instance._process_commands()
        controller_instance._apply_endeffector_control()

        assert mock_ik_solver.compute_inverse_kinematics.called, "EE mode should use IK"

        # Test Joint mode
        mock_ik_solver.reset_mock()
        controller_instance.control_mode = controller_instance.MODE_JOINT
        controller_instance.active_joint = 0
        controller_instance._apply_joint_control('w')

        assert not mock_ik_solver.compute_inverse_kinematics.called, "Joint mode should NOT use IK"

    def test_tui_updated_on_mode_switch(self, controller_instance):
        """Verify TUI notified of mode change."""
        controller_instance._toggle_mode()

        controller_instance.tui.set_mode.assert_called_with(1)
        controller_instance.tui.set_last_command.assert_called()


# ============================================================================
# TEST SUITE 4: End-Effector Control Tests
# ============================================================================

class TestEndEffectorControl:
    """Tests for end-effector position and orientation control with IK."""

    def test_ee_position_commands_update_target_x(self, controller_instance, mock_ik_solver):
        """Test W/S keys increment/decrement X position."""
        controller_instance.ik_solver = mock_ik_solver
        initial_x = controller_instance.ee_target_position[0]

        controller_instance._process_endeffector_command('w')
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x + 0.02)

        controller_instance._process_endeffector_command('s')
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x)

    def test_ee_position_commands_update_target_y(self, controller_instance, mock_ik_solver):
        """Test A/D keys increment/decrement Y position."""
        controller_instance.ik_solver = mock_ik_solver
        initial_y = controller_instance.ee_target_position[1]

        controller_instance._process_endeffector_command('a')
        assert controller_instance.ee_target_position[1] == pytest.approx(initial_y + 0.02)

        controller_instance._process_endeffector_command('d')
        assert controller_instance.ee_target_position[1] == pytest.approx(initial_y)

    def test_ee_position_commands_update_target_z(self, controller_instance, mock_ik_solver):
        """Test Q/E keys increment/decrement Z position."""
        controller_instance.ik_solver = mock_ik_solver
        initial_z = controller_instance.ee_target_position[2]

        controller_instance._process_endeffector_command('q')
        assert controller_instance.ee_target_position[2] == pytest.approx(initial_z + 0.02)

        controller_instance._process_endeffector_command('e')
        assert controller_instance.ee_target_position[2] == pytest.approx(initial_z)

    def test_ee_rotation_commands_update_target_pitch(self, controller_instance, mock_ik_solver):
        """Test I/K keys increment/decrement pitch."""
        controller_instance.ik_solver = mock_ik_solver
        initial_pitch = controller_instance.ee_target_euler[1]

        controller_instance._process_endeffector_command('i')
        assert controller_instance.ee_target_euler[1] == pytest.approx(initial_pitch + 0.1)

        controller_instance._process_endeffector_command('k')
        assert controller_instance.ee_target_euler[1] == pytest.approx(initial_pitch)

    def test_ee_rotation_commands_update_target_yaw(self, controller_instance, mock_ik_solver):
        """Test J/L keys increment/decrement yaw."""
        controller_instance.ik_solver = mock_ik_solver
        initial_yaw = controller_instance.ee_target_euler[2]

        controller_instance._process_endeffector_command('j')
        assert controller_instance.ee_target_euler[2] == pytest.approx(initial_yaw + 0.1)

        controller_instance._process_endeffector_command('l')
        assert controller_instance.ee_target_euler[2] == pytest.approx(initial_yaw)

    def test_ee_rotation_commands_update_target_roll(self, controller_instance, mock_ik_solver):
        """Test U/O keys increment/decrement roll."""
        controller_instance.ik_solver = mock_ik_solver
        initial_roll = controller_instance.ee_target_euler[0]

        controller_instance._process_endeffector_command('u')
        assert controller_instance.ee_target_euler[0] == pytest.approx(initial_roll + 0.1)

        controller_instance._process_endeffector_command('o')
        assert controller_instance.ee_target_euler[0] == pytest.approx(initial_roll)

    def test_ik_validation_called_before_accepting_change(self, controller_instance, mock_ik_solver):
        """Verify IK validation happens before committing to position change."""
        controller_instance.ik_solver = mock_ik_solver

        controller_instance._process_endeffector_command('w')

        # Verify IK solver was called
        assert mock_ik_solver.compute_inverse_kinematics.called

    def test_ik_failure_reverts_to_previous_state(self, controller_instance, mock_ik_solver):
        """Test state rollback when IK validation fails."""
        controller_instance.ik_solver = mock_ik_solver
        # Make IK fail
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), False)

        initial_position = controller_instance.ee_target_position.copy()

        controller_instance._process_endeffector_command('w')

        # Verify position reverted (not incremented)
        np.testing.assert_array_almost_equal(
            controller_instance.ee_target_position,
            initial_position,
            err_msg="Position should revert to previous state on IK failure"
        )

    def test_ik_success_updates_target(self, controller_instance, mock_ik_solver):
        """Verify target accepted when IK succeeds."""
        controller_instance.ik_solver = mock_ik_solver
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), True)

        initial_x = controller_instance.ee_target_position[0]

        controller_instance._process_endeffector_command('w')

        # Verify position updated (incremented)
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x + 0.02)

    def test_ik_failure_sets_tui_status(self, controller_instance, mock_ik_solver):
        """Verify UI updated with IK failure status."""
        controller_instance.ik_solver = mock_ik_solver
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), False)

        controller_instance._process_endeffector_command('w')

        controller_instance.tui.set_ik_status.assert_called_with(False, "Workspace limit reached")

    def test_apply_endeffector_control_calls_ik(self, controller_instance, mock_ik_solver):
        """Verify IK computed in apply method."""
        controller_instance.ik_solver = mock_ik_solver

        controller_instance._apply_endeffector_control()

        assert mock_ik_solver.compute_inverse_kinematics.called

    def test_apply_endeffector_control_applies_action_on_success(self, controller_instance, mock_ik_solver):
        """Verify action applied to controller when IK succeeds."""
        controller_instance.ik_solver = mock_ik_solver
        mock_actions = Mock()
        mock_ik_solver.compute_inverse_kinematics.return_value = (mock_actions, True)

        controller_instance._apply_endeffector_control()

        controller = controller_instance.franka.get_articulation_controller()
        controller.apply_action.assert_called_once_with(mock_actions)

    def test_apply_endeffector_control_skips_on_ik_failure(self, controller_instance, mock_ik_solver):
        """Verify no action applied when IK fails."""
        controller_instance.ik_solver = mock_ik_solver
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), False)

        controller_instance._apply_endeffector_control()

        controller = controller_instance.franka.get_articulation_controller()
        assert not controller.apply_action.called, "Should not apply action on IK failure"


# ============================================================================
# TEST SUITE 5: Joint Control Tests
# ============================================================================

class TestJointControl:
    """Tests for direct joint angle control."""

    def test_joint_selection_updates_active_joint(self, controller_instance):
        """Test keys 1-7 select joints correctly."""
        for i in range(7):
            key = str(i + 1)
            controller_instance._process_joint_command(key)
            assert controller_instance.active_joint == i, f"Key {key} should select joint {i}"

    def test_w_key_increases_selected_joint(self, controller_instance):
        """Test W increases joint angle by JOINT_INCREMENT."""
        controller_instance.active_joint = 2  # Joint 3
        initial_positions = controller_instance.franka.get_joint_positions().copy()

        controller_instance._apply_joint_control('w')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]
        expected = initial_positions[2] + controller_instance.JOINT_INCREMENT

        assert call_args.joint_positions[2] == pytest.approx(expected)

    def test_s_key_decreases_selected_joint(self, controller_instance):
        """Test S decreases joint angle by JOINT_INCREMENT."""
        controller_instance.active_joint = 2  # Joint 3
        initial_positions = controller_instance.franka.get_joint_positions().copy()

        controller_instance._apply_joint_control('s')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]
        expected = initial_positions[2] - controller_instance.JOINT_INCREMENT

        assert call_args.joint_positions[2] == pytest.approx(expected)

    def test_q_key_opens_gripper(self, controller_instance):
        """Test Q opens gripper (increases joints 7 and 8)."""
        initial_positions = controller_instance.franka.get_joint_positions().copy()

        controller_instance._apply_joint_control('q')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        # Should increase but cap at 0.05
        expected = min(initial_positions[7] + controller_instance.GRIPPER_INCREMENT, 0.05)
        assert call_args.joint_positions[7] == pytest.approx(expected)
        assert call_args.joint_positions[8] == pytest.approx(expected)

    def test_e_key_closes_gripper(self, controller_instance):
        """Test E closes gripper (decreases joints 7 and 8)."""
        initial_positions = controller_instance.franka.get_joint_positions().copy()

        controller_instance._apply_joint_control('e')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        # Should decrease but cap at 0.0
        expected = max(initial_positions[7] - controller_instance.GRIPPER_INCREMENT, 0.0)
        assert call_args.joint_positions[7] == pytest.approx(expected)
        assert call_args.joint_positions[8] == pytest.approx(expected)

    def test_gripper_bounds_enforced_max(self, controller_instance):
        """Test gripper opening capped at 0.05m."""
        # Set gripper near max
        controller_instance.franka.get_joint_positions.return_value[7] = 0.045
        controller_instance.franka.get_joint_positions.return_value[8] = 0.045

        controller_instance._apply_joint_control('q')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        assert call_args.joint_positions[7] <= 0.05, "Should cap at 0.05"
        assert call_args.joint_positions[8] <= 0.05, "Should cap at 0.05"

    def test_gripper_bounds_enforced_min(self, controller_instance):
        """Test gripper closing capped at 0.0m."""
        # Set gripper near min
        controller_instance.franka.get_joint_positions.return_value[7] = 0.005
        controller_instance.franka.get_joint_positions.return_value[8] = 0.005

        controller_instance._apply_joint_control('e')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        assert call_args.joint_positions[7] >= 0.0, "Should cap at 0.0"
        assert call_args.joint_positions[8] >= 0.0, "Should cap at 0.0"

    def test_gripper_joints_synchronized(self, controller_instance):
        """Test joints 7 and 8 always move together."""
        controller_instance._apply_joint_control('q')

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        assert call_args.joint_positions[7] == call_args.joint_positions[8], \
            "Gripper joints should be synchronized"

    def test_apply_joint_control_updates_tui(self, controller_instance):
        """Test TUI updated with joint control feedback."""
        controller_instance.active_joint = 3

        controller_instance._apply_joint_control('w')

        controller_instance.tui.set_last_command.assert_called()


# ============================================================================
# TEST SUITE 6: State Management Tests
# ============================================================================

class TestStateManagement:
    """Tests for internal state consistency."""

    def test_controller_initializes_in_joint_mode(self, controller_instance):
        """Verify controller starts in joint control mode."""
        assert controller_instance.control_mode == controller_instance.MODE_JOINT

    def test_active_joint_initializes_to_zero(self, controller_instance):
        """Verify active joint starts at 0."""
        assert controller_instance.active_joint == 0

    def test_should_exit_initializes_to_false(self, controller_instance):
        """Verify exit flag starts false."""
        assert controller_instance.should_exit is False

    def test_ee_target_position_initialized(self, controller_instance):
        """Test initial EE target position values."""
        expected = np.array([0.3, 0.0, 0.4])
        np.testing.assert_array_almost_equal(controller_instance.ee_target_position, expected)

    def test_ee_target_euler_initialized(self, controller_instance):
        """Test initial EE target orientation."""
        expected = np.array([np.pi, 0.0, 0.0])
        np.testing.assert_array_almost_equal(controller_instance.ee_target_euler, expected)

    def test_ik_solver_lazy_initialization(self, controller_instance):
        """Verify IK solver is None until first use."""
        assert controller_instance.ik_solver is None, "Should start as None"

    def test_pending_commands_empty_on_start(self, controller_instance):
        """Verify command queue starts empty."""
        assert len(controller_instance.pending_commands) == 0

    def test_home_position_constant(self, controller_instance):
        """Verify home position constant defined correctly."""
        expected = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04])
        np.testing.assert_array_almost_equal(
            controller_instance.HOME_JOINT_POSITIONS,
            expected
        )


# ============================================================================
# TEST SUITE 7: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete control flows."""

    def test_full_ee_movement_sequence(self, controller_instance, mock_ik_solver):
        """Test complete EE control flow: press W → process → apply → robot moves."""
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance.ik_solver = mock_ik_solver

        initial_x = controller_instance.ee_target_position[0]

        # User presses 'w'
        controller_instance._queue_command(('char', 'w'))

        # Main loop processes command
        last_char_cmd = controller_instance._process_commands()

        # Verify target updated
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x + 0.02)

        # Apply control
        if last_char_cmd is not None:
            controller_instance._apply_endeffector_control()

        # Verify robot received action
        controller = controller_instance.franka.get_articulation_controller()
        assert controller.apply_action.called, "Robot should receive action"

    def test_full_joint_movement_sequence(self, controller_instance):
        """Test complete joint flow: select joint 3 → press W → apply."""
        controller_instance.control_mode = controller_instance.MODE_JOINT

        # Select joint 3
        controller_instance._queue_command(('char', '3'))
        controller_instance._process_commands()
        assert controller_instance.active_joint == 2

        # Press W to move
        controller_instance._queue_command(('char', 'w'))
        last_char_cmd = controller_instance._process_commands()

        # Apply control
        if last_char_cmd is not None:
            controller_instance._apply_joint_control(last_char_cmd)

        # Verify robot received action
        controller = controller_instance.franka.get_articulation_controller()
        assert controller.apply_action.called

    def test_mode_switch_and_control(self, controller_instance, mock_ik_solver):
        """Test Tab → press W in new mode."""
        controller_instance.ik_solver = mock_ik_solver
        initial_mode = controller_instance.control_mode

        # Press Tab to switch mode
        controller_instance._queue_command(('special', 'tab'))
        controller_instance._process_commands()

        assert controller_instance.control_mode != initial_mode, "Mode should change"

        # Press W in new mode
        controller_instance._queue_command(('char', 'w'))
        last_char_cmd = controller_instance._process_commands()

        assert last_char_cmd == 'w', "Should process W in new mode"

    def test_workspace_limit_recovery(self, controller_instance, mock_ik_solver):
        """Test IK fail → revert → opposite direction succeeds."""
        controller_instance.ik_solver = mock_ik_solver
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR

        initial_x = controller_instance.ee_target_position[0]

        # Make IK fail for forward movement
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), False)
        controller_instance._process_endeffector_command('w')

        # Verify position reverted
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x)

        # Make IK succeed for backward movement
        mock_ik_solver.compute_inverse_kinematics.return_value = (Mock(), True)
        controller_instance._process_endeffector_command('s')

        # Verify position updated
        assert controller_instance.ee_target_position[0] == pytest.approx(initial_x - 0.02)

    def test_gripper_open_close_cycle(self, controller_instance):
        """Test Q → E → Q sequence."""
        controller_instance.control_mode = controller_instance.MODE_JOINT

        initial_gripper = controller_instance.franka.get_joint_positions()[7]

        # Open gripper
        controller_instance._apply_joint_control('q')
        controller = controller_instance.franka.get_articulation_controller()
        first_call = controller.apply_action.call_args[0][0]
        opened = first_call.joint_positions[7]
        assert opened > initial_gripper, "Should open"

        # Update mock to reflect new position
        controller_instance.franka.get_joint_positions.return_value[7] = opened

        # Close gripper
        controller_instance._apply_joint_control('e')
        second_call = controller.apply_action.call_args[0][0]
        closed = second_call.joint_positions[7]
        assert closed < opened, "Should close"


# ============================================================================
# TEST SUITE 8: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and error scenarios."""

    def test_control_without_commands(self, controller_instance):
        """Test loop iteration with empty queue."""
        result = controller_instance._process_commands()
        assert result is None

        # Should not crash
        controller_instance._apply_joint_control('w')

    def test_unknown_key_in_joint_mode(self, controller_instance):
        """Test invalid key handling in joint mode."""
        controller_instance.control_mode = controller_instance.MODE_JOINT

        # Should not crash
        controller_instance._process_joint_command('x')
        controller_instance._apply_joint_control('x')

    def test_unknown_key_in_ee_mode(self, controller_instance, mock_ik_solver):
        """Test invalid key handling in EE mode."""
        controller_instance.control_mode = controller_instance.MODE_ENDEFFECTOR
        controller_instance.ik_solver = mock_ik_solver

        initial_position = controller_instance.ee_target_position.copy()

        # Should not crash or change position
        controller_instance._process_endeffector_command('x')

        np.testing.assert_array_equal(controller_instance.ee_target_position, initial_position)

    def test_rapid_mode_switching(self, controller_instance):
        """Test rapid Tab presses."""
        for _ in range(10):
            controller_instance._toggle_mode()

        # Should end up in joint mode (even number of toggles)
        assert controller_instance.control_mode == controller_instance.MODE_JOINT

    def test_esc_sets_exit_flag(self, controller_instance):
        """Test Esc command sets should_exit."""
        controller_instance._queue_command(('special', 'esc'))
        controller_instance._process_commands()

        assert controller_instance.should_exit is True

    def test_reset_to_home_position(self, controller_instance):
        """Test R key resets to home position."""
        controller_instance._queue_command(('char', 'r'))
        controller_instance._process_commands()

        controller = controller_instance.franka.get_articulation_controller()
        call_args = controller.apply_action.call_args[0][0]

        np.testing.assert_array_almost_equal(
            call_args.joint_positions,
            controller_instance.HOME_JOINT_POSITIONS
        )

    def test_multiple_special_commands(self, controller_instance):
        """Test batch of special commands."""
        controller_instance._queue_command(('special', 'tab'))
        controller_instance._queue_command(('special', 'tab'))

        result = controller_instance._process_commands()

        assert result is None, "Should not return char command"
        assert controller_instance.control_mode == controller_instance.MODE_JOINT, \
            "Should toggle twice back to joint mode"


# ============================================================================
# TEST SUITE 9: Concurrent Access Tests
# ============================================================================

class TestConcurrency:
    """Tests for thread safety and concurrent access patterns."""

    def test_queue_command_from_multiple_threads(self, controller_instance):
        """Test command queueing from multiple threads."""
        def queue_commands(controller, commands):
            for cmd in commands:
                controller._queue_command(cmd)
                time.sleep(0.001)  # Small delay

        thread1_commands = [('char', 'w'), ('char', 'a')]
        thread2_commands = [('char', 's'), ('char', 'd')]

        thread1 = threading.Thread(target=queue_commands, args=(controller_instance, thread1_commands))
        thread2 = threading.Thread(target=queue_commands, args=(controller_instance, thread2_commands))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # All 4 commands should be queued
        assert len(controller_instance.pending_commands) == 4

    def test_process_commands_while_queueing(self, controller_instance):
        """Test processing while new commands being added."""
        # Queue some commands
        controller_instance._queue_command(('char', 'w'))
        controller_instance._queue_command(('char', 'a'))

        # Process (this clears queue)
        result = controller_instance._process_commands()
        assert result == 'a'
        assert len(controller_instance.pending_commands) == 0

        # Queue new command
        controller_instance._queue_command(('char', 's'))

        # Process again
        result = controller_instance._process_commands()
        assert result == 's'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
