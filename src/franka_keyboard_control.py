"""
Minimal Franka Arm Robot Control Example with PyInput Keyboard Control and Rich TUI

This example demonstrates:
- Loading and controlling a Franka Panda robot
- Dual control modes: Joint control and End-effector control (IK)
- PyInput keyboard integration for external keyboard input
- Rich TUI with live-updating interface and visual button feedback
- Incremental position updates based on key presses

Controls:
    Joint Control Mode:
        1-7: Select joint to control
        W/S: Increase/decrease selected joint angle
        Q/E: Open/close gripper
        R: Reset to home position

    End-Effector Control Mode:
        W/S: Move forward/backward (X-axis)
        A/D: Move left/right (Y-axis)
        Q/E: Move up/down (Z-axis)
        I/K: Pitch rotation (up/down)
        J/L: Yaw rotation (left/right)
        U/O: Roll rotation

    Both Modes:
        Tab: Switch between joint/end-effector control
        Esc: Exit application
"""

from isaacsim import SimulationApp

# SimulationApp will be created in FrankaKeyboardController.__init__()
# This allows tests to mock it before creation
simulation_app = None

import argparse
import numpy as np
import threading
import sys
import termios
import tty
from pynput import keyboard

# Isaac Sim imports must happen AFTER SimulationApp is created
# They will be imported inside __init__ after app initialization
World = None
ArticulationAction = None
Franka = None
KinematicsSolver = None
euler_angles_to_quat = None
quat_to_euler_angles = None

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TUIRenderer:  # pragma: no cover
    """Renders the terminal user interface for robot control using Rich library."""

    def __init__(self):  # pragma: no cover
        """Initialize the TUI renderer."""
        self.pressed_keys = set()
        self.control_mode = 0  # MODE_JOINT
        self.active_joint = 0

        # Telemetry data
        self.position = np.array([0.0, 0.0, 0.0])
        self.euler = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.gripper = 0.04
        self.joint_positions = np.zeros(7)  # 7 arm joint positions
        self.joint_limits = {}  # Will be set by controller
        self.ik_valid = True
        self.ik_message = ""
        self.last_command = "Ready"

        # Mode constants
        self.MODE_JOINT = 0
        self.MODE_ENDEFFECTOR = 1

        # Recording status
        self.recording_active = False
        self.recording_stats = {}
        self.checkpoint_flash_active = False
        self.recording_enabled = False

    def set_pressed_key(self, key):  # pragma: no cover
        """Mark a key as pressed for highlighting."""
        self.pressed_keys.add(key)

    def clear_pressed_key(self, key):  # pragma: no cover
        """Mark a key as released."""
        self.pressed_keys.discard(key)

    def update_telemetry(self, position, euler, gripper, joint_positions):  # pragma: no cover
        """Update telemetry values."""
        self.position = position
        self.euler = euler
        self.gripper = gripper
        self.joint_positions = joint_positions

    def set_mode(self, mode):  # pragma: no cover
        """Set the control mode."""
        self.control_mode = mode

    def set_active_joint(self, joint_index):
        """Set the active joint index."""
        self.active_joint = joint_index

    def set_ik_status(self, valid, message=""):
        """Update IK validation status."""
        self.ik_valid = valid
        self.ik_message = message

    def set_last_command(self, command):
        """Set the last executed command."""
        self.last_command = command

    def set_recording_status(self, is_recording: bool, stats: dict):
        """Update recording status for display.

        Args:
            is_recording: Whether recording is currently active
            stats: Dictionary with recording statistics
        """
        self.recording_active = is_recording
        self.recording_stats = stats.copy() if stats else {}

    def set_checkpoint_flash(self, active: bool):
        """Set checkpoint flash indicator state.

        Args:
            active: Whether to show the "SAVED" flash indicator
        """
        self.checkpoint_flash_active = active

    def set_recording_enabled(self, enabled: bool):
        """Set whether recording mode is enabled.

        Args:
            enabled: Whether recording is enabled
        """
        self.recording_enabled = enabled

    def _render_recording_panel(self) -> Panel:
        """Render recording status panel with controls.

        Returns:
            Rich Panel containing recording status display
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")

        # Recording status indicator (top of panel)
        if self.recording_active:
            status_text = Text("● REC", style="bold red blink")
        else:
            status_text = Text("○ IDLE", style="dim")

        # Add checkpoint flash indicator
        if self.checkpoint_flash_active:
            status_row = Text.assemble(status_text, " ", Text("SAVED", style="bold green reverse"))
        else:
            status_row = status_text

        table.add_row(Text("STATUS:", style="bold underline"), status_row)
        table.add_row("", "")

        # Episode stats
        total_frames = self.recording_stats.get('total_frames', 0)
        current_frames = self.recording_stats.get('current_episode_frames', 0)
        num_episodes = self.recording_stats.get('num_episodes', 0)
        num_success = self.recording_stats.get('num_success', 0)
        num_failed = self.recording_stats.get('num_failed', 0)
        current_return = self.recording_stats.get('current_episode_return', 0.0)

        table.add_row(Text("EPISODES:", style="bold underline"), "")
        table.add_row("Total:", f"{num_episodes}")
        table.add_row("Success:", Text(f"{num_success}", style="green"))
        table.add_row("Failed:", Text(f"{num_failed}", style="red"))
        table.add_row("", "")

        table.add_row(Text("FRAMES:", style="bold underline"), "")
        table.add_row("Total:", f"{total_frames}")
        table.add_row("Current Ep:", f"{current_frames}")
        table.add_row("Return:", f"{current_return:.2f}")
        table.add_row("", "")

        # Recording controls with button feedback
        table.add_row(Text("CONTROLS:", style="bold underline"), "")

        # Start/Stop button with pressed state
        backtick_btn = self._create_button("`", "`", width=4)
        table.add_row(backtick_btn, Text("Start/Stop", style="dim"))

        # Success button
        left_bracket_btn = self._create_button("[", "[", width=4)
        table.add_row(left_bracket_btn, Text("Mark Success", style="green dim"))

        # Failure button
        right_bracket_btn = self._create_button("]", "]", width=4)
        table.add_row(right_bracket_btn, Text("Mark Failed", style="red dim"))

        # Auto-save indicator (replaces manual save)
        table.add_row("", "")
        table.add_row(Text("AUTO-SAVE:", style="dim italic"), Text("Every 5s", style="dim italic"))

        border_style = "red" if self.recording_active else "dim"
        return Panel(table, title="Recording", border_style=border_style)

    def _create_button(self, label, key, width=6):
        """Create a button with optional highlighting.

        Args:
            label: Display text
            key: Key identifier for press detection
            width: Button width in characters
        """
        is_pressed = key in self.pressed_keys
        text = f" {label:^{width-2}} "

        if is_pressed:
            return Text(text, style="reverse bold")
        else:
            return Text(text, style="dim")

    def _render_endeffector_controls(self):
        """Render end-effector control panel."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        # Title
        title = Text("POSITION CONTROL", style="bold cyan")
        table.add_row(Text("", style=""))
        table.add_row(title, "", "", "")
        table.add_row(Text("", style=""))

        # Q - Z+
        table.add_row("", self._create_button("Q ↑", "q"), "", "")
        table.add_row("", Text("Z+", style="dim italic"), "", "")
        table.add_row(Text("", style=""))

        # WASD grid
        table.add_row(
            self._create_button("W ↑", "w"),
            self._create_button("A ←", "a"),
            self._create_button("S ↓", "s"),
            self._create_button("D →", "d")
        )
        table.add_row(
            Text("X+", style="dim italic"),
            Text("Y+", style="dim italic"),
            Text("X-", style="dim italic"),
            Text("Y-", style="dim italic")
        )
        table.add_row(Text("", style=""))

        # E - Z-
        table.add_row("", self._create_button("E ↓", "e"), "", "")
        table.add_row("", Text("Z-", style="dim italic"), "", "")

        # Separator
        table.add_row(Text("", style=""))
        table.add_row(Text("─" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Rotation title
        rot_title = Text("ROTATION CONTROL", style="bold magenta")
        table.add_row(rot_title, "", "", "")
        table.add_row(Text("", style=""))

        # U - Roll+
        table.add_row("", self._create_button("U ⟲", "u"), "", "")
        table.add_row("", Text("Roll+", style="dim italic"), "", "")
        table.add_row(Text("", style=""))

        # IJKL grid
        table.add_row(
            self._create_button("I ↑", "i"),
            self._create_button("J ⟲", "j"),
            self._create_button("K ↓", "k"),
            self._create_button("L ⟲", "l")
        )
        table.add_row(
            Text("Pitch+", style="dim italic"),
            Text("Yaw+", style="dim italic"),
            Text("Pitch-", style="dim italic"),
            Text("Yaw-", style="dim italic")
        )
        table.add_row(Text("", style=""))

        # O - Roll-
        table.add_row("", self._create_button("O ⟲", "o"), "", "")
        table.add_row("", Text("Roll-", style="dim italic"), "", "")

        # Separator
        table.add_row(Text("", style=""))
        table.add_row(Text("─" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Secondary controls
        table.add_row(
            Text("[Tab]", style="yellow"),
            Text("Switch Mode", style="dim"),
            "", ""
        )
        table.add_row(
            Text("[R]", style="yellow"),
            Text("Reset", style="dim"),
            Text("[Esc]", style="yellow"),
            Text("Exit", style="dim")
        )
        table.add_row(
            self._create_button("C", "c"),
            Text("Spawn Cube", style="dim"),
            "", ""
        )

        return Panel(table, title="Controls", border_style="blue")

    def _render_joint_controls(self):
        """Render joint control panel."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        # Title
        title = Text("JOINT CONTROL", style="bold cyan")
        table.add_row(Text("", style=""))
        table.add_row(title, "", "", "")
        table.add_row(Text("", style=""))

        # Joint selection
        sel_text = Text("SELECT JOINT:", style="bold")
        table.add_row(sel_text, "", "", "")
        table.add_row(Text("", style=""))

        # Joints 1-4
        table.add_row(
            self._create_button("1", "1", width=5),
            self._create_button("2", "2", width=5),
            self._create_button("3", "3", width=5),
            self._create_button("4", "4", width=5)
        )
        table.add_row(Text("", style=""))

        # Joints 5-7
        table.add_row(
            self._create_button("5", "5", width=5),
            self._create_button("6", "6", width=5),
            self._create_button("7", "7", width=5),
            ""
        )
        table.add_row(Text("", style=""))
        table.add_row(Text("", style=""))

        # Active joint indicator
        active_text = Text(f"Active: Joint {self.active_joint + 1}", style="bold green")
        table.add_row(active_text, "", "", "")
        table.add_row(Text("", style=""))
        table.add_row(Text("─" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Movement controls
        move_title = Text("MOVE JOINT:", style="bold")
        table.add_row(move_title, "", "", "")
        table.add_row(Text("", style=""))

        table.add_row(
            self._create_button("W ↑", "w"),
            Text("Increase", style="dim italic"),
            "", ""
        )
        table.add_row(
            self._create_button("S ↓", "s"),
            Text("Decrease", style="dim italic"),
            "", ""
        )

        table.add_row(Text("", style=""))
        table.add_row(Text("─" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Gripper controls
        grip_title = Text("GRIPPER:", style="bold")
        table.add_row(grip_title, "", "", "")
        table.add_row(Text("", style=""))

        table.add_row(
            self._create_button("Q", "q"),
            Text("Open", style="dim italic"),
            "", ""
        )
        table.add_row(
            self._create_button("E", "e"),
            Text("Close", style="dim italic"),
            "", ""
        )

        # Separator
        table.add_row(Text("", style=""))
        table.add_row(Text("─" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Secondary controls
        table.add_row(
            Text("[Tab]", style="yellow"),
            Text("Switch Mode", style="dim"),
            "", ""
        )
        table.add_row(
            Text("[R]", style="yellow"),
            Text("Reset", style="dim"),
            Text("[Esc]", style="yellow"),
            Text("Exit", style="dim")
        )
        table.add_row(
            self._create_button("C", "c"),
            Text("Spawn Cube", style="dim"),
            "", ""
        )

        return Panel(table, title="Controls", border_style="blue")

    def _render_state_panel(self):
        """Render robot state telemetry panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column(justify="left")

        # Position data
        table.add_row(Text("POSITION (m)", style="bold underline"), "")
        table.add_row("X:", f"{self.position[0]:.3f}")
        table.add_row("Y:", f"{self.position[1]:.3f}")
        table.add_row("Z:", f"{self.position[2]:.3f}")
        table.add_row("", "")

        # Orientation data (convert to degrees)
        roll_deg = np.degrees(self.euler[0])
        pitch_deg = np.degrees(self.euler[1])
        yaw_deg = np.degrees(self.euler[2])

        table.add_row(Text("ORIENTATION (deg)", style="bold underline"), "")
        table.add_row("Roll:", f"{roll_deg:>7.1f}°")
        table.add_row("Pitch:", f"{pitch_deg:>7.1f}°")
        table.add_row("Yaw:", f"{yaw_deg:>7.1f}°")
        table.add_row("", "")

        # Gripper state
        table.add_row(Text("GRIPPER", style="bold underline"), "")
        table.add_row("Opening:", f"{self.gripper:.4f} m")
        table.add_row("", "")

        # Joint positions (convert to degrees)
        table.add_row(Text("JOINT POSITIONS (deg)", style="bold underline"), "")

        # Format joint display strings for compact table view
        joint_displays = []
        for i in range(7):
            # Convert joint position to degrees
            joint_deg = np.degrees(self.joint_positions[i])

            # Get joint limits in degrees
            if i in self.joint_limits:
                min_rad, max_rad = self.joint_limits[i]
                min_deg = np.degrees(min_rad)
                max_deg = np.degrees(max_rad)
                limit_str = f"[{min_deg:>4.0f},{max_deg:>4.0f}]"
            else:
                limit_str = ""

            # Format the joint display
            joint_str = f"J{i+1}: {joint_deg:>6.2f}° {limit_str}"

            # Highlight active joint in Joint mode
            if self.control_mode == self.MODE_JOINT and i == self.active_joint:
                joint_displays.append(Text(joint_str, style="bold green"))
            else:
                joint_displays.append(joint_str)

        # Display in compact 2-row format: J1-J4 on first row, J5-J7 on second row
        row1 = Text.assemble("  ", joint_displays[0], "  ", joint_displays[1],
                            "  ", joint_displays[2], "  ", joint_displays[3])
        row2 = Text.assemble("  ", joint_displays[4], "  ", joint_displays[5],
                            "  ", joint_displays[6])
        table.add_row("", row1)
        table.add_row("", row2)
        table.add_row("", "")

        # IK status (only in end-effector mode)
        if self.control_mode == self.MODE_ENDEFFECTOR:
            status_style = "bold green" if self.ik_valid else "bold red"
            status_text = "✓ Valid" if self.ik_valid else "✗ Invalid"
            table.add_row(Text("IK STATUS", style="bold underline"), "")
            table.add_row("", Text(status_text, style=status_style))
            if self.ik_message:
                table.add_row("", Text(self.ik_message, style="italic"))
            table.add_row("", "")

        # Last command
        table.add_row(Text("LAST COMMAND", style="bold underline"), "")
        table.add_row("", Text(self.last_command, style="italic"))

        return Panel(table, title="Robot State", border_style="green")

    def render(self):
        """Render the complete TUI layout."""
        # Create main layout
        layout = Layout()

        # Mode header
        mode_name = "END-EFFECTOR CONTROL" if self.control_mode == self.MODE_ENDEFFECTOR else "JOINT CONTROL"
        mode_text = Text(f"MODE: {mode_name}", style="bold white on blue", justify="center")

        # Split into rows: header + content
        layout.split_column(
            Layout(Panel(mode_text, border_style="bright_blue"), size=3, name="header"),
            Layout(name="content")
        )

        # Split content based on whether recording is enabled
        if self.recording_enabled:
            layout["content"].split_row(
                Layout(name="controls", ratio=35),
                Layout(name="recording", ratio=25),
                Layout(name="state", ratio=40)
            )
            layout["recording"].update(self._render_recording_panel())
        else:
            layout["content"].split_row(
                Layout(name="controls", ratio=40),
                Layout(name="state", ratio=60)
            )

        # Render appropriate controls based on mode
        if self.control_mode == self.MODE_ENDEFFECTOR:
            layout["controls"].update(self._render_endeffector_controls())
        else:
            layout["controls"].update(self._render_joint_controls())

        # Render state panel
        layout["state"].update(self._render_state_panel())

        return layout


class SceneManager:
    """Manages scene objects for pick-and-place tasks.

    This class handles spawning and managing objects in the Isaac Sim scene,
    including cubes for manipulation tasks and goal markers for target locations.

    Attributes:
        world: The Isaac Sim World object
        cube: The spawned cube object (or None)
        goal_marker: The visual goal marker (or None)
        goal_position: Target position as numpy array (or None)
        cube_size: Size of the cube in meters
        workspace_bounds: Dict defining the workspace boundaries
    """

    DEFAULT_WORKSPACE_BOUNDS = {
        'x': [0.3, 0.6],
        'y': [-0.3, 0.3],
        'z': [0.02, 0.20]
    }

    DEFAULT_CUBE_SIZE = 0.04  # meters

    def __init__(self, world, workspace_bounds: dict = None):
        """Initialize the SceneManager.

        Args:
            world: The Isaac Sim World object
            workspace_bounds: Optional dict with 'x', 'y', 'z' keys, each containing
                             [min, max] bounds for random position generation
        """
        self.world = world
        self.cube = None
        self.goal_marker = None
        self.goal_position = None
        self.cube_size = self.DEFAULT_CUBE_SIZE
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS.copy()

        # Track spawned random cubes
        self.random_cube_counter = 0
        self.random_cubes = []

    def spawn_cube(self, position: list = None, size: float = 0.04, color: tuple = (1, 0, 0)):
        """Spawn a graspable cube in the scene.

        Args:
            position: [x, y, z] position. If None, uses random position.
            size: Cube side length in meters (default 0.04)
            color: RGB tuple (default red)

        Returns:
            The spawned cube object
        """
        if position is None:
            position = self._random_position()

        self.cube_size = size
        pos_array = np.array(position)

        # Try to use real Isaac Sim primitives
        try:
            from isaacsim.core.api.objects import DynamicCuboid
            self.cube = self.world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Cube",
                    name="target_cube",
                    position=pos_array,
                    scale=np.array([size, size, size]),
                    color=np.array(color[:3]),
                    density=1000.0  # Ensure proper mass for gravity
                )
            )
        except ImportError:
            # Fall back to mock for testing
            cube_mock = type('Cube', (), {
                'name': 'target_cube',
                'position': pos_array,
                'size': size,
                'color': color,
                'get_world_pose': lambda: (pos_array.copy(), np.array([0, 0, 0, 1])),
                'set_world_pose': lambda position, orientation=None: None
            })()
            self.cube = self.world.scene.add(cube_mock)

        return self.cube

    def spawn_goal_marker(self, position: list = None, color: tuple = (0, 1, 0, 0.5)) -> np.ndarray:
        """Spawn a visual goal marker in the scene.

        Args:
            position: [x, y, z] position. If None, uses random position.
            color: RGBA tuple (default semi-transparent green)

        Returns:
            The goal position as numpy array
        """
        if position is None:
            position = self._random_position()

        self.goal_position = np.array(position)

        # Try to use real Isaac Sim primitives
        try:
            from isaacsim.core.api.objects import VisualCuboid
            self.goal_marker = self.world.scene.add(
                VisualCuboid(
                    prim_path="/World/GoalMarker",
                    name="goal_marker",
                    position=self.goal_position,
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=np.array(color[:3])
                )
            )
        except ImportError:
            # Fall back to mock for testing
            marker_mock = type('GoalMarker', (), {
                'name': 'goal_marker',
                'position': self.goal_position,
                'color': color
            })()
            self.goal_marker = self.world.scene.add(marker_mock)

        return self.goal_position

    def spawn_random_cube(self, size: float = 0.04, color: tuple = (0.5, 0.5, 1.0)):
        """Spawn a random cube at 2m height that falls to the table.

        This spawns cubes with unique names for unlimited spawning. Each cube
        falls naturally with physics enabled.

        Args:
            size: Cube side length in meters (default 0.04)
            color: RGB tuple (default light blue: 0.5, 0.5, 1.0)

        Returns:
            The spawned cube object
        """
        # Generate random X, Y position within workspace bounds, Z=2.0m fixed
        x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
        y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])
        z = 2.0  # Fixed height for falling

        position = np.array([x, y, z])

        # Increment counter for unique naming
        self.random_cube_counter += 1
        prim_path = f"/World/RandomCube_{self.random_cube_counter:03d}"
        cube_name = f"random_cube_{self.random_cube_counter:03d}"

        # Try to use real Isaac Sim primitives
        try:
            from isaacsim.core.api.objects import DynamicCuboid
            cube = self.world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path,
                    name=cube_name,
                    position=position,
                    scale=np.array([size, size, size]),
                    color=np.array(color[:3]),
                    density=1000.0  # Same density as task cube for proper physics
                )
            )
        except ImportError:
            # Fall back to mock for testing
            cube_mock = type('RandomCube', (), {
                'name': cube_name,
                'position': position,
                'size': size,
                'color': color,
                'get_world_pose': lambda: (position.copy(), np.array([0, 0, 0, 1])),
                'set_world_pose': lambda position, orientation=None: None
            })()
            cube = self.world.scene.add(cube_mock)

        self.random_cubes.append(cube)
        return cube

    def get_cube_pose(self) -> tuple:
        """Get the current pose of the cube.

        Returns:
            Tuple of (position, orientation) as numpy arrays, or (None, None) if no cube
        """
        if self.cube is None:
            return None, None
        return self.cube.get_world_pose()

    def get_goal_position(self) -> np.ndarray:
        """Get the goal position.

        Returns:
            Goal position as numpy array, or None if not set
        """
        return self.goal_position

    def reset_scene(self) -> tuple:
        """Reset the scene with new random positions for cube and goal.

        Returns:
            Tuple of (cube_position, goal_position) as lists
        """
        cube_pos = self._random_position()
        goal_pos = self._random_position()

        # Ensure goal is sufficiently far from cube
        min_separation = 0.1  # meters
        while np.linalg.norm(np.array(cube_pos) - np.array(goal_pos)) < min_separation:
            goal_pos = self._random_position()

        if self.cube is not None:
            self.cube.set_world_pose(position=np.array(cube_pos))
            # Update the mock's internal state
            self.cube.position = np.array(cube_pos)
            new_pos = np.array(cube_pos)
            self.cube.get_world_pose = lambda: (new_pos, np.array([0, 0, 0, 1]))

        self.goal_position = np.array(goal_pos)

        return cube_pos, goal_pos

    def check_grasp(self, ee_pos: np.ndarray, gripper_width: float,
                    cube_size: float = None, grasp_threshold: float = 0.05) -> bool:
        """Check if the cube is grasped by the gripper.

        Grasp is detected when the end-effector is close to the cube AND
        the gripper is closed enough to hold the cube.

        Args:
            ee_pos: End-effector position [x, y, z]
            gripper_width: Current gripper opening width in meters
            cube_size: Size of the cube (uses stored size if None)
            grasp_threshold: Maximum distance to consider grasped

        Returns:
            True if cube is grasped, False otherwise
        """
        if self.cube is None:
            return False

        if cube_size is None:
            cube_size = self.cube_size

        cube_pos, _ = self.cube.get_world_pose()
        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))

        is_close = distance < grasp_threshold
        is_gripper_closed = gripper_width < cube_size

        return is_close and is_gripper_closed

    def check_task_complete(self, threshold: float = 0.03) -> bool:
        """Check if the cube has been placed at the goal.

        Args:
            threshold: Maximum distance to consider task complete

        Returns:
            True if cube is at goal position, False otherwise
        """
        if self.cube is None or self.goal_position is None:
            return False

        cube_pos, _ = self.cube.get_world_pose()
        distance = np.linalg.norm(np.array(cube_pos) - np.array(self.goal_position))

        return distance < threshold

    def _random_position(self, z_range: list = None) -> list:
        """Generate a random position within workspace bounds.

        Args:
            z_range: Optional [min, max] for z. Uses workspace_bounds['z'] if None.

        Returns:
            [x, y, z] position list
        """
        x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
        y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])

        z_bounds = z_range or self.workspace_bounds['z']
        z = np.random.uniform(z_bounds[0], z_bounds[1])

        return [x, y, z]


class DemoRecorder:
    """Records demonstrations for imitation learning and RL training.

    This class captures state-action trajectories during teleoperation,
    with support for episode segmentation and success/failure labeling.

    Attributes:
        obs_dim: Dimension of observation vectors
        action_dim: Dimension of action vectors
        observations: List of recorded observations
        actions: List of recorded actions
        rewards: List of recorded rewards
        dones: List of done flags
        is_recording: Whether recording is active
    """

    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize the DemoRecorder.

        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Data buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        # Episode tracking
        self.episode_starts = []
        self.episode_lengths = []
        self.episode_returns = []
        self.episode_success = []

        # Recording state
        self.is_recording = False
        self.current_episode_start = 0
        self.current_episode_return = 0.0
        self._pending_success = None

    def start_recording(self) -> None:
        """Start recording a new episode."""
        self.is_recording = True
        self.current_episode_start = len(self.observations)
        self.current_episode_return = 0.0
        self._pending_success = None

    def stop_recording(self) -> None:
        """Stop recording."""
        self.is_recording = False

    def record_step(self, obs: np.ndarray, action: np.ndarray,
                    reward: float, done: bool) -> None:
        """Record a single timestep.

        Args:
            obs: Observation vector
            action: Action vector
            reward: Reward value
            done: Whether episode is done
        """
        if not self.is_recording:
            return

        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.current_episode_return += reward

    def mark_episode_success(self, success: bool) -> None:
        """Mark the current episode as successful or failed.

        Args:
            success: True if episode was successful
        """
        self._pending_success = success

    def finalize_episode(self) -> None:
        """Finalize the current episode and record its metadata."""
        episode_length = len(self.observations) - self.current_episode_start

        if episode_length > 0:
            self.episode_starts.append(self.current_episode_start)
            self.episode_lengths.append(episode_length)
            self.episode_returns.append(self.current_episode_return)

            # Default to False if not explicitly marked
            success = self._pending_success if self._pending_success is not None else False
            self.episode_success.append(success)

        # Reset for next episode
        self.current_episode_start = len(self.observations)
        self.current_episode_return = 0.0
        self._pending_success = None

    def get_stats(self) -> dict:
        """Get current recording statistics.

        Returns:
            Dictionary with recording statistics
        """
        num_success = sum(1 for s in self.episode_success if s)
        num_failed = len(self.episode_success) - num_success

        return {
            'is_recording': self.is_recording,
            'total_frames': len(self.observations),
            'current_episode_frames': len(self.observations) - self.current_episode_start,
            'num_episodes': len(self.episode_starts),
            'num_success': num_success,
            'num_failed': num_failed,
        }

    def clear(self) -> None:
        """Clear all recorded data and reset state."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.episode_starts = []
        self.episode_lengths = []
        self.episode_returns = []
        self.episode_success = []
        self.is_recording = False
        self.current_episode_start = 0
        self.current_episode_return = 0.0
        self._pending_success = None

    def save(self, filepath: str, metadata: dict = None) -> None:
        """Save demonstrations to NPZ file.

        Args:
            filepath: Path to save the .npz file
            metadata: Optional dictionary of additional metadata
        """
        # Auto-finalize any pending episode data (user may have forgotten [ or ])
        pending_frames = len(self.observations) - self.current_episode_start
        if pending_frames > 0:
            # Mark as failed by default if not explicitly marked
            if self._pending_success is None:
                self._pending_success = False
            self.finalize_episode()

        # Build metadata
        save_metadata = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'num_episodes': len(self.episode_starts),
            'total_frames': len(self.observations),
        }
        if metadata:
            save_metadata.update(metadata)

        # Convert lists to arrays, preserving dtypes
        if self.observations:
            obs_array = np.array(self.observations)
            action_array = np.array(self.actions)
        else:
            obs_array = np.array([]).reshape(0, self.obs_dim)
            action_array = np.array([]).reshape(0, self.action_dim)

        np.savez_compressed(
            filepath,
            observations=obs_array,
            actions=action_array,
            rewards=np.array(self.rewards, dtype=np.float32),
            dones=np.array(self.dones, dtype=bool),
            episode_starts=np.array(self.episode_starts, dtype=np.int64),
            episode_lengths=np.array(self.episode_lengths, dtype=np.int64),
            episode_returns=np.array(self.episode_returns, dtype=np.float32),
            episode_success=np.array(self.episode_success, dtype=bool),
            metadata=np.array(save_metadata, dtype=object),
        )

    @classmethod
    def load(cls, filepath: str) -> 'DemoRecorder':
        """Load demonstrations from NPZ file.

        Args:
            filepath: Path to the .npz file

        Returns:
            DemoRecorder instance with loaded data
        """
        data = np.load(filepath, allow_pickle=True)
        metadata = data['metadata'].item()

        # Create new recorder with correct dimensions
        recorder = cls(
            obs_dim=metadata['obs_dim'],
            action_dim=metadata['action_dim']
        )

        # Load data into lists
        recorder.observations = list(data['observations'])
        recorder.actions = list(data['actions'])
        recorder.rewards = list(data['rewards'])
        recorder.dones = list(data['dones'])
        recorder.episode_starts = list(data['episode_starts'])
        recorder.episode_lengths = list(data['episode_lengths'])
        recorder.episode_returns = list(data['episode_returns'])
        recorder.episode_success = list(data['episode_success'])

        # Update current position
        recorder.current_episode_start = len(recorder.observations)

        return recorder


class ActionMapper:
    """Maps keyboard commands to normalized action space.

    Action space layout (7D):
        [0]: delta_x   - Forward/backward movement
        [1]: delta_y   - Left/right movement
        [2]: delta_z   - Up/down movement
        [3]: delta_roll  - Roll rotation
        [4]: delta_pitch - Pitch rotation
        [5]: delta_yaw   - Yaw rotation
        [6]: gripper   - Gripper open/close command
    """

    # Key to action mapping
    # Each key maps to a 7D action vector
    KEY_TO_ACTION = {
        # Position controls
        'w': [1, 0, 0, 0, 0, 0, 0],    # +X (forward)
        's': [-1, 0, 0, 0, 0, 0, 0],   # -X (backward)
        'a': [0, 1, 0, 0, 0, 0, 0],    # +Y (left)
        'd': [0, -1, 0, 0, 0, 0, 0],   # -Y (right)
        'q': [0, 0, 1, 0, 0, 0, 0],    # +Z (up)
        'e': [0, 0, -1, 0, 0, 0, 0],   # -Z (down)
        # Rotation controls
        'u': [0, 0, 0, 1, 0, 0, 0],    # +roll
        'o': [0, 0, 0, -1, 0, 0, 0],   # -roll
        'i': [0, 0, 0, 0, 1, 0, 0],    # +pitch
        'k': [0, 0, 0, 0, -1, 0, 0],   # -pitch
        'j': [0, 0, 0, 0, 0, 1, 0],    # +yaw
        'l': [0, 0, 0, 0, 0, -1, 0],   # -yaw
        # Gripper controls (dedicated keys)
        'g': [0, 0, 0, 0, 0, 0, -1],   # close gripper
        'h': [0, 0, 0, 0, 0, 0, 1],    # open gripper
    }

    def __init__(self):
        """Initialize the ActionMapper."""
        self.action_dim = 7

    def map_key(self, key: str) -> np.ndarray:
        """Map a keyboard key to an action vector.

        Args:
            key: The key character (e.g., 'w', 'a', 's', 'd')

        Returns:
            7D action vector as float32 numpy array
        """
        if key is None or key not in self.KEY_TO_ACTION:
            return np.zeros(self.action_dim, dtype=np.float32)

        return np.array(self.KEY_TO_ACTION[key], dtype=np.float32)


class ObservationBuilder:
    """Builds observation vectors from robot state.

    Observation layout (23D):
        [0:7]   - joint positions (7)
        [7:10]  - ee position (3)
        [10:14] - ee orientation quaternion (4)
        [14:15] - gripper width (1)
        [15:18] - cube position (3)
        [18:21] - goal position (3)
        [21:22] - cube grasped flag (1)
        [22:23] - distance to cube (1)
    """

    def __init__(self):
        """Initialize the ObservationBuilder."""
        self.obs_dim = 23

    def build(self, joint_positions: np.ndarray, ee_position: np.ndarray,
              ee_orientation: np.ndarray, gripper_width: float,
              cube_position: np.ndarray, goal_position: np.ndarray,
              cube_grasped: bool) -> np.ndarray:
        """Build an observation vector from robot state.

        Args:
            joint_positions: 7 joint angles in radians
            ee_position: End-effector [x, y, z] position
            ee_orientation: End-effector quaternion [w, x, y, z]
            gripper_width: Gripper opening in meters
            cube_position: Cube [x, y, z] position
            goal_position: Goal [x, y, z] position
            cube_grasped: Whether cube is currently grasped

        Returns:
            23D observation vector as float32
        """
        # Compute distance to cube
        dist_to_cube = np.linalg.norm(np.array(ee_position) - np.array(cube_position))

        obs = np.concatenate([
            np.array(joint_positions, dtype=np.float32),       # [0:7]
            np.array(ee_position, dtype=np.float32),           # [7:10]
            np.array(ee_orientation, dtype=np.float32),        # [10:14]
            np.array([gripper_width], dtype=np.float32),       # [14:15]
            np.array(cube_position, dtype=np.float32),         # [15:18]
            np.array(goal_position, dtype=np.float32),         # [18:21]
            np.array([1.0 if cube_grasped else 0.0], dtype=np.float32),  # [21:22]
            np.array([dist_to_cube], dtype=np.float32),        # [22:23]
        ])

        return obs.astype(np.float32)


class RewardComputer:
    """Computes rewards for pick-and-place task.

    Supports both sparse and dense reward modes.
    """

    # Observation indices for reading state
    EE_POS_IDX = slice(7, 10)
    CUBE_POS_IDX = slice(15, 18)
    GOAL_POS_IDX = slice(18, 21)
    GRASPED_IDX = 21

    # Reward constants
    TASK_COMPLETE_REWARD = 10.0
    GRASP_BONUS = 5.0
    DROP_PENALTY = -5.0
    DISTANCE_SCALE = 10.0

    def __init__(self, mode: str = 'dense'):
        """Initialize the RewardComputer.

        Args:
            mode: 'dense' for shaped rewards, 'sparse' for only task completion
        """
        self.mode = mode

    def compute(self, obs: np.ndarray, action: np.ndarray,
                next_obs: np.ndarray, info: dict) -> float:
        """Compute reward for a transition.

        Args:
            obs: Previous observation
            action: Action taken
            next_obs: Resulting observation
            info: Additional info dict with flags like 'task_complete', 'cube_grasped'

        Returns:
            Scalar reward value
        """
        if self.mode == 'sparse':
            return self._sparse_reward(info)
        else:
            return self._dense_reward(obs, next_obs, info)

    def _sparse_reward(self, info: dict) -> float:
        """Compute sparse reward (only on task completion)."""
        if info.get('task_complete', False):
            return self.TASK_COMPLETE_REWARD
        return 0.0

    def _dense_reward(self, obs: np.ndarray, next_obs: np.ndarray, info: dict) -> float:
        """Compute dense shaped reward."""
        reward = 0.0

        # Check for grasp bonus
        if info.get('just_grasped', False):
            reward += self.GRASP_BONUS

        # Check for drop penalty
        if info.get('cube_dropped', False):
            reward += self.DROP_PENALTY

        # Check for task completion
        if info.get('task_complete', False):
            reward += self.TASK_COMPLETE_REWARD
            return reward

        # Distance-based shaping
        cube_grasped = info.get('cube_grasped', False)

        if not cube_grasped:
            # Phase 1: Reaching - reward getting closer to cube
            ee_pos = obs[self.EE_POS_IDX]
            cube_pos = obs[self.CUBE_POS_IDX]
            next_ee_pos = next_obs[self.EE_POS_IDX]
            next_cube_pos = next_obs[self.CUBE_POS_IDX]

            prev_dist = np.linalg.norm(ee_pos - cube_pos)
            curr_dist = np.linalg.norm(next_ee_pos - next_cube_pos)

            reward += (prev_dist - curr_dist) * self.DISTANCE_SCALE
        else:
            # Phase 2: Placing - reward moving cube toward goal
            cube_pos = obs[self.CUBE_POS_IDX]
            goal_pos = obs[self.GOAL_POS_IDX]
            next_cube_pos = next_obs[self.CUBE_POS_IDX]
            next_goal_pos = next_obs[self.GOAL_POS_IDX]

            prev_dist = np.linalg.norm(cube_pos - goal_pos)
            curr_dist = np.linalg.norm(next_cube_pos - next_goal_pos)

            reward += (prev_dist - curr_dist) * self.DISTANCE_SCALE

        return reward


class DemoPlayer:
    """Plays back recorded demonstrations.

    This class loads recorded demonstration data and provides methods
    to access individual episodes and filter by success status.

    Attributes:
        observations: All recorded observations
        actions: All recorded actions
        episode_starts: Start index of each episode
        episode_lengths: Length of each episode
        episode_success: Success flag for each episode
        num_episodes: Total number of episodes
        total_frames: Total number of frames
    """

    def __init__(self, filepath: str):
        """Load demonstrations from NPZ file.

        Args:
            filepath: Path to the .npz file containing demonstrations
        """
        data = np.load(filepath, allow_pickle=True)

        self.observations = data['observations']
        self.actions = data['actions']
        self.episode_starts = data['episode_starts']
        self.episode_lengths = data['episode_lengths']
        self.episode_success = data['episode_success']

        self.num_episodes = len(self.episode_starts)
        self.total_frames = len(self.observations)

    def get_episode(self, episode_idx: int) -> tuple:
        """Get observations and actions for a specific episode.

        Args:
            episode_idx: Index of the episode to retrieve

        Returns:
            Tuple of (observations, actions) arrays for the episode

        Raises:
            IndexError: If episode_idx is out of range
        """
        if episode_idx < 0 or episode_idx >= self.num_episodes:
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self.num_episodes})")

        start = self.episode_starts[episode_idx]
        length = self.episode_lengths[episode_idx]
        end = start + length

        return self.observations[start:end], self.actions[start:end]

    def get_successful_episodes(self) -> list:
        """Get indices of all successful episodes.

        Returns:
            List of episode indices where success flag is True
        """
        return [i for i in range(self.num_episodes) if self.episode_success[i]]


class FrankaKeyboardController:
    """Controller for Franka robot with keyboard input via pynput and Rich TUI."""

    # Control modes
    MODE_JOINT = 0
    MODE_ENDEFFECTOR = 1

    # Joint control parameters
    JOINT_INCREMENT = 0.05  # radians per key press
    GRIPPER_INCREMENT = 0.01  # meters per key press

    # End-effector control parameters
    POSITION_INCREMENT = 0.02  # meters per key press
    ROTATION_INCREMENT = 0.1  # radians per key press

    # Home position (neutral pose)
    HOME_JOINT_POSITIONS = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04])

    # Franka Panda joint limits (radians) - Standard specifications
    FRANKA_JOINT_LIMITS = {
        0: (-2.8973, 2.8973),   # Joint 1
        1: (-1.7628, 1.7628),   # Joint 2
        2: (-2.8973, 2.8973),   # Joint 3
        3: (-3.0718, -0.0698),  # Joint 4
        4: (-2.8973, 2.8973),   # Joint 5
        5: (-0.0175, 3.7525),   # Joint 6
        6: (-2.8973, 2.8973),   # Joint 7
    }

    def __init__(self, enable_recording: bool = False, demo_path: str = None,
                 reward_mode: str = "dense"):
        """Initialize the Franka robot and keyboard controller.

        Args:
            enable_recording: Enable demonstration recording mode
            demo_path: Path to save demonstrations (auto-generated with timestamp if None)
            reward_mode: Reward computation mode ('dense' or 'sparse')
        """
        # Create SimulationApp if not already created (e.g., by tests)
        global simulation_app, World, ArticulationAction, Franka, KinematicsSolver
        global euler_angles_to_quat, quat_to_euler_angles

        if simulation_app is None:
            simulation_app = SimulationApp({"headless": False})
        self.simulation_app = simulation_app

        # Import Isaac Sim modules after SimulationApp is created
        # (SimulationApp initialization makes these modules available)
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

        # Create TUI renderer
        self.tui = TUIRenderer()
        self.tui.set_last_command("Initializing...")
        self.tui.set_recording_enabled(enable_recording)

        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="my_franka")
        )
        self.world.scene.add_default_ground_plane()

        # Control state
        self.control_mode = self.MODE_JOINT
        self.active_joint = 0  # Which joint is being controlled (0-6 for arm, 7-8 for gripper)
        self.should_exit = False

        # Thread-safe command queue
        self.command_lock = threading.Lock()
        self.pending_commands = []

        # IK solver for end-effector control
        self.ik_solver = None

        # Recording configuration
        self.enable_recording = enable_recording
        self.reward_mode = reward_mode

        # Generate timestamped filename if no path provided
        if demo_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.demo_save_path = f"demos/recording_{timestamp}.npz"
        else:
            self.demo_save_path = demo_path

        # Recording components (initialized if enable_recording is True)
        self.recorder = None
        self.action_mapper = None
        self.obs_builder = None
        self.reward_computer = None
        self.current_obs = None
        self.cube_grasped = False
        self.prev_grasped = False

        # Checkpoint/auto-save state
        self.checkpoint_frame_counter = 0
        self.checkpoint_interval_frames = 50  # 5 seconds at ~10 Hz
        self.checkpoint_flash_frames = 0  # For "SAVED" flash indicator
        self.checkpoint_flash_duration = 10  # ~1 second of flash

        # Always initialize scene_manager for cube spawning (works in both modes)
        self.scene_manager = SceneManager(self.world)

        # Initialize recording components if enabled
        if self.enable_recording:
            self._init_recording_components()

        # End-effector target (position + orientation as euler angles)
        self.ee_target_position = np.array([0.3, 0.0, 0.4])
        self.ee_target_euler = np.array([np.pi, 0.0, 0.0])  # [roll, pitch, yaw]

        # Terminal settings (for disabling echo)
        self.old_terminal_settings = None

        # Setup keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()

        init_msg = "Initialization complete"
        if self.enable_recording:
            init_msg += " [RECORDING MODE]"
        self.tui.set_last_command(init_msg)

    def _init_recording_components(self):
        """Initialize recording components for demonstration collection."""
        # Initialize demo recorder
        self.recorder = DemoRecorder(obs_dim=23, action_dim=7)

        # Initialize action mapper and observation builder
        self.action_mapper = ActionMapper()
        self.obs_builder = ObservationBuilder()

        # Initialize reward computer
        self.reward_computer = RewardComputer(mode=self.reward_mode)

        self.tui.set_last_command("Recording components initialized")

    def _setup_recording_scene(self):
        """Set up the scene for recording (spawn cube and goal)."""
        if self.scene_manager is not None:
            # Spawn cube and goal marker
            self.scene_manager.spawn_cube()
            self.scene_manager.spawn_goal_marker()
            self.tui.set_last_command("Recording scene ready - Press ` to start")

    def _build_current_observation(self) -> np.ndarray:
        """Build observation from current robot and scene state."""
        if self.obs_builder is None:
            return None

        # Get robot state
        joint_positions = self.franka.get_joint_positions()[:7]
        ee_pos, ee_quat = self.franka.end_effector.get_world_pose()
        gripper_width = self.franka.get_joint_positions()[7] * 2  # Both fingers

        # Get scene state
        if self.scene_manager is not None and self.scene_manager.cube is not None:
            cube_pos, _ = self.scene_manager.get_cube_pose()
            goal_pos = self.scene_manager.get_goal_position()
        else:
            cube_pos = np.zeros(3)
            goal_pos = np.zeros(3)

        # Check grasp state
        if self.scene_manager is not None:
            self.cube_grasped = self.scene_manager.check_grasp(ee_pos, gripper_width)
        else:
            self.cube_grasped = False

        return self.obs_builder.build(
            joint_positions=joint_positions,
            ee_position=ee_pos,
            ee_orientation=ee_quat,
            gripper_width=gripper_width,
            cube_position=cube_pos,
            goal_position=goal_pos,
            cube_grasped=self.cube_grasped
        )

    def _record_step(self, action: np.ndarray):
        """Record a single step during demonstration collection."""
        if self.recorder is None or not self.recorder.is_recording:
            return

        # Build next observation
        next_obs = self._build_current_observation()

        # Build info dict for reward computation
        just_grasped = self.cube_grasped and not self.prev_grasped
        cube_dropped = not self.cube_grasped and self.prev_grasped

        # Check task completion
        task_complete = False
        if self.scene_manager is not None:
            task_complete = self.scene_manager.check_task_complete()

        info = {
            'cube_grasped': self.cube_grasped,
            'just_grasped': just_grasped,
            'cube_dropped': cube_dropped,
            'task_complete': task_complete
        }

        # Compute reward
        reward = 0.0
        if self.reward_computer is not None and self.current_obs is not None:
            reward = self.reward_computer.compute(
                self.current_obs, action, next_obs, info
            )

        # Record step
        done = task_complete
        self.recorder.record_step(self.current_obs, action, reward, done)

        # Update state
        self.prev_grasped = self.cube_grasped
        self.current_obs = next_obs

        # Update TUI with recording stats
        stats = self.recorder.get_stats()
        self.tui.set_recording_status(self.recorder.is_recording, stats)

        # Auto-finalize episode on task completion
        if task_complete:
            self.recorder.mark_episode_success(True)
            self.recorder.finalize_episode()
            self._reset_recording_episode()
            self.tui.set_last_command("Task complete! Episode finalized.")

    def _reset_recording_episode(self):
        """Reset scene for a new recording episode."""
        if self.scene_manager is not None:
            self.scene_manager.reset_scene()
        self.cube_grasped = False
        self.prev_grasped = False
        self.current_obs = self._build_current_observation()

    def _on_key_press(self, key):
        """Handle key press events from pynput."""
        try:
            # Handle character keys
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                self.tui.set_pressed_key(key_char)
                self._queue_command(('char', key_char))
            # Handle special keys
            elif key == keyboard.Key.tab:
                self.tui.set_pressed_key('tab')
                self._queue_command(('special', 'tab'))
            elif key == keyboard.Key.esc:
                self.tui.set_pressed_key('esc')
                self._queue_command(('special', 'esc'))
        except Exception as e:
            self.tui.set_last_command(f"Error: {e}")

    def _on_key_release(self, key):
        """Handle key release events from pynput."""
        try:
            # Clear pressed state for visual feedback
            if hasattr(key, 'char') and key.char:
                self.tui.clear_pressed_key(key.char.lower())
            elif key == keyboard.Key.tab:
                self.tui.clear_pressed_key('tab')
            elif key == keyboard.Key.esc:
                self.tui.clear_pressed_key('esc')
        except Exception:
            pass

    def _queue_command(self, command):
        """Add a command to the thread-safe queue."""
        with self.command_lock:
            self.pending_commands.append(command)

    def _process_commands(self):
        """Process all pending commands from the queue.

        Returns:
            str or None: The last 'char' command processed, or None if no char commands.
        """
        with self.command_lock:
            commands = self.pending_commands.copy()
            self.pending_commands.clear()

        last_char_cmd = None
        for cmd_type, cmd_value in commands:
            if cmd_type == 'special':
                if cmd_value == 'tab':
                    self._toggle_mode()
                elif cmd_value == 'esc':
                    self.should_exit = True
                    self.tui.set_last_command("Exiting...")
            elif cmd_type == 'char':
                # Universal commands work in any mode
                if cmd_value in ('`', '[', ']'):
                    self._handle_recording_command(cmd_value)
                elif cmd_value == 'c':
                    self._handle_spawn_random_cube()
                elif self.control_mode == self.MODE_JOINT:
                    self._process_joint_command(cmd_value)
                else:
                    self._process_endeffector_command(cmd_value)
                last_char_cmd = cmd_value  # Track last char command

        return last_char_cmd

    def _toggle_mode(self):
        """Toggle between joint and end-effector control modes."""
        self.control_mode = 1 - self.control_mode
        mode_name = "END-EFFECTOR" if self.control_mode == self.MODE_ENDEFFECTOR else "JOINT"

        self.tui.set_mode(self.control_mode)
        self.tui.set_last_command(f"Switched to {mode_name} mode")

        # Initialize end-effector target to current position when switching to EE mode
        if self.control_mode == self.MODE_ENDEFFECTOR:
            ee_position, ee_orientation = self.franka.end_effector.get_world_pose()
            self.ee_target_position = ee_position
            # Convert quaternion to euler for easier incremental control
            self.ee_target_euler = quat_to_euler_angles(ee_orientation)

    def _handle_recording_command(self, key: str):
        """Handle recording control keys.

        Args:
            key: The key pressed ('`' toggle, '[' success, ']' failure)
        """
        if key == '`':
            # Toggle recording
            if self.recorder is None:
                return  # No recorder initialized
            if self.recorder.is_recording:
                self.recorder.stop_recording()
                self.tui.set_last_command("Recording stopped")
            else:
                self.recorder.start_recording()
                self.tui.set_last_command("Recording started")

        elif key == '[':
            # Mark episode as success, finalize, and reset scene
            # Allow finalization even if recording is paused (user may pause then decide outcome)
            if self.recorder is not None and len(self.recorder.observations) > self.recorder.current_episode_start:
                self.recorder.mark_episode_success(True)
                self.recorder.finalize_episode()
                self._reset_recording_episode()
                stats = self.recorder.get_stats()
                self.tui.set_recording_status(True, stats)
                self.tui.set_last_command(f"Episode {stats['num_episodes']} SUCCESS - Scene reset")

        elif key == ']':
            # Mark episode as failure, finalize, and reset scene
            # Allow finalization even if recording is paused (user may pause then decide outcome)
            if self.recorder is not None and len(self.recorder.observations) > self.recorder.current_episode_start:
                self.recorder.mark_episode_success(False)
                self.recorder.finalize_episode()
                self._reset_recording_episode()
                stats = self.recorder.get_stats()
                self.tui.set_recording_status(True, stats)
                self.tui.set_last_command(f"Episode {stats['num_episodes']} FAILED - Scene reset")

    def _handle_spawn_random_cube(self):
        """Handle 'C' key press to spawn random falling cube."""
        if self.scene_manager is None:
            self.tui.set_last_command("Cube spawning requires scene manager")
            return

        cube = self.scene_manager.spawn_random_cube()
        count = self.scene_manager.random_cube_counter
        self.tui.set_last_command(f"Spawned random cube #{count}")

    def _perform_checkpoint_save(self) -> bool:
        """Perform checkpoint save of recording data.

        Returns:
            True if save was performed, False if no data to save
        """
        if self.recorder is None or len(self.recorder.observations) == 0:
            return False

        self.recorder.save(self.demo_save_path)
        self.tui.set_last_command(f"Checkpoint: {len(self.recorder.observations)} frames")
        self.checkpoint_flash_frames = self.checkpoint_flash_duration
        return True

    def _process_joint_command(self, key):
        """Process commands in joint control mode."""
        if key in ['1', '2', '3', '4', '5', '6', '7']:
            self.active_joint = int(key) - 1
            self.tui.set_active_joint(self.active_joint)
            self.tui.set_last_command(f"Selected Joint {self.active_joint + 1}")
        elif key == 'r':
            self.tui.set_last_command("Resetting to home position")
            self._reset_to_home()

    def _process_endeffector_command(self, key):
        """Process commands in end-effector control mode."""
        # Store previous valid state
        previous_position = self.ee_target_position.copy()
        previous_euler = self.ee_target_euler.copy()

        command_desc = ""

        # Position commands
        if key == 'w':
            self.ee_target_position[0] += self.POSITION_INCREMENT
            command_desc = "Move +X"
        elif key == 's':
            self.ee_target_position[0] -= self.POSITION_INCREMENT
            command_desc = "Move -X"
        elif key == 'a':
            self.ee_target_position[1] += self.POSITION_INCREMENT
            command_desc = "Move +Y"
        elif key == 'd':
            self.ee_target_position[1] -= self.POSITION_INCREMENT
            command_desc = "Move -Y"
        elif key == 'q':
            self.ee_target_position[2] += self.POSITION_INCREMENT
            command_desc = "Move +Z"
        elif key == 'e':
            self.ee_target_position[2] -= self.POSITION_INCREMENT
            command_desc = "Move -Z"
        # Rotation commands
        elif key == 'i':
            self.ee_target_euler[1] += self.ROTATION_INCREMENT  # pitch up
            command_desc = "Pitch +"
        elif key == 'k':
            self.ee_target_euler[1] -= self.ROTATION_INCREMENT  # pitch down
            command_desc = "Pitch -"
        elif key == 'j':
            self.ee_target_euler[2] += self.ROTATION_INCREMENT  # yaw left
            command_desc = "Yaw +"
        elif key == 'l':
            self.ee_target_euler[2] -= self.ROTATION_INCREMENT  # yaw right
            command_desc = "Yaw -"
        elif key == 'u':
            self.ee_target_euler[0] += self.ROTATION_INCREMENT  # roll left
            command_desc = "Roll +"
        elif key == 'o':
            self.ee_target_euler[0] -= self.ROTATION_INCREMENT  # roll right
            command_desc = "Roll -"

        # Validate IK before accepting the change
        if command_desc:
            if not self._validate_ik_solution():
                # Revert to previous valid state if IK fails
                self.ee_target_position = previous_position
                self.ee_target_euler = previous_euler
                self.tui.set_ik_status(False, "Workspace limit reached")
                self.tui.set_last_command(f"{command_desc} - FAILED")
            else:
                self.tui.set_ik_status(True)
                self.tui.set_last_command(command_desc)

    def _validate_ik_solution(self):
        """Validate that IK solution exists for current target pose.

        Returns:
            bool: True if IK solution found, False otherwise.
        """
        # Initialize IK solver if not done yet
        if self.ik_solver is None:
            self.ik_solver = KinematicsSolver(self.franka)

        # Convert euler angles to quaternion
        target_orientation = euler_angles_to_quat(self.ee_target_euler)

        # Compute IK
        _, success = self.ik_solver.compute_inverse_kinematics(
            target_position=self.ee_target_position,
            target_orientation=target_orientation
        )

        return success

    def _reset_to_home(self):
        """Reset the robot to home position."""
        articulation_controller = self.franka.get_articulation_controller()
        action = ArticulationAction(joint_positions=self.HOME_JOINT_POSITIONS)
        articulation_controller.apply_action(action)

    def _apply_joint_control(self, key):
        """Apply joint control based on key press."""
        current_positions = self.franka.get_joint_positions()
        target_positions = current_positions.copy()

        if key == 'w':
            target_positions[self.active_joint] += self.JOINT_INCREMENT
            self.tui.set_last_command(f"Joint {self.active_joint + 1}: {target_positions[self.active_joint]:.3f} rad")
        elif key == 's':
            target_positions[self.active_joint] -= self.JOINT_INCREMENT
            self.tui.set_last_command(f"Joint {self.active_joint + 1}: {target_positions[self.active_joint]:.3f} rad")
        elif key == 'q':
            # Open gripper
            target_positions[7] = min(target_positions[7] + self.GRIPPER_INCREMENT, 0.05)
            target_positions[8] = target_positions[7]
            self.tui.set_last_command(f"Gripper opening: {target_positions[7]:.4f} m")
        elif key == 'e':
            # Close gripper
            target_positions[7] = max(target_positions[7] - self.GRIPPER_INCREMENT, 0.0)
            target_positions[8] = target_positions[7]
            self.tui.set_last_command(f"Gripper closing: {target_positions[7]:.4f} m")

        # Apply action
        articulation_controller = self.franka.get_articulation_controller()
        action = ArticulationAction(joint_positions=target_positions)
        articulation_controller.apply_action(action)

    def _apply_endeffector_control(self):
        """Apply end-effector control using IK."""
        # Initialize IK solver if not done yet
        if self.ik_solver is None:
            self.ik_solver = KinematicsSolver(self.franka)

        # Convert euler angles to quaternion
        target_orientation = euler_angles_to_quat(self.ee_target_euler)

        # Compute IK (should always succeed since we validate before accepting changes)
        actions, success = self.ik_solver.compute_inverse_kinematics(
            target_position=self.ee_target_position,
            target_orientation=target_orientation
        )

        if success:
            # Apply the IK solution
            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(actions)

    def _update_tui_state(self):
        """Update TUI with current robot state."""
        # Get current joint positions
        current_positions = self.franka.get_joint_positions()
        gripper_opening = current_positions[7] if len(current_positions) > 7 else 0.04
        joint_positions = current_positions[:7]  # First 7 joints are the arm joints

        # Update telemetry
        self.tui.update_telemetry(
            position=self.ee_target_position,
            euler=self.ee_target_euler,
            gripper=gripper_opening,
            joint_positions=joint_positions
        )

    def _disable_terminal_echo(self):
        """Disable terminal echo to prevent key presses from appearing in TUI."""
        try:
            # Save current terminal settings
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)

            # Get current settings and modify them
            new_settings = termios.tcgetattr(sys.stdin)

            # Disable echo (ECHO flag)
            new_settings[3] = new_settings[3] & ~termios.ECHO

            # Apply new settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        except Exception as e:
            # If we can't modify terminal settings, continue anyway
            self.tui.set_last_command(f"Warning: Could not disable echo: {e}")

    def _restore_terminal_settings(self):
        """Restore original terminal settings."""
        try:
            if self.old_terminal_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        except Exception:
            pass  # Ignore errors on cleanup

    def run(self):
        """Main simulation loop with Rich TUI."""
        self.tui.set_last_command("Starting simulation...")

        # Set up recording scene BEFORE first reset if enabled
        if self.enable_recording:
            self._setup_recording_scene()

        # Reset world after scene setup to properly initialize physics
        self.world.reset()

        # Try to retrieve actual joint limits from the robot
        try:
            # Attempt to get DOF limits if available
            dof_limits = self.franka.get_dof_limits()
            if dof_limits is not None and len(dof_limits) >= 7:
                # Extract limits for first 7 joints (arm joints)
                joint_limits = {}
                for i in range(7):
                    joint_limits[i] = (dof_limits[i][0], dof_limits[i][1])
                self.tui.joint_limits = joint_limits
            else:
                raise AttributeError("DOF limits not available")
        except (AttributeError, IndexError):
            # Fall back to hardcoded Franka Panda limits
            self.tui.joint_limits = self.FRANKA_JOINT_LIMITS

        # Set initial home position
        self._reset_to_home()
        self.world.step(render=True)

        # Build initial observation if recording enabled
        if self.enable_recording:
            self.current_obs = self._build_current_observation()

        reset_needed = False
        last_key_processed = None

        ready_msg = "Ready - Use keyboard to control"
        if self.enable_recording:
            ready_msg += " | `=Record, [=Success, ]=Fail (Auto-saves every 5s)"
        self.tui.set_last_command(ready_msg)

        # Disable terminal echo to prevent keypresses from appearing
        self._disable_terminal_echo()

        try:
            # Start Live TUI context
            with Live(self.tui.render(), refresh_per_second=10, screen=True) as live:
                while self.simulation_app.is_running() and not self.should_exit:
                    self.world.step(render=True)

                    if self.world.is_stopped() and not reset_needed:
                        reset_needed = True

                    if self.world.is_playing():
                        if reset_needed:
                            self.world.reset()
                            self._reset_to_home()
                            reset_needed = False

                        # Process pending commands and get last char command
                        last_char_cmd = self._process_commands()

                        # Update last_key_processed if we got a new command
                        if last_char_cmd is not None:
                            last_key_processed = last_char_cmd

                        # Apply continuous control in end-effector mode
                        action = None
                        if self.control_mode == self.MODE_ENDEFFECTOR and last_key_processed:
                            if last_key_processed in ['w', 's', 'a', 'd', 'q', 'e', 'i', 'k', 'j', 'l', 'u', 'o', 'g', 'h']:
                                self._apply_endeffector_control()
                                # Get action for recording
                                if self.enable_recording and self.action_mapper is not None:
                                    action = self.action_mapper.map_key(last_key_processed)
                        elif self.control_mode == self.MODE_JOINT and last_key_processed:
                            if last_key_processed in ['w', 's', 'q', 'e']:
                                self._apply_joint_control(last_key_processed)
                            last_key_processed = None  # Only apply once per key press in joint mode

                        # Record step if recording is active
                        if self.enable_recording and action is not None:
                            self._record_step(action)

                        # Checkpoint auto-save (only when recording is active)
                        if self.enable_recording and self.recorder is not None:
                            if self.recorder.is_recording:
                                self.checkpoint_frame_counter += 1
                                if self.checkpoint_frame_counter >= self.checkpoint_interval_frames:
                                    self._perform_checkpoint_save()
                                    self.checkpoint_frame_counter = 0

                            # Decrement flash counter
                            if self.checkpoint_flash_frames > 0:
                                self.checkpoint_flash_frames -= 1

                        # Update TUI flash state
                        if self.enable_recording:
                            self.tui.set_checkpoint_flash(self.checkpoint_flash_frames > 0)

                        # Update TUI state
                        self._update_tui_state()
                        live.update(self.tui.render())
        finally:
            # Auto-save recording on exit if there's data
            if self.enable_recording and self.recorder is not None:
                if len(self.recorder.observations) > 0:
                    self.tui.set_last_command("Saving recording data...")

                    # Finalize any pending episode
                    pending_frames = len(self.recorder.observations) - self.recorder.current_episode_start
                    if pending_frames > 0 and self.recorder.is_recording:
                        self.recorder.mark_episode_success(False)
                        self.recorder.finalize_episode()

                    self.recorder.save(self.demo_save_path)
                    print(f"\nAuto-saved {len(self.recorder.observations)} frames to {self.demo_save_path}")

            # Always restore terminal settings on exit
            self._restore_terminal_settings()
            self.listener.stop()
            self.simulation_app.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Franka Keyboard Control with optional recording mode'
    )
    parser.add_argument(
        '--enable-recording', action='store_true',
        help='Enable demonstration recording mode with pick-and-place task'
    )
    parser.add_argument(
        '--demo-path', type=str, default=None,
        help='Path to save demonstrations (default: demos/recording_TIMESTAMP.npz)'
    )
    parser.add_argument(
        '--reward-mode', type=str, default='dense',
        choices=['dense', 'sparse'],
        help='Reward computation mode (default: dense)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = FrankaKeyboardController(
        enable_recording=args.enable_recording,
        demo_path=args.demo_path,
        reward_mode=args.reward_mode
    )
    controller.run()
