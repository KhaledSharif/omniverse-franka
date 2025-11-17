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
        self.ik_valid = True
        self.ik_message = ""
        self.last_command = "Ready"

        # Mode constants
        self.MODE_JOINT = 0
        self.MODE_ENDEFFECTOR = 1

    def set_pressed_key(self, key):  # pragma: no cover
        """Mark a key as pressed for highlighting."""
        self.pressed_keys.add(key)

    def clear_pressed_key(self, key):  # pragma: no cover
        """Mark a key as released."""
        self.pressed_keys.discard(key)

    def update_telemetry(self, position, euler, gripper):  # pragma: no cover
        """Update telemetry values."""
        self.position = position
        self.euler = euler
        self.gripper = gripper

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

        # Split content into left (controls) and right (state)
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

    def __init__(self):
        """Initialize the Franka robot and keyboard controller."""
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

        self.tui.set_last_command("Initialization complete")

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
                if self.control_mode == self.MODE_JOINT:
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
        # Get current gripper position
        current_positions = self.franka.get_joint_positions()
        gripper_opening = current_positions[7] if len(current_positions) > 7 else 0.04

        # Update telemetry
        self.tui.update_telemetry(
            position=self.ee_target_position,
            euler=self.ee_target_euler,
            gripper=gripper_opening
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
        self.world.reset()

        # Set initial home position
        self._reset_to_home()
        self.world.step(render=True)

        reset_needed = False
        last_key_processed = None

        self.tui.set_last_command("Ready - Use keyboard to control")

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
                        if self.control_mode == self.MODE_ENDEFFECTOR and last_key_processed:
                            if last_key_processed in ['w', 's', 'a', 'd', 'q', 'e', 'i', 'k', 'j', 'l', 'u', 'o']:
                                self._apply_endeffector_control()
                        elif self.control_mode == self.MODE_JOINT and last_key_processed:
                            if last_key_processed in ['w', 's', 'q', 'e']:
                                self._apply_joint_control(last_key_processed)
                            last_key_processed = None  # Only apply once per key press in joint mode

                        # Update TUI state
                        self._update_tui_state()
                        live.update(self.tui.render())
        finally:
            # Always restore terminal settings on exit
            self._restore_terminal_settings()
            self.listener.stop()
            self.simulation_app.close()


if __name__ == "__main__":
    controller = FrankaKeyboardController()
    controller.run()
