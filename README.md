# Franka Arm Keyboard Control for Isaac Sim

A minimal, keyboard-controlled Franka Panda robot example with 
dual control modes (joint and end-effector) and intelligent workspace limit detection.

## What We Built Today

This project implements a standalone Isaac Sim example that allows 
real-time keyboard control of a Franka Panda robot arm with:

- **Dual Control Modes**: Switch between direct joint control and IK-based end-effector control
- **Demonstration Recording**: Record robot teleoperation for imitation learning with automatic checkpoint saves
- **Rich Terminal UI**: Real-time status display with recording indicators and visual button feedback
- **PyInput Integration**: System-wide keyboard capture using pynput library (thread-safe)
- **Intelligent Workspace Limits**: IK validation prevents movements outside reachable workspace
- **Incremental Control**: Small, precise position and rotation adjustments
- **Clean Architecture**: Thread-safe command queue, proper SimulationApp lifecycle

## Quick Start

### Prerequisites
- Isaac Sim 5.0.0 standalone
- Python 3.11 (bundled with Isaac Sim)

### Run

**Teleoperation only:**
```bash
./run.sh
```

**Recording demonstrations:**
```bash
./run.sh --enable-recording
```

**Replay recorded demonstrations:**
```bash
./run.sh replay.py demos/recording_TIMESTAMP.npz
```

**Train behavioral cloning model:**
```bash
./run.sh train_bc.py demos/recording_TIMESTAMP.npz
```

**Train RL agent (requires gymnasium and stable-baselines3):**
```python
from franka_rl_env import FrankaPickPlaceEnv
from stable_baselines3 import PPO

env = FrankaPickPlaceEnv(reward_mode='dense')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("models/ppo_franka")
```

The Isaac Sim GUI will launch with the Franka arm loaded. Use keyboard controls immediately.

## Features

### 🎮 Dual Control Modes

**Joint Control Mode** (Default)
- Direct control of individual joint angles
- Select joints 1-7 with number keys
- Precise manipulation of robot configuration
- No workspace restrictions

**End-Effector Control Mode**
- Control gripper position in 3D space (X, Y, Z)
- Control gripper orientation (roll, pitch, yaw)
- Automatic inverse kinematics solving
- Workspace limit validation

### 🛡️ Intelligent Workspace Limits

The end-effector control mode includes IK validation:
- Every movement is validated **before** being applied
- If IK cannot solve for a target pose, the movement is **rejected**
- Robot stays at last valid position
- Clear feedback: `"Reached workspace limit - cannot move further in this direction"`
- No warning spam - single message when you hit a boundary
- Move in opposite direction or adjust orientation to continue

### 🧵 Thread-Safe Architecture

- PyInput listener runs in separate thread
- Thread-safe command queue for keyboard events
- Proper synchronization between keyboard input and simulation loop
- Clean shutdown handling

## Control Mappings

### Joint Control Mode

| Key | Action |
|-----|--------|
| `1-7` | Select joint to control (Joint 1 through Joint 7) |
| `W` | Increase selected joint angle (+0.05 rad) |
| `S` | Decrease selected joint angle (-0.05 rad) |
| `Q` | Open gripper |
| `E` | Close gripper |
| `R` | Reset to home position |

**Example Workflow:**
1. Press `3` → Select joint 3
2. Press `W` multiple times → Rotate joint 3 upward
3. Press `5` → Select joint 5
4. Press `S` → Rotate joint 5 downward

### End-Effector Control Mode

#### Translation

| Key | Action | Increment |
|-----|--------|-----------|
| `W` | Move forward (+X axis) | 0.02 m |
| `S` | Move backward (-X axis) | 0.02 m |
| `A` | Move left (+Y axis) | 0.02 m |
| `D` | Move right (-Y axis) | 0.02 m |
| `Q` | Move up (+Z axis) | 0.02 m |
| `E` | Move down (-Z axis) | 0.02 m |

#### Rotation

| Key | Action | Increment |
|-----|--------|-----------|
| `I` | Pitch up | 0.1 rad (~5.7°) |
| `K` | Pitch down | 0.1 rad (~5.7°) |
| `J` | Yaw left | 0.1 rad (~5.7°) |
| `L` | Yaw right | 0.1 rad (~5.7°) |
| `U` | Roll left | 0.1 rad (~5.7°) |
| `O` | Roll right | 0.1 rad (~5.7°) |

### Global Controls

| Key | Action |
|-----|--------|
| `Tab` | Switch between Joint and End-Effector modes |
| `Esc` | Exit application |

### Recording Controls (when using --enable-recording)

| Key | Action |
|-----|--------|
| `` ` `` | Start/Stop recording |
| `[` | Mark current episode as SUCCESS and finalize |
| `]` | Mark current episode as FAILURE and finalize |

**Recording features:**
- **Auto-save every 5 seconds**: Recording data is automatically checkpointed to `demos/recording_TIMESTAMP.npz`
- **Visual feedback**: Red "● REC" indicator blinks during recording, "SAVED" flash appears during checkpoint
- **Button press feedback**: Recording control buttons highlight when pressed, just like movement controls
- **Auto-save on exit**: Press Esc to exit - any pending recording data is automatically finalized and saved

## Mode Comparison

| Aspect | Joint Control | End-Effector Control |
|--------|---------------|---------------------|
| **Control Space** | Joint angles (configuration space) | Cartesian position + orientation (task space) |
| **Best For** | Fine-tuning specific joints, avoiding singularities | Intuitive 3D positioning, task-oriented control |
| **Workspace** | Unrestricted (any valid joint angles) | Limited by IK solver and robot kinematics |
| **Precision** | High (direct angle control) | Depends on IK solution quality |
| **Learning Curve** | Steeper (need to understand joint relationships) | Easier (natural 3D thinking) |
| **Use Cases** | Joint space planning, exploring configurations | Pick and place, reaching targets, orienting tools |

**When to use Joint Control:**
- Need to reach specific joint configurations
- Working near singularities or workspace boundaries
- Fine-tuning robot pose after coarse positioning
- Learning how joint angles affect end-effector

**When to use End-Effector Control:**
- Need to position gripper at specific 3D location
- Task requires specific orientation (e.g., vertical grasp)
- More intuitive for spatial reasoning
- Performing pick-and-place operations

## Common Workflows

### 1. Moving to a Specific Position

**Goal:** Position the gripper at coordinates (0.5, 0.2, 0.3)

```
1. Press Tab to switch to End-Effector mode
2. Press W repeatedly to move forward (watch X coordinate)
3. Press A repeatedly to move left (watch Y coordinate)
4. Press Q/E to adjust height (watch Z coordinate)
5. If you hit "Reached workspace limit", adjust orientation first (I/K/J/L)
6. Continue adjusting until target is reached
```

### 2. Orienting the Gripper

**Goal:** Point gripper downward for top-down grasp

```
1. Tab to End-Effector mode
2. Press I repeatedly to pitch down (~90° = 1.57 rad)
3. Adjust position with W/A/S/D if needed
4. Fine-tune with J/L for yaw alignment
```

### 3. Recovering from Workspace Limits

**Scenario:** Message says "Reached workspace limit"

```
Option A - Adjust Orientation:
1. Try changing orientation with I/K/J/L/U/O
2. Different orientations may unlock new positions
3. Retry original movement direction

Option B - Switch to Joint Mode:
1. Press Tab to switch to Joint mode
2. Manually adjust joints to move away from boundary
3. Press Tab to return to End-Effector mode
4. Continue with validated movements
```

### 4. Pick and Place Sequence

**Goal:** Simulate picking an object

```
1. Tab to End-Effector mode
2. Position above object (W/A/S/D/Q/E)
3. Orient gripper (I/K for pitch, J/L for yaw)
4. Press E to close gripper
5. Press Q to lift up
6. Move to target location (W/A/S/D)
7. Press E to open gripper and release
```

### 5. Recording a Demonstration

**Goal:** Record a pick-and-place demonstration for imitation learning

```
1. Run ./run.sh --enable-recording to start recording mode
2. Position robot at starting pose
3. Press ` (backtick) to start recording
4. Perform the pick-and-place task
5. Press [ to mark episode as success (or ] for failure)
6. Repeat steps 3-5 for multiple episodes
7. Press Esc to exit - data auto-saves to demos/recording_TIMESTAMP.npz
```

**Notes:**
- Recording automatically saves every 5 seconds (checkpoint-style)
- Watch the left panel for "● REC" indicator (blinks during recording)
- "SAVED" flash appears when checkpoint occurs
- All data is preserved even if you don't manually save

## Technical Implementation

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FrankaKeyboardController                │
│                                                             │
│  ┌──────────────┐          ┌─────────────────┐              │
│  │   PyInput    │          │  Command Queue  │              │
│  │   Listener   │─────────▶│  (Thread-Safe)  │              │
│  │  (Thread)    │          └────────┬────────┘              │
│  └──────────────┘                   │                       │
│                                     │                       │
│  ┌──────────────────────────────────▼─────────────────┐     │
│  │         Simulation Loop (Main Thread)              │     │
│  │                                                    │     │
│  │  1. world.step(render=True)                        │     │
│  │  2. _process_commands() ◄─── Dequeue commands      │     │
│  │  3. Mode check:                                    │     │
│  │     ┌────────────┬────────────┐                    │     │
│  │     │ Joint Mode │ EE Mode    │                    │     │
│  │     │            │            │                    │     │
│  │     │ Direct     │ Validate   │                    │     │
│  │     │ Position   │ IK First   │                    │     │
│  │     │ Setting    │ ▼          │                    │     │
│  │     │            │ Apply IK   │                    │     │
│  │     └────────────┴────────────┘                    │     │
│  │  4. articulation_controller.apply_action()         │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Key Components                          │   │
│  │  • SimulationApp - Isaac Sim launcher                │   │
│  │  • World - Physics simulation orchestrator           │   │
│  │  • Franka - Robot wrapper with articulation          │   │
│  │  • KinematicsSolver - IK computation                 │   │
│  │  • ArticulationController - Joint control interface  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

**Main Class: `FrankaKeyboardController`**

Key Methods:

```python
__init__()
├── Initialize SimulationApp, World, Franka robot
├── Setup keyboard listener (pynput)
├── Initialize state (mode, targets, commands)
└── Print controls

# Keyboard Input (runs in separate thread)
_on_key_press(key) → _queue_command(cmd)

# Command Processing (main thread)
_process_commands()
├── Dequeue all pending commands
├── Route to mode-specific handler
│   ├── _process_joint_command()
│   └── _process_endeffector_command()
│       ├── Apply position/rotation delta
│       ├── _validate_ik_solution() ◄── KEY FEATURE
│       ├── If invalid: revert to previous state
│       └── If valid: accept and print feedback

# Control Application
_apply_joint_control()
└── Direct joint position setting via ArticulationAction

_apply_endeffector_control()
└── Compute IK and apply joint positions

# Main Loop
run()
└── while simulation_app.is_running():
    ├── world.step(render=True)
    ├── _process_commands()
    └── Apply control based on current mode
```

**Key Design Decisions:**

1. **Thread Separation**: Keyboard input captured in pynput thread, processed in main simulation thread
2. **Command Queue**: Thread-safe deque prevents race conditions
3. **State Validation**: IK checked before committing to position changes
4. **Mode Encapsulation**: Separate handlers for joint vs end-effector logic

### IK Validation Logic

The workspace limit detection feature works as follows:

```python
def _process_endeffector_command(self, key):
    # 1. Store current valid state
    previous_position = self.ee_target_position.copy()
    previous_euler = self.ee_target_euler.copy()

    # 2. Apply the requested movement
    if key == 'w':
        self.ee_target_position[0] += POSITION_INCREMENT
    # ... other keys

    # 3. VALIDATE before accepting
    if not self._validate_ik_solution():
        # IK cannot solve for this pose
        # REVERT to previous valid state
        self.ee_target_position = previous_position
        self.ee_target_euler = previous_euler
        print("Reached workspace limit")
    else:
        # IK succeeded - movement is valid
        print("X={:.3f}".format(self.ee_target_position[0]))

def _validate_ik_solution(self):
    """Test if IK can solve for current target."""
    target_orientation = euler_angles_to_quat(self.ee_target_euler)
    _, success = self.ik_solver.compute_inverse_kinematics(
        target_position=self.ee_target_position,
        target_orientation=target_orientation
    )
    return success
```

**Why This Works:**

- **Predictive**: Tests IK *before* committing to the change
- **Atomic**: Movement either succeeds completely or reverts completely
- **User-Friendly**: No invalid states, no spammy warnings
- **Recoverable**: User can immediately try different direction/orientation

## Demonstration Recording System

### Overview

The project includes a complete demonstration recording system for imitation learning:

- **DemoRecorder**: Records (observation, action, reward, done) tuples during teleoperation
- **Episode Management**: Mark episodes as success/failure, automatic finalization
- **Auto-Save**: Checkpoint-style saves every 5 seconds to prevent data loss
- **Rich TUI**: Visual recording indicators, button feedback, status display
- **Observation Space**: 23D state vector (EE pose, joint state, gripper, cube position)
- **Action Space**: 7D control vector (EE velocity, angular velocity, gripper)

### Recording Workflow

1. **Start Recording Session**: `./record.sh`
2. **Control Robot**: Use normal teleoperation controls
3. **Toggle Recording**: Press `` ` `` to start/stop recording
4. **Mark Episodes**: Press `[` for success, `]` for failure
5. **Auto-Save**: Data checkpoints every 5 seconds automatically
6. **Exit**: Press Esc - final save happens automatically

### Recorded Data Format

NPZ file contains:
```python
{
    'observations': np.ndarray,        # Shape (N, 23)
    'actions': np.ndarray,             # Shape (N, 7)
    'rewards': np.ndarray,             # Shape (N,)
    'dones': np.ndarray,               # Shape (N,)
    'episode_starts': np.ndarray,      # Episode boundary indices
    'episode_success': np.ndarray      # Boolean success flags
}
```

### Training Behavioral Cloning

Use the recorded demonstrations to train an imitation learning policy:

```bash
./run.sh train_bc.py demos/recording_TIMESTAMP.npz
```

This trains a behavioral cloning model using the recorded demonstrations and saves the trained policy to `models/`.

### Visual Feedback

When recording is active, the left TUI panel shows:
- **Status**: "● REC" (blinking red) or "○ IDLE" (dim)
- **Episode**: Current episode number
- **Frames**: Number of recorded frames
- **Controls**: Button feedback (highlights when pressed)
- **Auto-save**: "SAVED" flash indicator during checkpoint
- **Last Action**: Most recent command executed

All controls use the same visual button feedback system as the movement controls for consistency.

## Reinforcement Learning Environment

The project includes a Gymnasium-compatible environment for training RL agents:

### FrankaPickPlaceEnv

**Location:** `src/franka_rl_env.py`

A standard Gymnasium environment wrapper for the Franka pick-and-place task, compatible with Stable-Baselines3, RLlib, CleanRL, and other RL libraries.

**Observation Space (23D):**
- Joint positions (7)
- End-effector position (3) and orientation (4, quaternion)
- Gripper width (1)
- Cube position (3) and goal position (3)
- Cube grasped flag (1) and distance to cube (1)

**Action Space (7D, continuous [-1, 1]):**
- End-effector velocity (3): dx, dy, dz
- End-effector angular velocity (3): droll, dpitch, dyaw
- Gripper command (1)

**Features:**
- Dense and sparse reward modes
- Configurable action scaling
- Episode termination on: task completion, cube dropped, out-of-bounds
- Episode truncation at max_episode_steps (default 500)
- 59 comprehensive tests

### Example Usage

```python
from franka_rl_env import FrankaPickPlaceEnv
from stable_baselines3 import PPO

# Create environment
env = FrankaPickPlaceEnv(
    reward_mode='dense',      # 'dense' or 'sparse'
    max_episode_steps=500,    # Episode truncation limit
)

# Train with Stable-Baselines3
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save and load
model.save("models/ppo_franka")
model = PPO.load("models/ppo_franka")

# Evaluate
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
