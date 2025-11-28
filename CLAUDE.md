# CLAUDE.md - Project Guide

## Overview

Franka Panda robot keyboard teleoperation in NVIDIA Isaac Sim 5.0.0 with demonstration recording for imitation learning and RL training.

## Running Tests

**IMPORTANT:** Always use `./run_tests.sh` to run tests, never `python -m pytest` directly.

```bash
./run_tests.sh                    # Run all tests
./run_tests.sh -v                 # Verbose output
./run_tests.sh -k test_name       # Run specific test
./run_tests.sh --cov              # With coverage
```

**Why:** This project depends on Isaac Sim's bundled Python environment (`~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh`) which includes NumPy and other dependencies configured for Isaac Sim. Using system Python will fail due to missing or incompatible packages.

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `./run.sh` | Universal launcher - runs any Python script with Isaac Sim's Python |
| `./run.sh --enable-recording` | Run teleoperation with recording enabled |
| `./run.sh replay.py <demo.npz>` | Replay/inspect recorded demos |
| `./run.sh train_bc.py <demo.npz>` | Train behavioral cloning model |
| `./run.sh train_rl.py` | Train PPO agent (add `--headless` for faster training) |
| `./run.sh eval_policy.py <model.zip>` | Evaluate trained RL policy |
| `./run_tests.sh` | Run pytest tests |

All scripts use Isaac Sim's Python interpreter.

## RL Training Workflow

```bash
# 1. Record demonstrations
./run.sh --enable-recording

# 2. Train PPO with BC warmstart
./run.sh train_rl.py --headless --bc-warmstart demos/recording.npz --timesteps 500000

# 3. Monitor training
tensorboard --logdir runs/

# 4. Evaluate trained policy
./run.sh eval_policy.py models/ppo_franka.zip --episodes 100
```

## Project Structure

```
src/
  franka_keyboard_control.py      # Main application + all classes
  franka_rl_env.py                # Gymnasium RL environment wrapper
  train_rl.py                     # PPO training with BC warmstart
  eval_policy.py                  # Policy evaluation script
  train_bc.py                     # Behavioral cloning training
  replay.py                       # Demo playback script
  test_franka_keyboard_control.py # 201 tests for main controller
  test_franka_rl_env.py           # 59 tests for RL environment
demos/                            # Recorded demonstration files (.npz)
models/                           # Trained model files (.zip, .pt)
runs/                             # TensorBoard logs
```

## Key Classes

### In `src/franka_keyboard_control.py`:
- **FrankaKeyboardController** - Main controller with keyboard input and simulation loop
- **TUIRenderer** - Rich terminal UI with robot state and recording status
- **SceneManager** - Spawns cube/goal marker (uses real Isaac Sim primitives or mocks)
- **DemoRecorder** - Records (obs, action, reward, done) transitions
- **DemoPlayer** - Loads and iterates recorded demonstrations
- **ActionMapper** - Maps keyboard input to 7D action vectors
- **ObservationBuilder** - Builds 23D observation vectors
- **RewardComputer** - Computes dense/sparse rewards

### In `src/franka_rl_env.py`:
- **FrankaPickPlaceEnv** - Gymnasium environment for RL training (compatible with Stable-Baselines3)

## Teleoperation Controls

### Universal Controls (both modes)
- **Tab** - Switch between joint/end-effector mode
- **Esc** - Exit
- **R** - Reset to home position
- **C** - Spawn random cube (falls from 2m height)

### Recording Controls
- **`** (backtick) - Toggle recording on/off
- **[** - Mark episode as success and finalize
- **]** - Mark episode as failure and finalize

Recording automatically saves every 5 seconds to `demos/recording_TIMESTAMP.npz`.
On exit (Esc), any pending data is auto-saved and finalized.

## Observation Space (23D)

- Joint positions (7)
- End-effector position (3) and orientation (4 quaternion)
- Gripper width (1)
- Cube position (3)
- Goal position (3)
- Cube grasped flag (1)
- Distance to cube (1)

## Action Space (7D)

- End-effector velocity: dx, dy, dz (3)
- End-effector angular velocity: droll, dpitch, dyaw (3)
- Gripper command (1)

## Development Notes

- Mock objects in SceneManager include `name` attribute (required by Isaac Sim's `world.scene.add()`)
- Tests mock Isaac Sim imports to run without GPU/simulator
- Isaac Sim primitives: `DynamicCuboid`, `VisualCuboid` from `isaacsim.core.api.objects`
- Install RL dependencies: `./run.sh -m pip install stable-baselines3 tensorboard imitation`
