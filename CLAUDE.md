# CLAUDE.md - Project Guide

## Overview

Franka Panda robot keyboard teleoperation in NVIDIA Isaac Sim 5.0.0 with demonstration recording for imitation learning.

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
| `./run.sh` | Run the teleoperation application |
| `./run_tests.sh` | Run pytest tests |
| `./record.sh` | Record demonstrations (adds `--enable-recording` flag) |
| `./replay.sh <demo.npz>` | Replay/inspect recorded demos |
| `./sb3.sh <demo.npz>` | Train behavioral cloning model |

All scripts use Isaac Sim's Python interpreter.

## Project Structure

```
franka_keyboard_control.py   # Main application + all classes
test_franka_keyboard_control.py  # 150 tests (TDD)
replay_demo.py               # Demo playback script
train_bc.py                  # Behavioral cloning training
demos/                       # Recorded demonstration files (.npz)
models/                      # Trained model files (.pt)
```

## Key Classes (all in franka_keyboard_control.py)

- **FrankaKeyboardController** - Main controller with keyboard input and simulation loop
- **TUIRenderer** - Rich terminal UI with robot state and recording status
- **SceneManager** - Spawns cube/goal marker (uses real Isaac Sim primitives or mocks)
- **DemoRecorder** - Records (obs, action, reward, done) transitions
- **DemoPlayer** - Loads and iterates recorded demonstrations
- **ActionMapper** - Maps keyboard input to 7D action vectors
- **ObservationBuilder** - Builds 23D observation vectors
- **RewardComputer** - Computes dense/sparse rewards

## Recording Controls

- **`** (backtick) - Toggle recording on/off
- **[** - Mark episode as success and finalize
- **]** - Mark episode as failure and finalize

Recording automatically saves every 5 seconds to `demos/recording_TIMESTAMP.npz`.
On exit (Esc), any pending data is auto-saved and finalized.

## Observation Space (23D)

- End-effector position (3) and orientation (4 quaternion)
- Joint positions (7) and velocities (7)
- Gripper width (1)
- Cube relative position (3, zeros if no cube)

## Action Space (7D)

- End-effector velocity: dx, dy, dz (3)
- End-effector angular velocity: droll, dpitch, dyaw (3)
- Gripper command (1)

## Development Notes

- Mock objects in SceneManager include `name` attribute (required by Isaac Sim's `world.scene.add()`)
- Tests mock Isaac Sim imports to run without GPU/simulator
- Isaac Sim primitives: `DynamicCuboid`, `VisualCuboid` from `isaacsim.core.api.objects`
