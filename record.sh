#!/bin/bash
# Record demonstrations using Franka keyboard teleoperation
#
# Usage:
#   ./record.sh                           # Record to demos/recording_TIMESTAMP.npz
#   ./record.sh --demo-path my_demo.npz   # Custom output path
#   ./record.sh --reward-mode sparse      # Use sparse rewards
#
# Recording Controls:
#   ` (backtick): Start/Stop recording
#   [ : Mark episode as SUCCESS and reset scene
#   ] : Mark episode as FAILURE and reset scene
#
# Recording auto-saves every 5 seconds and on exit (Esc)
#
# Robot controls work the same as run.sh (Tab to switch modes, WASD for movement)

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Create demos directory if it doesn't exist
mkdir -p demos

echo "Starting recording session..."
echo ""
echo "Recording Controls:"
echo "  \` = Start/Stop recording"
echo "  [ = Mark SUCCESS (finalize episode, reset scene)"
echo "  ] = Mark FAILURE (finalize episode, reset scene)"
echo ""
echo "Auto-saves every 5 seconds and on exit (Esc)"
echo ""
echo "Robot Controls: Tab to switch modes, WASD/QE for movement, Esc to exit"
echo ""

$ISAAC_PYTHON franka_keyboard_control.py --enable-recording "$@"
