#!/bin/bash
# Convenience wrapper to run the Franka keyboard control using Isaac Sim's Python interpreter
#
# Usage:
#   ./run.sh                    # Run the application
#   ./run.sh [arguments]        # Run with custom arguments

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Run the application with all passed arguments
echo "Running Franka keyboard control with Isaac Sim's Python..."
$ISAAC_PYTHON franka_keyboard_control.py "$@"
