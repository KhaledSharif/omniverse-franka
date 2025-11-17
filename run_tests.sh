#!/bin/bash
# Convenience wrapper to run pytest tests using Isaac Sim's Python interpreter
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh -v                 # Run with verbose output
#   ./run_tests.sh -k test_name       # Run specific test
#   ./run_tests.sh --cov              # Run with coverage

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

$ISAAC_PYTHON -m pip install lark

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Check if pytest is installed
$ISAAC_PYTHON -c "import pytest" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "pytest not found in Isaac Sim's Python environment."
    echo "Installing pytest..."
    $ISAAC_PYTHON -m pip install pytest pytest-cov
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pytest"
        exit 1
    fi
    echo "pytest installed successfully!"
    echo ""
fi

# Run pytest with all passed arguments
echo "Running tests with Isaac Sim's Python..."
$ISAAC_PYTHON -m pytest test_franka_keyboard_control.py "$@"
