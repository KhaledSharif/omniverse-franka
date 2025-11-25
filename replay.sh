#!/bin/bash
# Replay and inspect recorded demonstrations
#
# Usage:
#   ./replay.sh demos/recording.npz              # Visual playback of all episodes
#   ./replay.sh demos/recording.npz --info       # Show statistics only (no simulator)
#   ./replay.sh demos/recording.npz --episode 0  # Replay specific episode
#   ./replay.sh demos/recording.npz --speed 0.5  # Slow motion playback
#   ./replay.sh demos/recording.npz --successful # Only replay successful episodes

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Check if demo file provided
if [ -z "$1" ]; then
    echo "Usage: ./replay.sh <demo_file.npz> [options]"
    echo ""
    echo "Options:"
    echo "  --info           Show demo statistics only (no visual playback)"
    echo "  --episode N      Replay specific episode index"
    echo "  --speed FACTOR   Playback speed multiplier (default: 1.0)"
    echo "  --successful     Only replay successful episodes"
    echo ""
    echo "Examples:"
    echo "  ./replay.sh demos/recording.npz --info"
    echo "  ./replay.sh demos/recording.npz --episode 0 --speed 0.5"
    exit 1
fi

$ISAAC_PYTHON replay_demo.py "$@"
