#!/bin/bash
# Train behavioral cloning model using recorded demonstrations
#
# Usage:
#   ./sb3.sh demos/recording.npz                        # Train with defaults
#   ./sb3.sh demos/recording.npz --epochs 100           # Custom epochs
#   ./sb3.sh demos/recording.npz --batch-size 128       # Custom batch size
#   ./sb3.sh demos/recording.npz --lr 0.001             # Custom learning rate
#   ./sb3.sh demos/recording.npz --output model.pt      # Custom output path
#   ./sb3.sh demos/recording.npz --successful-only      # Train only on successful demos

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Check if demo file provided
if [ -z "$1" ]; then
    echo "Usage: ./sb3.sh <demo_file.npz> [options]"
    echo ""
    echo "Options:"
    echo "  --epochs N          Training epochs (default: 50)"
    echo "  --batch-size N      Batch size (default: 64)"
    echo "  --lr RATE           Learning rate (default: 1e-4)"
    echo "  --output PATH       Output model path (default: models/bc_policy.pt)"
    echo "  --successful-only   Only train on successful episodes"
    echo ""
    echo "Examples:"
    echo "  ./sb3.sh demos/recording.npz"
    echo "  ./sb3.sh demos/recording.npz --epochs 100 --successful-only"
    echo "  ./sb3.sh demos/recording.npz --output models/expert.pt"
    exit 1
fi

# Create models directory
mkdir -p models

echo "Training behavioral cloning model..."
$ISAAC_PYTHON train_bc.py "$@"
