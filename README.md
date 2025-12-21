# NeuralFeels-MuJoCo

Visuotactile perception for dexterous manipulation using neural fields in MuJoCo simulation.

## Quick Start

```bash
# Clone and enter project
git clone https://github.com/andomeder/neuralfeels-mujoco.git
cd neuralfeels-mujoco

# Install system dependencies (Arch Linux)
sudo pacman -S suitesparse intel-compute-runtime level-zero-loader level-zero-headers

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libsuitesparse-dev

# One-command setup
make dev-setup

# Or step by step
make install      # Install Python deps (auto-detects GPU)
make gpu-info     # Verify GPU detection
```

## System Requirements

### Arch Linux (Intel Arc GPU)

```bash
sudo pacman -S suitesparse intel-compute-runtime level-zero-loader level-zero-headers
```

### Ubuntu/Debian

```bash
sudo apt-get install libsuitesparse-dev
```

## Installation Notes

This project uses **mise** for version management and **uv** for fast package management.

### Theseus (Pose Optimization)

Theseus requires special installation with `--no-build-isolation` to find the existing PyTorch installation:

```bash
# Install prerequisites
pip install Cython scikit-sparse --no-build-isolation

# Install theseus (works with Intel XPU, runs on CPU for optimization)
pip install theseus-ai --no-build-isolation
```

**Note**: Theseus optimization runs on CPU even when using Intel XPU for neural networks. This is because Theseus doesn't have native XPU support, but the CPU backend works well for pose graph optimization.

### Supported GPUs

| GPU | Backend | Status |
|-----|---------|--------|
| Intel Arc B580 | XPU | Tested |
| NVIDIA (CUDA) | CUDA | Supported |
| AMD (ROCm) | ROCm | Supported |

## Project Structure

```
neuralfeels-mujoco/
├── envs/                  # MuJoCo environments
│   ├── allegro_hand_env.py
│   └── tactile_sim.py
├── perception/            # Neural perception modules
│   ├── neural_sdf.py
│   ├── depth_fusion.py
│   └── pose_tracking.py
├── src/utils/             # Utilities
│   └── gpu_utils.py
├── scripts/               # Entry points
├── configs/               # Hydra configs
├── datasets/              # Collected episodes
└── outputs/               # Checkpoints, videos
```

## Usage

```bash
make collect-data   # Collect demonstration episodes
make train          # Train neural SDF perception
make eval           # Evaluate on test set
make demo           # Run live visualization
make video          # Generate demo video
```

## Development

```bash
make format         # Format with black + ruff
make lint           # Lint check
make test           # Run tests
make check          # Format + lint + test
```

## License

MIT
