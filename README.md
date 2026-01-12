# NeuralFeels-MuJoCo

Visuotactile perception for dexterous manipulation using neural implicit fields in MuJoCo simulation.

Implements a real-time 3D reconstruction system combining vision and simulated tactile sensing for in-hand object manipulation with an Allegro Hand. Based on [NeuralFeels](https://www.science.org/doi/10.1126/scirobotics.adl0628) (Science Robotics 2024).

## Demo

```bash
make video STEPS=100 OUTPUT=outputs/demo.mp4
make demo  # live visualization
```

## What This Does

1. Simulates an Allegro Hand manipulating objects in MuJoCo
2. Generates tactile depth maps from contact forces (4 fingertips, 32x32 each)
3. Fuses monocular visual depth with tactile measurements
4. Trains a neural SDF online to reconstruct object geometry
5. Tracks object pose using visual odometry and pose graph optimization

## Quick Start

```bash
git clone https://github.com/andomeder/neuralfeels-mujoco.git
cd neuralfeels-mujoco
make dev-setup
make demo
```

## Requirements

**Hardware**: Intel Arc B580 (tested), NVIDIA CUDA, or AMD ROCm. 8GB RAM.

**Arch Linux**:
```bash
sudo pacman -S suitesparse intel-compute-runtime level-zero-loader level-zero-headers
```

**Ubuntu/Debian**:
```bash
sudo apt-get install libsuitesparse-dev
```

## Installation

Uses mise for Python version management and uv for packages.

```bash
make dev-setup  # automatic
# or
make install    # just dependencies
```

Theseus (pose optimization) needs special handling:
```bash
pip install Cython scikit-sparse --no-build-isolation
pip install theseus-ai --no-build-isolation
```

Theseus runs on CPU only. This is expected.

## GPU Support

| GPU | Backend | Status |
|-----|---------|--------|
| Intel Arc B580 | XPU | Tested, PyTorch 2.9.0+xpu |
| NVIDIA | CUDA | Supported |
| AMD | ROCm | Supported |

## Usage

```bash
make collect-data EPISODES=10  # collect manipulation data
make train                     # run perception pipeline
make eval                      # compute metrics
make demo                      # live visualization
make video STEPS=200           # generate video
```

## Structure

```
envs/
  allegro_hand_env.py    # 16-DOF hand, Gymnasium interface
  tactile_sim.py         # contact forces to depth maps
  assets/                # MJCF models

perception/
  neural_sdf.py          # implicit surface network
  depth_fusion.py        # vision + tactile fusion
  depth_model.py         # DPT monocular depth
  pose_tracking.py       # Theseus pose graph
  pipeline.py            # orchestration
  metrics.py             # F-score, ADD-S

scripts/
  demo.py, train.py, eval.py, collect_data.py

tests/                   # 77 tests
```

## Technical Details

**Neural SDF**: Sinusoidal positional encoding (10 frequencies), 8-layer MLP with 256 hidden units, Softplus activation. Surface + free-space + Eikonal losses. Mesh extraction via marching cubes.

**Tactile Simulation**: MuJoCo contact forces projected to fingertip local frame, rendered as 32x32 depth maps, Gaussian blur (sigma=1.5).

**Pose Tracking**: ORB features, essential matrix decomposition, Theseus LM optimizer on 5-keyframe sliding window.

**Depth Fusion**: DPT monocular depth (weight 0.3) combined with tactile depth (weight 1.0) where contacts exist.

## Known Limitations

- SDF optimization in pose graph disabled (XPU/CPU device mixing issues with Theseus)
- Uses primitive sphere; YCB objects not yet integrated
- Open-loop rotation policy drops objects sometimes

## References

1. Suresh et al., "Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation", Science Robotics 2024
2. Ortiz et al., "iSDF: Real-Time Neural Signed Distance Fields for Robot Perception", RSS 2022
3. Wang et al., "TACTO: A Fast, Flexible and Open-source Simulator for High-Resolution Vision-based Tactile Sensors", RA-L 2022

## Development

```bash
make format  # black + ruff
make lint
make test    # 77 tests
make check   # all of the above
```

## License

MIT
