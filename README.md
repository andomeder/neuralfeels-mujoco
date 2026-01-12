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

Theseus (pose optimization) requires special handling:
```bash
uv pip install Cython scikit-sparse --no-build-isolation
uv pip install theseus-ai --no-build-isolation
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
make ablation STEPS=100        # run ablation study (vision/tactile/fusion)
make demo                      # live visualization
make video STEPS=200           # generate video
```

### Ablation Study

Compare perception modalities (vision-only, tactile-only, visuotactile fusion):

```bash
make ablation STEPS=100 OBJECTS='sphere box cylinder'
```

This generates:
- `outputs/ablation/metrics.json` - Raw F-scores for each modality and object
- `outputs/ablation/comparison.png` - Bar chart comparing F-scores
- `outputs/ablation/table.md` - Markdown table for results

**Ablation Results** (using monocular depth - see [Issue #1](https://github.com/andomeder/neuralfeels-mujoco/issues/1) for GT depth improvement):

| Modality | Sphere | Box | Cylinder | Average |
|----------|--------|-----|----------|---------|
| Vision-only | 0.0% | 0.0% | 0.0% | **0.0%** |
| Tactile-only | 0.0% | 17.0% | 30.3% | **15.8%** |
| Visuotactile | 0.0% | 0.0% | 2.0% | **0.7%** |

**Winner**: Tactile-only (15.8% average F-score)

Note: Tactile-only outperforms vision-based methods because the neural SDF uses a sphere prior that degrades when trained on noisy monocular depth. This highlights the value of high-confidence tactile measurements for shape reconstruction.

Run `make ablation` to regenerate metrics.

### Object Selection

The environment supports 7 different objects (MuJoCo primitives mimicking YCB shapes):

| Object | Shape | Description |
|--------|-------|-------------|
| `sphere` | Sphere | Tennis ball-like (default) |
| `box` | Box | Cracker box (6x10x2 cm) |
| `cylinder` | Cylinder | Soup can (7cm diameter, 10cm height) |
| `capsule` | Capsule | Banana-like elongated shape |
| `ellipsoid` | Ellipsoid | Apple-like asymmetric sphere |
| `mug` | Composite | Mug with handle |
| `hammer` | Composite | Tool with box head + cylinder handle |

```python
from envs.allegro_hand_env import AllegroHandEnv

# Single object
env = AllegroHandEnv(object_name="box")
obs, info = env.reset()

# Random object each episode
env = AllegroHandEnv(randomize_object=True)
for episode in range(10):
    obs, info = env.reset()
    print(f"Object: {info['object_name']}")
```

Generate object showcase:
```bash
python scripts/visualize_objects.py --output outputs/objects.png
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

tests/                   # 104 tests
```

## Technical Details

**Neural SDF**: Sinusoidal positional encoding (10 frequencies), 8-layer MLP with 256 hidden units, Softplus activation. Surface + free-space + Eikonal losses. Mesh extraction via marching cubes.

**Tactile Simulation**: MuJoCo contact forces projected to fingertip local frame, rendered as 32x32 depth maps, Gaussian blur (sigma=1.5).

**Pose Tracking**: ORB features, essential matrix decomposition, Theseus LM optimizer on 5-keyframe sliding window.

**Depth Fusion**: DPT monocular depth (weight 0.3) combined with tactile depth (weight 1.0) where contacts exist.

## Known Limitations

- SDF optimization in pose graph disabled (XPU/CPU device mixing issues with Theseus)
- Open-loop rotation policy drops objects sometimes
- Objects are MuJoCo primitives, not actual YCB meshes (sufficient for diversity demonstration)

## References

1. Suresh et al., "Neural feels with neural fields: Visuo-tactile perception for in-hand manipulation", Science Robotics 2024
2. Ortiz et al., "iSDF: Real-Time Neural Signed Distance Fields for Robot Perception", RSS 2022
3. Wang et al., "TACTO: A Fast, Flexible and Open-source Simulator for High-Resolution Vision-based Tactile Sensors", RA-L 2022

## Development

```bash
make format  # black + ruff
make lint
make test    # 104 tests
make check   # all of the above
```

## License

MIT
