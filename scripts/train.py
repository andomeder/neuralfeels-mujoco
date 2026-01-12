"""Training script for visuotactile perception pipeline.

Processes collected episodes through the perception pipeline to train
the neural SDF and evaluate reconstruction quality.

Usage:
    python scripts/train.py --episodes datasets --output outputs
    python scripts/train.py --episodes datasets/episode_000 --online
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from perception.depth_fusion import (
    SE3,
    create_default_camera_intrinsics,
)
from perception.neural_sdf import NeuralSDF, extract_mesh, sdf_loss
from perception.pipeline import PerceptionConfig, VisuotactilePerception
from src.utils.gpu_utils import get_device


def load_episode(episode_dir: Path) -> dict:
    """Load an episode from disk.

    Args:
        episode_dir: Path to episode directory

    Returns:
        Dictionary with episode data
    """
    rgb_files = sorted((episode_dir / "rgb").glob("*.png"))
    tactile_files = sorted((episode_dir / "tactile").glob("*.npy"))

    # Load arrays
    qpos = np.load(episode_dir / "qpos.npy")
    qvel = np.load(episode_dir / "qvel.npy")
    object_pos = np.load(episode_dir / "object_pos.npy")
    object_quat = np.load(episode_dir / "object_quat.npy")

    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load RGB and tactile frames
    rgb_frames = []
    tactile_frames = []
    for rgb_file, tactile_file in zip(rgb_files, tactile_files):
        rgb = cv2.imread(str(rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb)
        tactile_frames.append(np.load(tactile_file))

    return {
        "rgb": np.stack(rgb_frames),
        "tactile": np.stack(tactile_frames),
        "qpos": qpos,
        "qvel": qvel,
        "object_pos": object_pos,
        "object_quat": object_quat,
        "metadata": metadata,
        "num_frames": len(rgb_frames),
    }


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ],
        dtype=np.float32,
    )


def create_dummy_fingertip_poses(qpos: np.ndarray) -> list[SE3]:
    """Create approximate fingertip poses from joint positions.

    This is a simplified approximation. In practice, you would use
    forward kinematics from the robot model.

    Args:
        qpos: Joint positions (16,)

    Returns:
        List of 4 SE3 poses
    """
    # Approximate fingertip positions relative to palm
    # These are rough estimates based on Allegro hand geometry
    base_positions = np.array(
        [
            [0.0, 0.0, 0.12],  # Index
            [0.0, 0.03, 0.12],  # Middle
            [0.0, 0.06, 0.12],  # Ring
            [0.05, -0.02, 0.08],  # Thumb
        ],
        dtype=np.float32,
    )

    poses = []
    for i in range(4):
        # Simple identity rotation
        R = np.eye(3, dtype=np.float32)
        t = base_positions[i]
        poses.append(SE3(rotation=R, translation=t))

    return poses


def create_camera_pose() -> SE3:
    """Create camera pose (fixed camera looking at hand).

    Returns:
        SE3 camera pose
    """
    # Camera positioned at (0.8, 0, 0.5) looking at origin
    R = np.array(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    t = np.array([0.8, 0, 0.5], dtype=np.float32)
    return SE3(rotation=R, translation=t)


def train_on_episode(
    episode_data: dict,
    perception: VisuotactilePerception,
    verbose: bool = True,
) -> dict:
    """Train perception on a single episode.

    Args:
        episode_data: Loaded episode dictionary
        perception: Perception pipeline
        verbose: Print progress

    Returns:
        Training metrics dictionary
    """
    num_frames = episode_data["num_frames"]
    camera_pose = create_camera_pose()

    losses = []
    frame_times = []

    for frame_idx in range(num_frames):
        t_start = time.time()

        rgb = episode_data["rgb"][frame_idx]
        tactile = episode_data["tactile"][frame_idx]
        qpos = episode_data["qpos"][frame_idx]

        # Create fingertip poses
        fingertip_poses = create_dummy_fingertip_poses(qpos)

        # Process frame
        state = perception.process_frame(
            rgb=rgb,
            tactile=tactile,
            fingertip_poses=fingertip_poses,
            camera_pose=camera_pose,
        )

        losses.append(state.total_loss)
        frame_times.append(time.time() - t_start)

        if verbose and frame_idx % 20 == 0:
            avg_fps = 1.0 / (np.mean(frame_times[-20:]) + 1e-8)
            print(
                f"  Frame {frame_idx}/{num_frames}, loss={state.total_loss:.4f}, FPS={avg_fps:.1f}"
            )

    return {
        "avg_loss": np.mean(losses),
        "final_loss": losses[-1] if losses else 0.0,
        "avg_fps": 1.0 / (np.mean(frame_times) + 1e-8),
        "num_frames": num_frames,
        "losses": losses,
    }


def train_offline(
    episodes_dir: Path,
    output_dir: Path,
    max_episodes: int = 10,
    config: Optional[PerceptionConfig] = None,
):
    """Train perception on collected episodes (offline).

    Args:
        episodes_dir: Directory containing episodes
        output_dir: Output directory for checkpoints and metrics
        max_episodes: Maximum number of episodes to process
        config: Perception configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Find episodes
    if episodes_dir.is_dir() and (episodes_dir / "rgb").exists():
        # Single episode
        episode_dirs = [episodes_dir]
    else:
        # Multiple episodes
        episode_dirs = sorted(episodes_dir.glob("episode_*"))

    if not episode_dirs:
        print(f"No episodes found in {episodes_dir}")
        return

    episode_dirs = episode_dirs[:max_episodes]
    print(f"Found {len(episode_dirs)} episodes to process")

    # Initialize perception
    print("\nInitializing perception pipeline...")
    if config is None:
        config = PerceptionConfig(
            sdf_update_freq=5,
            mesh_resolution=128,
        )

    perception = VisuotactilePerception(config)
    camera_intrinsics = create_default_camera_intrinsics(
        width=224,
        height=224,
        fov_degrees=45.0,
    )
    perception.initialize(camera_intrinsics=camera_intrinsics)
    print("Perception pipeline ready!\n")

    all_metrics = []

    for ep_idx, episode_dir in enumerate(episode_dirs):
        print(f"\n{'=' * 60}")
        print(f"Processing Episode {ep_idx + 1}/{len(episode_dirs)}: {episode_dir.name}")
        print("=" * 60)

        # Load episode
        try:
            episode_data = load_episode(episode_dir)
            print(f"Loaded {episode_data['num_frames']} frames")
        except Exception as e:
            print(f"Error loading episode: {e}")
            continue

        # Train on episode
        metrics = train_on_episode(episode_data, perception, verbose=True)
        metrics["episode"] = episode_dir.name
        all_metrics.append(metrics)

        print(f"\nEpisode {ep_idx + 1} complete:")
        print(f"  Average loss: {metrics['avg_loss']:.4f}")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Average FPS: {metrics['avg_fps']:.1f}")

        # Save checkpoint after each episode
        checkpoint_path = checkpoint_dir / f"checkpoint_ep{ep_idx:03d}.pt"
        perception.save_checkpoint(str(checkpoint_path))

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "final.pt"
    perception.save_checkpoint(str(final_checkpoint))
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")

    # Extract and save final mesh
    verts, faces = perception.get_mesh()
    if verts is not None:
        mesh_path = output_dir / "final_mesh.npz"
        np.savez(mesh_path, vertices=verts, faces=faces)
        print(f"Final mesh saved to: {mesh_path}")
        print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    # Save metrics summary
    summary = {
        "num_episodes": len(all_metrics),
        "total_frames": sum(m["num_frames"] for m in all_metrics),
        "avg_loss_overall": np.mean([m["avg_loss"] for m in all_metrics]),
        "avg_fps_overall": np.mean([m["avg_fps"] for m in all_metrics]),
        "episodes": all_metrics,
    }

    metrics_path = metrics_dir / "training_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(
            summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x
        )
    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Episodes processed: {summary['num_episodes']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Average loss: {summary['avg_loss_overall']:.4f}")
    print(f"Average FPS: {summary['avg_fps_overall']:.1f}")


def train_sdf_only(
    episodes_dir: Path,
    output_dir: Path,
    num_iterations: int = 1000,
    batch_size: int = 4096,
):
    """Train just the neural SDF on point cloud data.

    This is a simpler training mode that only trains the SDF
    without running the full perception pipeline.

    Args:
        episodes_dir: Directory containing episodes
        output_dir: Output directory
        num_iterations: Number of training iterations
        batch_size: Points per batch
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Training on device: {device}")

    # Find and load all point clouds from episodes
    episode_dirs = sorted(episodes_dir.glob("episode_*"))
    if not episode_dirs:
        print("No episodes found")
        return

    print(f"Loading point data from {len(episode_dirs)} episodes...")

    all_points = []
    for episode_dir in episode_dirs[:5]:  # Limit to 5 episodes for speed
        # Create approximate surface points from object positions
        object_pos = np.load(episode_dir / "object_pos.npy")

        # Generate points around object trajectory
        for pos in object_pos[::10]:  # Sample every 10th frame
            # Generate points on a sphere around the object
            theta = np.random.uniform(0, 2 * np.pi, 100)
            phi = np.random.uniform(0, np.pi, 100)
            r = 0.03  # Object radius approximately

            x = pos[0] + r * np.sin(phi) * np.cos(theta)
            y = pos[1] + r * np.sin(phi) * np.sin(theta)
            z = pos[2] + r * np.cos(phi)

            points = np.stack([x, y, z], axis=-1).astype(np.float32)
            all_points.append(points)

    if not all_points:
        print("No point data found")
        return

    surface_points = np.concatenate(all_points, axis=0)
    print(f"Collected {len(surface_points)} surface points")

    # Normalize to unit cube
    center = surface_points.mean(axis=0)
    scale = np.abs(surface_points - center).max()
    surface_points = (surface_points - center) / (scale + 1e-8)

    # Initialize neural SDF
    model = NeuralSDF(hidden_dim=256, num_layers=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    print(f"\nTraining neural SDF for {num_iterations} iterations...")

    for iteration in range(num_iterations):
        # Sample batch of surface points
        indices = np.random.choice(
            len(surface_points), min(batch_size // 2, len(surface_points)), replace=False
        )
        batch_surface = torch.tensor(surface_points[indices], device=device)

        # Sample free-space points
        free_points = np.random.uniform(-1, 1, (batch_size // 2, 3)).astype(np.float32)
        batch_free = torch.tensor(free_points, device=device)

        # Compute approximate SDF for free points
        free_sdf = torch.tensor(
            np.linalg.norm(free_points[:, None, :] - surface_points[None, :, :], axis=-1).min(
                axis=1
            ),
            device=device,
        )

        # Training step
        optimizer.zero_grad()
        losses = sdf_loss(model, batch_surface, batch_free, free_sdf)
        losses["total"].backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}/{num_iterations}: "
                f"total={losses['total'].item():.4f}, "
                f"surface={losses['surface'].item():.4f}, "
                f"eikonal={losses['eikonal'].item():.4f}"
            )

    # Extract and save mesh
    print("\nExtracting mesh...")
    verts, faces, _ = extract_mesh(model, resolution=128, bounds=(-1, 1), device=device)

    if verts is not None:
        # Rescale to original coordinates
        verts = verts * scale + center

        mesh_path = output_dir / "sdf_mesh.npz"
        np.savez(mesh_path, vertices=verts, faces=faces)
        print(f"Mesh saved to: {mesh_path}")
        print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    # Save model
    model_path = output_dir / "sdf_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train visuotactile perception")
    parser.add_argument(
        "--episodes",
        type=str,
        default="datasets",
        help="Path to episodes directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and metrics",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=10,
        help="Maximum episodes to process",
    )
    parser.add_argument(
        "--sdf-only",
        action="store_true",
        help="Train only the neural SDF (simplified mode)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Training iterations (for --sdf-only mode)",
    )
    args = parser.parse_args()

    episodes_dir = Path(args.episodes)
    output_dir = Path(args.output)

    if args.sdf_only:
        train_sdf_only(
            episodes_dir=episodes_dir,
            output_dir=output_dir,
            num_iterations=args.iterations,
        )
    else:
        train_offline(
            episodes_dir=episodes_dir,
            output_dir=output_dir,
            max_episodes=args.max_episodes,
        )


if __name__ == "__main__":
    main()
