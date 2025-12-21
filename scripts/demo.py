"""Live demo and video generation for visuotactile perception.

This script provides two modes:
1. Live: Real-time visualization during manipulation
2. Video: Generate a demo video from collected or live episodes

Usage:
    python scripts/demo.py --mode live
    python scripts/demo.py --mode video --output outputs/videos/demo.mp4
    python scripts/demo.py --mode replay --episode datasets/episode_000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from envs.allegro_hand_env import AllegroHandEnv
from envs.tactile_sim import visualize_tactile
from perception.depth_fusion import (
    SE3,
    CameraIntrinsics,
    create_default_camera_intrinsics,
    visualize_depth_fusion,
)
from perception.metrics import f_score, sample_mesh_surface
from perception.neural_sdf import NeuralSDF, extract_mesh
from perception.pipeline import PerceptionConfig, VisuotactilePerception
from scripts.collect_data import RotationPolicy
from src.utils.gpu_utils import get_device


class DemoVisualizer:
    """Visualization manager for the perception demo."""

    def __init__(
        self,
        window_name: str = "NeuralFeels-MuJoCo Demo",
        display_size: tuple[int, int] = (1280, 720),
    ):
        """Initialize visualizer.

        Args:
            window_name: OpenCV window name
            display_size: Output display resolution (width, height)
        """
        self.window_name = window_name
        self.display_size = display_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create_composite_frame(
        self,
        rgb: np.ndarray,
        tactile: np.ndarray,
        depth_visual: Optional[np.ndarray] = None,
        depth_fused: Optional[np.ndarray] = None,
        mesh_render: Optional[np.ndarray] = None,
        metrics: Optional[dict] = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Create a composite visualization frame.

        Layout:
        +------------------+------------------+
        |       RGB        |    Tactile Grid  |
        +------------------+------------------+
        |   Depth (Fused)  |  Mesh Render     |
        +------------------+------------------+

        Args:
            rgb: RGB observation (H, W, 3)
            tactile: Tactile observations (4, 32, 32)
            depth_visual: Visual depth map (H, W)
            depth_fused: Fused depth map (H, W)
            mesh_render: Rendered mesh view (H, W, 3)
            metrics: Dict with metrics to display
            frame_idx: Current frame index

        Returns:
            Composite visualization frame
        """
        H, W = rgb.shape[:2]
        cell_size = 300  # Size of each grid cell

        # Create 2x2 grid
        composite = np.zeros((cell_size * 2, cell_size * 2, 3), dtype=np.uint8)

        # Top-left: RGB
        rgb_resized = cv2.resize(rgb, (cell_size, cell_size))
        # Convert RGB to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
        composite[0:cell_size, 0:cell_size] = rgb_bgr
        self._add_label(composite, "RGB Input", 5, 20)

        # Top-right: Tactile grid
        tactile_viz = visualize_tactile(tactile, ["Idx", "Mid", "Rng", "Thb"])
        tactile_resized = cv2.resize(tactile_viz, (cell_size, cell_size))
        composite[0:cell_size, cell_size : cell_size * 2] = tactile_resized
        self._add_label(composite, "Tactile Sensors", cell_size + 5, 20)

        # Bottom-left: Depth (fused or visual)
        if depth_fused is not None:
            depth_to_show = depth_fused
            depth_label = "Fused Depth"
        elif depth_visual is not None:
            depth_to_show = depth_visual
            depth_label = "Visual Depth"
        else:
            depth_to_show = np.zeros((H, W), dtype=np.float32)
            depth_label = "Depth (N/A)"

        depth_norm = (
            (depth_to_show - depth_to_show.min())
            / (depth_to_show.max() - depth_to_show.min() + 1e-8)
            * 255
        ).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        depth_resized = cv2.resize(depth_color, (cell_size, cell_size))
        composite[cell_size : cell_size * 2, 0:cell_size] = depth_resized
        self._add_label(composite, depth_label, 5, cell_size + 20)

        # Bottom-right: Mesh render or placeholder
        if mesh_render is not None:
            mesh_resized = cv2.resize(mesh_render, (cell_size, cell_size))
            if len(mesh_resized.shape) == 2:
                mesh_resized = cv2.cvtColor(mesh_resized, cv2.COLOR_GRAY2BGR)
            composite[cell_size : cell_size * 2, cell_size : cell_size * 2] = mesh_resized
            self._add_label(composite, "Neural SDF Mesh", cell_size + 5, cell_size + 20)
        else:
            # Placeholder with "Reconstructing..." text
            cv2.putText(
                composite,
                "Reconstructing...",
                (cell_size + 60, cell_size + cell_size // 2),
                self.font,
                0.7,
                (128, 128, 128),
                2,
            )
            self._add_label(composite, "Neural SDF Mesh", cell_size + 5, cell_size + 20)

        # Add metrics overlay
        if metrics:
            self._add_metrics_overlay(composite, metrics, frame_idx)

        # Add frame counter
        cv2.putText(
            composite,
            f"Frame: {frame_idx}",
            (composite.shape[1] - 120, composite.shape[0] - 10),
            self.font,
            0.5,
            (255, 255, 255),
            1,
        )

        return composite

    def _add_label(self, img: np.ndarray, text: str, x: int, y: int):
        """Add a label with background."""
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.5, 1)
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), self.font, 0.5, (255, 255, 255), 1)

    def _add_metrics_overlay(self, img: np.ndarray, metrics: dict, frame_idx: int):
        """Add metrics overlay to bottom of image."""
        y_offset = img.shape[0] - 40
        x_offset = 10

        metric_strs = []
        if "f_score" in metrics:
            metric_strs.append(f"F-score: {metrics['f_score']:.1%}")
        if "pose_drift" in metrics:
            metric_strs.append(f"Drift: {metrics['pose_drift']:.1f}mm")
        if "fps" in metrics:
            metric_strs.append(f"FPS: {metrics['fps']:.1f}")

        text = " | ".join(metric_strs)
        if text:
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.5, 1)
            cv2.rectangle(
                img,
                (x_offset - 2, y_offset - th - 2),
                (x_offset + tw + 2, y_offset + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(img, text, (x_offset, y_offset), self.font, 0.5, (0, 255, 0), 1)

    def render_mesh_simple(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        image_size: int = 300,
    ) -> np.ndarray:
        """Simple mesh rendering using orthographic projection.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            image_size: Output image size

        Returns:
            Rendered mesh image
        """
        if vertices is None or len(vertices) == 0:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)

        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Center and scale vertices
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        scale = 0.8 * image_size / (2 * np.abs(vertices_centered).max() + 1e-8)

        # Project to 2D (orthographic, XY plane)
        points_2d = vertices_centered[:, :2] * scale + image_size / 2
        points_2d = points_2d.astype(np.int32)

        # Draw edges
        for face in faces:
            for i in range(3):
                p1 = tuple(points_2d[face[i]])
                p2 = tuple(points_2d[face[(i + 1) % 3]])
                cv2.line(img, p1, p2, (0, 200, 100), 1)

        # Draw vertices
        for pt in points_2d:
            cv2.circle(img, tuple(pt), 2, (0, 255, 200), -1)

        return img

    def show(self, frame: np.ndarray) -> bool:
        """Display frame and check for quit.

        Returns:
            True if should continue, False if quit requested
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and key != 27  # q or ESC to quit

    def close(self):
        """Close visualization window."""
        cv2.destroyAllWindows()


def get_fingertip_poses(env: AllegroHandEnv) -> list[SE3]:
    """Extract fingertip SE3 poses from environment.

    Args:
        env: Allegro hand environment

    Returns:
        List of 4 SE3 poses for fingertips
    """
    import mujoco

    poses = []
    for finger_idx in range(4):
        body_name = f"fingertip_{finger_idx}"
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        pos = env.data.xpos[body_id].copy()
        mat = env.data.xmat[body_id].reshape(3, 3).copy()

        poses.append(SE3(rotation=mat, translation=pos))

    return poses


def get_camera_pose(env: AllegroHandEnv) -> SE3:
    """Extract camera SE3 pose from environment.

    Args:
        env: Allegro hand environment

    Returns:
        SE3 pose of camera
    """
    import mujoco

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")

    # Get camera position and orientation from MuJoCo
    # Camera position is stored in model.cam_pos
    pos = env.model.cam_pos[cam_id].copy()

    # Camera rotation matrix from cam_mat0
    mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()

    return SE3(rotation=mat, translation=pos)


def run_live_demo(
    max_steps: int = 200,
    use_perception: bool = True,
    save_video: Optional[str] = None,
):
    """Run live demo with real-time perception.

    Args:
        max_steps: Maximum steps per episode
        use_perception: Whether to run neural perception pipeline
        save_video: Optional path to save video
    """
    print("Initializing environment...")
    env = AllegroHandEnv(render_mode="rgb_array")
    policy = RotationPolicy()
    visualizer = DemoVisualizer()

    # Initialize perception pipeline
    perception = None
    if use_perception:
        print("Initializing perception pipeline...")
        print("  (This may take a moment to load DPT model)")
        try:
            config = PerceptionConfig(
                sdf_update_freq=5,
                mesh_resolution=64,  # Lower for speed
            )
            perception = VisuotactilePerception(config)
            camera_intrinsics = create_default_camera_intrinsics(
                width=env.IMG_SIZE,
                height=env.IMG_SIZE,
                fov_degrees=45.0,
            )
            perception.initialize(camera_intrinsics=camera_intrinsics)
            print("  Perception pipeline ready!")
        except Exception as e:
            print(f"  Warning: Could not initialize perception: {e}")
            print("  Running without neural perception...")
            perception = None

    # Video writer
    video_writer = None
    if save_video:
        Path(save_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (600, 600))

    print("\nStarting live demo...")
    print("Press 'q' or ESC to quit\n")

    obs, info = env.reset()
    fps_history = []
    running = True

    for step in range(max_steps):
        if not running:
            break

        t_start = time.time()

        # Get action from policy
        t = step / env.control_freq
        action = policy.get_action(t)
        action_normalized = (action - 0.5) * 2

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_normalized)

        # Process through perception
        depth_fused = None
        mesh_render = None
        metrics = {}

        if perception is not None:
            try:
                fingertip_poses = get_fingertip_poses(env)
                camera_pose = get_camera_pose(env)

                state = perception.process_frame(
                    rgb=obs["rgb"],
                    tactile=obs["tactile"],
                    fingertip_poses=fingertip_poses,
                    camera_pose=camera_pose,
                )

                # Get mesh for visualization
                verts, faces = perception.get_mesh()
                if verts is not None and len(verts) > 0:
                    mesh_render = visualizer.render_mesh_simple(verts, faces)
                    metrics["reconstructed"] = True

                metrics["loss"] = state.total_loss

            except Exception as e:
                print(f"Perception error at step {step}: {e}")

        # Calculate FPS
        dt = time.time() - t_start
        fps = 1.0 / (dt + 1e-8)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        metrics["fps"] = np.mean(fps_history)

        # Create visualization
        frame = visualizer.create_composite_frame(
            rgb=obs["rgb"],
            tactile=obs["tactile"],
            depth_fused=depth_fused,
            mesh_render=mesh_render,
            metrics=metrics,
            frame_idx=step,
        )

        # Show and check for quit
        running = visualizer.show(frame)

        # Save to video
        if video_writer is not None:
            video_writer.write(frame)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {save_video}")

    visualizer.close()
    env.close()

    if perception is not None:
        # Save final checkpoint
        checkpoint_path = "outputs/checkpoints/demo_final.pt"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        perception.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    print("\nDemo complete!")


def run_replay_demo(
    episode_dir: str,
    use_perception: bool = True,
    save_video: Optional[str] = None,
):
    """Replay a collected episode with perception.

    Args:
        episode_dir: Path to episode directory
        use_perception: Whether to run neural perception
        save_video: Optional path to save video
    """
    episode_path = Path(episode_dir)
    if not episode_path.exists():
        print(f"Error: Episode not found: {episode_dir}")
        return

    print(f"Loading episode from: {episode_dir}")

    # Load episode data
    rgb_files = sorted((episode_path / "rgb").glob("*.png"))
    tactile_files = sorted((episode_path / "tactile").glob("*.npy"))
    qpos = np.load(episode_path / "qpos.npy")

    with open(episode_path / "metadata.json") as f:
        metadata = json.load(f)

    print(f"Episode has {len(rgb_files)} frames")

    visualizer = DemoVisualizer()

    # Video writer
    video_writer = None
    if save_video:
        Path(save_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (600, 600))

    print("\nReplaying episode...")
    print("Press 'q' or ESC to quit\n")

    running = True
    for idx, (rgb_file, tactile_file) in enumerate(zip(rgb_files, tactile_files)):
        if not running:
            break

        # Load frame data
        rgb = cv2.imread(str(rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tactile = np.load(tactile_file)

        # Create visualization (simplified without perception)
        frame = visualizer.create_composite_frame(
            rgb=rgb,
            tactile=tactile,
            frame_idx=idx,
        )

        running = visualizer.show(frame)

        if video_writer is not None:
            video_writer.write(frame)

        # Control playback speed
        time.sleep(0.05)

    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {save_video}")

    visualizer.close()
    print("\nReplay complete!")


def main():
    parser = argparse.ArgumentParser(description="NeuralFeels-MuJoCo Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "video", "replay"],
        help="Demo mode: live, video, or replay",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/demo.mp4",
        help="Output video path (for video mode)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="datasets/episode_000",
        help="Episode directory (for replay mode)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="Disable neural perception pipeline",
    )
    args = parser.parse_args()

    if args.mode == "live":
        run_live_demo(
            max_steps=args.steps,
            use_perception=not args.no_perception,
            save_video=None,
        )
    elif args.mode == "video":
        run_live_demo(
            max_steps=args.steps,
            use_perception=not args.no_perception,
            save_video=args.output,
        )
    elif args.mode == "replay":
        run_replay_demo(
            episode_dir=args.episode,
            use_perception=not args.no_perception,
            save_video=args.output if args.output != "outputs/videos/demo.mp4" else None,
        )


if __name__ == "__main__":
    main()
