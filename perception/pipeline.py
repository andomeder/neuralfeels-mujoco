"""Integrated visuotactile perception pipeline.

Combines all perception components for online 3D reconstruction and pose tracking:
- Neural SDF for object shape representation
- Depth fusion for visuotactile depth estimation
- Pose tracking for 6-DOF object pose estimation

This is the main entry point for the perception system during manipulation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from perception.depth_fusion import (
    SE3,
    CameraIntrinsics,
    VisuotactileDepthFusion,
    create_default_camera_intrinsics,
    unproject_depth_to_points,
)
from perception.neural_sdf import NeuralSDF, extract_mesh, sdf_loss
from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig
from src.utils.gpu_utils import get_device


@dataclass
class PerceptionConfig:
    """Configuration for the perception pipeline."""

    # Neural SDF
    sdf_hidden_dim: int = 256
    sdf_num_layers: int = 8
    sdf_num_frequencies: int = 10
    sdf_learning_rate: float = 1e-4
    sdf_update_freq: int = 5  # Update SDF every N frames

    # Depth fusion
    visual_weight: float = 0.3
    tactile_weight: float = 1.0
    depth_model: str = "Intel/dpt-hybrid-midas"

    # Pose tracking
    pose_window_size: int = 5
    pose_sdf_weight: float = 0.01
    pose_reg_weight: float = 0.01

    # Keyframe management
    keyframe_min_interval: int = 5  # Minimum frames between keyframes
    keyframe_loss_threshold: float = 0.01  # Add keyframe if loss > threshold

    # Mesh extraction
    mesh_resolution: int = 128
    mesh_bounds: tuple = (-0.3, 0.3)  # Covers hand workspace around origin

    sdf_workspace_radius: float = 0.5
    sdf_min_world_z: float = -0.1

    # Device
    device: str = "auto"


@dataclass
class Keyframe:
    """A keyframe storing observations for replay."""

    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    tactile: np.ndarray
    points: np.ndarray  # 3D points in world frame
    pose: np.ndarray  # 4x4 object pose
    loss: float = 0.0


@dataclass
class PerceptionState:
    """Current state of the perception system."""

    frame_count: int = 0
    current_pose: Optional[np.ndarray] = None
    mesh_vertices: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None
    total_loss: float = 0.0
    keyframes: list = field(default_factory=list)


class VisuotactilePerception:
    """Main perception pipeline for visuotactile 3D reconstruction.

    Combines:
    - Neural SDF for implicit surface representation
    - Depth fusion for combining visual + tactile depth
    - Pose tracking for object pose estimation

    Usage:
        pipeline = VisuotactilePerception(config)
        pipeline.initialize()

        for frame in episode:
            state = pipeline.process_frame(
                rgb=frame['rgb'],
                tactile=frame['tactile'],
                fingertip_poses=frame['fingertip_poses'],
                camera_pose=frame['camera_pose']
            )
            # state contains current pose and mesh
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        """Initialize perception pipeline.

        Args:
            config: Perception configuration (uses defaults if None)
        """
        self.config = config if config is not None else PerceptionConfig()

        # Determine device
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(self.config.device)

        # Components (initialized lazily)
        self.neural_sdf: Optional[NeuralSDF] = None
        self.depth_fusion: Optional[VisuotactileDepthFusion] = None
        self.pose_tracker: Optional[PoseGraphOptimizer] = None
        self.optimizer: Optional[torch.optim.Adam] = None

        # Camera intrinsics (set during initialization)
        self.camera_intrinsics: Optional[CameraIntrinsics] = None

        # State
        self.state = PerceptionState()
        self._initialized = False

    def initialize(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        initial_pose: Optional[np.ndarray] = None,
    ):
        """Initialize all perception components.

        Args:
            camera_intrinsics: Camera intrinsic parameters
            initial_pose: Initial object pose (4x4 matrix)
        """
        # Camera intrinsics
        if camera_intrinsics is None:
            self.camera_intrinsics = create_default_camera_intrinsics()
        else:
            self.camera_intrinsics = camera_intrinsics

        # Neural SDF
        self.neural_sdf = NeuralSDF(
            hidden_dim=self.config.sdf_hidden_dim,
            num_layers=self.config.sdf_num_layers,
            num_frequencies=self.config.sdf_num_frequencies,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.neural_sdf.parameters(),
            lr=self.config.sdf_learning_rate,
            weight_decay=1e-6,
        )

        # Depth fusion (load DPT model)
        try:
            self.depth_fusion = VisuotactileDepthFusion(
                camera_intrinsics=self.camera_intrinsics,
                visual_weight=self.config.visual_weight,
                tactile_weight=self.config.tactile_weight,
                depth_model=self.config.depth_model,
                device=self.device,
            )
        except Exception as e:
            print(f"Warning: Could not load DPT model: {e}")
            print("Using depth fusion without visual depth estimation")
            self.depth_fusion = VisuotactileDepthFusion(
                camera_intrinsics=self.camera_intrinsics,
                visual_weight=self.config.visual_weight,
                tactile_weight=self.config.tactile_weight,
                depth_model=None,
            )

        # Pose tracker
        pose_config = PoseTrackingConfig(
            window_size=self.config.pose_window_size,
            sdf_weight=self.config.pose_sdf_weight,
            reg_weight=self.config.pose_reg_weight,
            device="cpu",  # Theseus runs on CPU
        )

        self.pose_tracker = PoseGraphOptimizer(
            config=pose_config,
            sdf_model=self.neural_sdf,
            camera_K=self.camera_intrinsics.to_matrix(),
        )

        # Initialize pose
        if initial_pose is None:
            initial_pose = np.eye(4, dtype=np.float32)
        self.pose_tracker.initialize(initial_pose)

        # Reset state
        self.state = PerceptionState()
        self.state.current_pose = initial_pose

        self._initialized = True

    def process_frame(
        self,
        rgb: np.ndarray,
        tactile: np.ndarray,
        fingertip_poses: list[SE3],
        camera_pose: SE3,
        depth: Optional[np.ndarray] = None,
        depth_scale: Optional[float] = None,
    ) -> PerceptionState:
        """Process a single frame through the perception pipeline.

        Args:
            rgb: RGB image (H, W, 3) uint8
            tactile: Tactile depth maps (4, 32, 32) float32
            fingertip_poses: SE3 poses of 4 fingertips in world frame
            camera_pose: SE3 pose of camera in world frame
            depth: Optional ground truth depth (H, W) for debugging
            depth_scale: Optional scale factor for DPT depth to metric

        Returns:
            Updated PerceptionState with current pose and mesh
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        assert self.depth_fusion is not None
        assert self.camera_intrinsics is not None
        assert self.pose_tracker is not None
        assert self.neural_sdf is not None
        assert self.optimizer is not None

        depth_fusion = self.depth_fusion
        camera_intrinsics = self.camera_intrinsics
        pose_tracker = self.pose_tracker

        self.state.frame_count += 1

        # 1. Depth fusion: combine visual and tactile depth
        if depth is None and depth_fusion.depth_estimator is not None:
            visual_depth = depth_fusion.estimate_visual_depth(rgb)
            if depth_scale is not None and depth_scale > 0:
                valid_mask = visual_depth > 0
                if valid_mask.any():
                    median = np.median(visual_depth[valid_mask])
                    if median > 0:
                        visual_depth = visual_depth * (depth_scale / median)
        else:
            visual_depth = (
                depth
                if depth is not None
                else np.zeros(
                    (camera_intrinsics.height, camera_intrinsics.width),
                    dtype=np.float32,
                )
            )

        # Split tactile into list
        tactile_list = [tactile[i] for i in range(tactile.shape[0])]

        fused_depth, confidence = depth_fusion.fuse(
            visual_depth,
            tactile_list,
            fingertip_poses,
            camera_pose,
        )

        # 2. Unproject depth to 3D points
        points_cam = unproject_depth_to_points(fused_depth, camera_intrinsics)

        # Transform points to world frame
        cam_T = camera_pose.to_matrix()
        points_world = self._transform_points(points_cam, cam_T)

        # 3. Pose tracking: update object pose estimate
        self.state.current_pose = pose_tracker.add_frame(
            rgb=rgb,
            depth=fused_depth,
            points=points_world,
            timestamp=self.state.frame_count,
        )

        # 4. Neural SDF training (every N frames)
        if self.state.frame_count % self.config.sdf_update_freq == 0:
            points_world_sdf = self._filter_points_for_sdf(points_world, fingertip_poses)
            loss = self._train_sdf_step(points_world_sdf)
            self.state.total_loss = loss

            # Check if we should add a keyframe
            self._maybe_add_keyframe(rgb, fused_depth, tactile, points_world, loss)

        # 5. Extract mesh periodically
        if self.state.frame_count % (self.config.sdf_update_freq * 4) == 0:
            self._extract_mesh()

        return self.state

    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform 3D points by a 4x4 matrix.

        Args:
            points: (N, 3) points
            transform: (4, 4) transformation matrix

        Returns:
            Transformed points (N, 3)
        """
        if len(points) == 0:
            return points

        # Convert to homogeneous
        ones = np.ones((len(points), 1), dtype=np.float32)
        points_homo = np.hstack([points, ones])

        # Transform
        points_transformed = (transform @ points_homo.T).T

        return points_transformed[:, :3]

    def _filter_points_for_sdf(
        self, points_world: np.ndarray, fingertip_poses: list[SE3]
    ) -> np.ndarray:
        if len(points_world) == 0:
            return points_world

        return points_world

    def _train_sdf_step(self, points_world: np.ndarray, num_iterations: int = 5) -> float:
        """Run SDF training steps.

        Args:
            points_world: Surface points in world frame (N, 3)
            num_iterations: Number of training iterations per call

        Returns:
            Final training loss
        """
        if len(points_world) < 100:
            return 0.0

        if self.neural_sdf is None or self.optimizer is None:
            raise RuntimeError("SDF not initialized. Call initialize() first.")

        neural_sdf = self.neural_sdf
        optimizer = self.optimizer

        neural_sdf.train()

        # Sample surface points
        num_surface = min(2048, len(points_world))
        indices = np.random.choice(len(points_world), num_surface, replace=False)
        surface_points = points_world[indices]

        # Transform to object frame using current pose
        if self.state.current_pose is not None:
            obj_T_inv = np.linalg.inv(self.state.current_pose)
            surface_points = self._transform_points(surface_points, obj_T_inv)

        # Generate free-space points (random in bounding box)
        num_free = num_surface
        bounds = self.config.mesh_bounds
        free_points = np.random.uniform(bounds[0], bounds[1], size=(num_free, 3)).astype(np.float32)

        # Compute approximate SDF for free points (distance to nearest surface)
        free_sdf = np.linalg.norm(
            free_points[:, None, :] - surface_points[None, :, :], axis=-1
        ).min(axis=1)

        # Convert to tensors
        surface_tensor = torch.tensor(surface_points, device=self.device)
        free_tensor = torch.tensor(free_points, device=self.device)
        free_sdf_tensor = torch.tensor(free_sdf, device=self.device)

        total_loss = 0.0
        for _ in range(num_iterations):
            optimizer.zero_grad()
            losses = sdf_loss(
                neural_sdf,
                surface_tensor,
                free_tensor,
                free_sdf_tensor,
            )

            losses["total"].backward()
            optimizer.step()
            total_loss = losses["total"].item()

        return total_loss

    def _maybe_add_keyframe(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tactile: np.ndarray,
        points: np.ndarray,
        loss: float,
    ):
        """Add a keyframe if criteria are met.

        Args:
            rgb: RGB image
            depth: Fused depth map
            tactile: Tactile observations
            points: 3D points
            loss: Current training loss
        """
        # Check minimum interval
        if len(self.state.keyframes) > 0:
            last_kf = self.state.keyframes[-1]
            if self.state.frame_count - last_kf.frame_id < self.config.keyframe_min_interval:
                return

        # Check loss threshold (add keyframe if loss is high = new information)
        if loss > self.config.keyframe_loss_threshold or len(self.state.keyframes) == 0:
            keyframe = Keyframe(
                frame_id=self.state.frame_count,
                rgb=rgb.copy(),
                depth=depth.copy(),
                tactile=tactile.copy(),
                points=points.copy(),
                pose=(
                    self.state.current_pose.copy()
                    if self.state.current_pose is not None
                    else np.eye(4)
                ),
                loss=loss,
            )
            self.state.keyframes.append(keyframe)

            # Limit keyframe count
            max_keyframes = 50
            if len(self.state.keyframes) > max_keyframes:
                self.state.keyframes.pop(0)

    def _extract_mesh(self):
        """Extract mesh from neural SDF."""
        if self.neural_sdf is None:
            return

        try:
            verts, faces, normals = extract_mesh(
                self.neural_sdf,
                resolution=self.config.mesh_resolution,
                bounds=self.config.mesh_bounds,
                device=self.device,
            )

            if verts is not None:
                self.state.mesh_vertices = verts
                self.state.mesh_faces = faces
        except Exception as e:
            print(f"Mesh extraction failed: {e}")

    def get_mesh(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the current reconstructed mesh.

        Returns:
            Tuple of (vertices, faces) or (None, None) if not available
        """
        return self.state.mesh_vertices, self.state.mesh_faces

    def get_pose(self) -> Optional[np.ndarray]:
        """Get the current object pose.

        Returns:
            4x4 pose matrix or None
        """
        return self.state.current_pose

    def save_checkpoint(self, path: str):
        """Save perception state to checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "config": self.config,
            "frame_count": self.state.frame_count,
            "current_pose": self.state.current_pose,
            "mesh_vertices": self.state.mesh_vertices,
            "mesh_faces": self.state.mesh_faces,
            "neural_sdf_state": self.neural_sdf.state_dict() if self.neural_sdf else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load perception state from checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint["config"]
        self.initialize()

        self.state.frame_count = checkpoint["frame_count"]
        self.state.current_pose = checkpoint["current_pose"]
        self.state.mesh_vertices = checkpoint["mesh_vertices"]
        self.state.mesh_faces = checkpoint["mesh_faces"]

        if checkpoint["neural_sdf_state"] and self.neural_sdf:
            self.neural_sdf.load_state_dict(checkpoint["neural_sdf_state"])
        if checkpoint["optimizer_state"] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])


def create_perception_pipeline(
    camera_fov: float = 45.0,
    image_size: int = 224,
    device: str = "auto",
) -> VisuotactilePerception:
    """Factory function to create a perception pipeline.

    Args:
        camera_fov: Camera field of view in degrees
        image_size: Image width/height
        device: Device for computation

    Returns:
        Initialized VisuotactilePerception pipeline
    """
    config = PerceptionConfig(device=device)

    camera_intrinsics = create_default_camera_intrinsics(
        width=image_size,
        height=image_size,
        fov_degrees=camera_fov,
    )

    pipeline = VisuotactilePerception(config)
    pipeline.initialize(camera_intrinsics=camera_intrinsics)

    return pipeline
