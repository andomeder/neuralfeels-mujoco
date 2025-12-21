"""Visuotactile depth fusion module.

Combines monocular visual depth estimation with high-confidence tactile depth
measurements from fingertip sensors. Tactile provides accurate local depth at
contact points, while vision fills in the rest of the scene.

References:
- NeuralFeels (Science Robotics 2024): Touch as local vision
- TACTO: Tactile simulation as perspective camera model
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.gpu_utils import get_device


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )


@dataclass
class SE3:
    """SE(3) pose representation (rotation + translation)."""

    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def inverse(self) -> "SE3":
        """Compute inverse transformation."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return SE3(rotation=R_inv, translation=t_inv)


class MonocularDepthEstimator:
    """Monocular depth estimation using DPT (Dense Prediction Transformer).

    Uses Intel DPT-Large model by default, can switch to DPT-Hybrid for speed.
    """

    def __init__(
        self,
        model_name: str = "Intel/dpt-hybrid-midas",
        device: Optional[torch.device] = None,
    ):
        """Initialize depth estimator.

        Args:
            model_name: HuggingFace model name. Options:
                - "Intel/dpt-large" (more accurate, slower)
                - "Intel/dpt-hybrid-midas" (balanced, default)
            device: Torch device for inference
        """
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        self.device = device if device is not None else get_device()
        self.model_name = model_name

        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Predict depth from RGB image.

        Args:
            rgb: RGB image (H, W, 3) uint8

        Returns:
            depth: Depth map (H, W) float32, normalized to [0, 1]
        """
        # Preprocess
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Resize to original resolution
        depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize to [0, 1]
        depth = depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth.astype(np.float32)


class TactileProjector:
    """Projects tactile depth maps from fingertip frame to camera frame.

    Treats tactile sensors as small perspective cameras following TACTO approach.
    """

    TACTILE_RES = 32  # Tactile sensor resolution
    SENSOR_SIZE = 0.02  # Physical sensor size in meters

    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        num_fingers: int = 4,
    ):
        """Initialize tactile projector.

        Args:
            camera_intrinsics: Camera intrinsic parameters
            num_fingers: Number of fingertip sensors
        """
        self.camera_K = camera_intrinsics
        self.num_fingers = num_fingers

    def project_tactile_to_camera(
        self,
        tactile_depth: np.ndarray,
        fingertip_pose: SE3,
        camera_pose: SE3,
        depth_scale: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a single tactile depth map to camera image plane.

        Args:
            tactile_depth: Tactile depth map (32, 32) in [0, 1]
            fingertip_pose: Fingertip pose in world frame
            camera_pose: Camera pose in world frame
            depth_scale: Scale factor to convert tactile depth to meters

        Returns:
            Tuple of:
                - pixel_coords: (N, 2) pixel coordinates in camera image
                - depths: (N,) depth values in camera frame
                - mask: (H, W) binary mask of projected tactile region
        """
        # Generate 3D points from tactile depth map
        # Tactile sensor coordinate system: x-right, y-up, z-out (towards object)
        u = np.linspace(-0.5, 0.5, self.TACTILE_RES) * self.SENSOR_SIZE
        v = np.linspace(-0.5, 0.5, self.TACTILE_RES) * self.SENSOR_SIZE
        uu, vv = np.meshgrid(u, v)

        # Find active contact points (non-zero depth)
        active_mask = tactile_depth > 0.01
        if not active_mask.any():
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros((self.camera_K.height, self.camera_K.width), dtype=np.uint8),
            )

        # 3D points in fingertip local frame (depth is along z-axis)
        x_local = uu[active_mask]
        y_local = vv[active_mask]
        z_local = tactile_depth[active_mask] * depth_scale

        points_local = np.stack([x_local, y_local, z_local], axis=-1)

        # Transform to world frame
        points_world = (fingertip_pose.rotation @ points_local.T).T + fingertip_pose.translation

        # Transform to camera frame
        cam_to_world = camera_pose.to_matrix()
        world_to_cam = np.linalg.inv(cam_to_world)

        points_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
        points_cam = (world_to_cam @ points_homo.T).T[:, :3]

        # Filter points behind camera
        valid = points_cam[:, 2] > 0
        if not valid.any():
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros((self.camera_K.height, self.camera_K.width), dtype=np.uint8),
            )

        points_cam = points_cam[valid]

        # Project to image plane
        K = self.camera_K.to_matrix()
        points_proj = K @ points_cam.T
        pixel_coords = points_proj[:2] / points_proj[2:3]
        pixel_coords = pixel_coords.T  # (N, 2)

        # Filter points outside image bounds
        in_bounds = (
            (pixel_coords[:, 0] >= 0)
            & (pixel_coords[:, 0] < self.camera_K.width)
            & (pixel_coords[:, 1] >= 0)
            & (pixel_coords[:, 1] < self.camera_K.height)
        )

        pixel_coords = pixel_coords[in_bounds]
        depths = points_cam[in_bounds, 2]

        # Create projection mask
        mask = np.zeros((self.camera_K.height, self.camera_K.width), dtype=np.uint8)
        if len(pixel_coords) > 0:
            u_px = pixel_coords[:, 0].astype(np.int32)
            v_px = pixel_coords[:, 1].astype(np.int32)
            mask[v_px, u_px] = 1

        return pixel_coords.astype(np.float32), depths.astype(np.float32), mask


class VisuotactileDepthFusion:
    """Fuses monocular visual depth with tactile depth measurements.

    Following NeuralFeels approach:
    - Vision provides global depth with lower confidence
    - Tactile provides high-confidence local depth at contact points
    - Weighted fusion combines both modalities
    """

    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        num_fingers: int = 4,
        visual_weight: float = 0.3,
        tactile_weight: float = 1.0,
        depth_model: Optional[str] = "Intel/dpt-hybrid-midas",
        device: Optional[torch.device] = None,
    ):
        """Initialize depth fusion.

        Args:
            camera_intrinsics: Camera intrinsic parameters
            num_fingers: Number of tactile sensors
            visual_weight: Confidence weight for visual depth [0, 1]
            tactile_weight: Confidence weight for tactile depth [0, 1]
            depth_model: HuggingFace model for monocular depth (None to skip)
            device: Torch device for inference
        """
        self.camera_K = camera_intrinsics
        self.num_fingers = num_fingers
        self.visual_weight = visual_weight
        self.tactile_weight = tactile_weight

        self.projector = TactileProjector(camera_intrinsics, num_fingers)

        self.depth_estimator = None
        if depth_model is not None:
            self.depth_estimator = MonocularDepthEstimator(depth_model, device)

    def estimate_visual_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Get monocular depth estimate from RGB.

        Args:
            rgb: RGB image (H, W, 3) uint8

        Returns:
            depth: Visual depth (H, W) normalized to [0, 1]
        """
        if self.depth_estimator is None:
            raise RuntimeError("Depth estimator not initialized")
        return self.depth_estimator.predict(rgb)

    def fuse(
        self,
        visual_depth: np.ndarray,
        tactile_depths: list[np.ndarray],
        fingertip_poses: list[SE3],
        camera_pose: SE3,
        depth_scale: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse visual and tactile depth measurements.

        Args:
            visual_depth: Visual depth map (H, W) normalized [0, 1]
            tactile_depths: List of 4 tactile depth maps, each (32, 32) in [0, 1]
            fingertip_poses: SE3 poses of fingertips in world frame
            camera_pose: SE3 pose of camera in world frame
            depth_scale: Scale factor for tactile depth to meters

        Returns:
            Tuple of:
                - fused_depth: Fused depth map (H, W)
                - confidence: Confidence map (H, W) - higher where tactile available
        """
        H, W = visual_depth.shape
        fused_depth = visual_depth.copy()
        confidence = np.full((H, W), self.visual_weight, dtype=np.float32)

        # Accumulate tactile depth measurements
        tactile_accumulator = np.zeros((H, W), dtype=np.float32)
        tactile_count = np.zeros((H, W), dtype=np.float32)

        for finger_idx, (tactile_depth, fingertip_pose) in enumerate(
            zip(tactile_depths, fingertip_poses)
        ):
            pixel_coords, depths, mask = self.projector.project_tactile_to_camera(
                tactile_depth,
                fingertip_pose,
                camera_pose,
                depth_scale,
            )

            if len(pixel_coords) == 0:
                continue

            # Accumulate depth values for averaging
            for (u, v), d in zip(pixel_coords, depths):
                u_int, v_int = int(u), int(v)
                if 0 <= u_int < W and 0 <= v_int < H:
                    tactile_accumulator[v_int, u_int] += d
                    tactile_count[v_int, u_int] += 1

        # Average accumulated tactile depths
        tactile_mask = tactile_count > 0
        if tactile_mask.any():
            tactile_depth_map = np.zeros((H, W), dtype=np.float32)
            tactile_depth_map[tactile_mask] = (
                tactile_accumulator[tactile_mask] / tactile_count[tactile_mask]
            )

            # Normalize tactile depth to match visual depth scale
            if tactile_depth_map[tactile_mask].max() > 0:
                tactile_normalized = tactile_depth_map.copy()
                # Simple min-max normalization within tactile region
                tac_min = tactile_depth_map[tactile_mask].min()
                tac_max = tactile_depth_map[tactile_mask].max()
                if tac_max > tac_min:
                    tactile_normalized[tactile_mask] = (
                        tactile_depth_map[tactile_mask] - tac_min
                    ) / (tac_max - tac_min)

                # Weighted fusion: tactile overrides visual where available
                # fused = (visual * visual_weight + tactile * tactile_weight) / (sum of weights)
                w_visual = self.visual_weight
                w_tactile = self.tactile_weight

                # In overlap regions, blend according to weights
                fused_depth[tactile_mask] = (
                    visual_depth[tactile_mask] * w_visual
                    + tactile_normalized[tactile_mask] * w_tactile
                ) / (w_visual + w_tactile)

                # Update confidence
                confidence[tactile_mask] = (w_visual + w_tactile) / 2

        return fused_depth, confidence

    def fuse_from_rgb(
        self,
        rgb: np.ndarray,
        tactile_depths: list[np.ndarray],
        fingertip_poses: list[SE3],
        camera_pose: SE3,
        depth_scale: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Complete pipeline: RGB -> visual depth -> fused depth.

        Args:
            rgb: RGB image (H, W, 3) uint8
            tactile_depths: List of tactile depth maps
            fingertip_poses: Fingertip poses in world frame
            camera_pose: Camera pose in world frame
            depth_scale: Scale for tactile depth

        Returns:
            Tuple of:
                - fused_depth: Fused depth map (H, W)
                - visual_depth: Raw visual depth map (H, W)
                - confidence: Confidence map (H, W)
        """
        visual_depth = self.estimate_visual_depth(rgb)

        fused_depth, confidence = self.fuse(
            visual_depth,
            tactile_depths,
            fingertip_poses,
            camera_pose,
            depth_scale,
        )

        return fused_depth, visual_depth, confidence


def unproject_depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unproject depth map to 3D point cloud.

    Args:
        depth: Depth map (H, W)
        intrinsics: Camera intrinsic parameters
        mask: Optional binary mask for valid regions

    Returns:
        points: 3D points (N, 3) in camera frame
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    if mask is None:
        mask = depth > 0

    # Get valid pixel coordinates
    u_valid = u[mask].flatten()
    v_valid = v[mask].flatten()
    z = depth[mask].flatten()

    # Unproject using pinhole camera model
    x = (u_valid - intrinsics.cx) * z / intrinsics.fx
    y = (v_valid - intrinsics.cy) * z / intrinsics.fy

    points = np.stack([x, y, z], axis=-1)
    return points.astype(np.float32)


def visualize_depth_fusion(
    visual_depth: np.ndarray,
    fused_depth: np.ndarray,
    confidence: np.ndarray,
    rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create visualization comparing visual and fused depth.

    Args:
        visual_depth: Visual depth map (H, W) normalized [0, 1]
        fused_depth: Fused depth map (H, W)
        confidence: Confidence map (H, W)
        rgb: Optional RGB image for context

    Returns:
        visualization: Composite image showing all modalities
    """
    import cv2

    H, W = visual_depth.shape

    def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
        """Convert depth to colormap."""
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(
            np.uint8
        )
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)

    visual_color = depth_to_colormap(visual_depth)
    fused_color = depth_to_colormap(fused_depth)

    # Confidence as grayscale overlay
    conf_color = (confidence * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_color, cv2.COLORMAP_HOT)

    # Create grid: [RGB | Visual | Fused | Confidence]
    if rgb is not None:
        rgb_resized = cv2.resize(rgb, (W, H))
        row = np.hstack([rgb_resized, visual_color, fused_color, conf_color])
    else:
        row = np.hstack([visual_color, fused_color, conf_color])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = (
        ["RGB", "Visual Depth", "Fused Depth", "Confidence"]
        if rgb is not None
        else [
            "Visual Depth",
            "Fused Depth",
            "Confidence",
        ]
    )
    for i, label in enumerate(labels):
        x = i * W + 5
        cv2.putText(row, label, (x, 20), font, 0.5, (255, 255, 255), 1)

    return row


def create_default_camera_intrinsics(
    width: int = 224,
    height: int = 224,
    fov_degrees: float = 45.0,
) -> CameraIntrinsics:
    """Create camera intrinsics from FOV (matches MuJoCo camera).

    Args:
        width: Image width
        height: Image height
        fov_degrees: Vertical field of view in degrees

    Returns:
        CameraIntrinsics for the specified camera
    """
    fov_rad = np.deg2rad(fov_degrees)
    fy = height / (2 * np.tan(fov_rad / 2))
    fx = fy  # Assuming square pixels

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )
