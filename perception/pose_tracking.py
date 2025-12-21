"""Pose graph optimization for object tracking during manipulation.

Uses Theseus for differentiable SE3 optimization with:
- Visual odometry factors (ORB feature matching)
- SDF consistency factors (points should lie on neural surface)
- Regularization factors (smooth motion prior)

References:
- NeuralFeels: Sliding window pose graph with SDF/ICP factors
- Theseus: Differentiable nonlinear optimization library
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import theseus as th
import torch
import torch.nn as nn

from src.utils.gpu_utils import get_device


@dataclass
class PoseTrackingConfig:
    """Configuration for pose tracking."""

    window_size: int = 5  # Sliding window size
    max_lm_iterations: int = 20  # Levenberg-Marquardt iterations
    lm_step_size: float = 0.5  # LM step size
    sdf_weight: float = 0.01  # Weight for SDF consistency
    reg_weight: float = 0.01  # Weight for motion regularization
    icp_weight: float = 1.0  # Weight for ICP/odometry
    device: str = "cpu"  # Force CPU for Theseus (XPU not supported)


@dataclass
class Keyframe:
    """A keyframe for pose tracking."""

    timestamp: int
    pose: th.SE3  # Object pose in world frame
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None  # 3D points in camera frame
    descriptors: Optional[np.ndarray] = None  # ORB descriptors


class VisualOdometry:
    """Simple visual odometry using ORB features.

    Estimates relative camera motion between frames using feature matching.
    """

    def __init__(self, num_features: int = 500):
        """Initialize ORB detector and matcher.

        Args:
            num_features: Maximum number of ORB features to detect
        """
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_features(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect ORB keypoints and compute descriptors.

        Args:
            rgb: RGB image (H, W, 3) uint8

        Returns:
            keypoints: (N, 2) array of keypoint coordinates
            descriptors: (N, 32) array of ORB descriptors
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

        keypoints = np.array([kp.pt for kp in kps], dtype=np.float32)
        descriptors = descs if descs is not None else np.zeros((0, 32), dtype=np.uint8)

        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> np.ndarray:
        """Match ORB descriptors between two frames.

        Args:
            desc1: Descriptors from first frame (N, 32)
            desc2: Descriptors from second frame (M, 32)
            ratio_thresh: Lowe's ratio test threshold

        Returns:
            matches: (K, 2) array of matched indices [idx1, idx2]
        """
        if len(desc1) < 2 or len(desc2) < 2:
            return np.zeros((0, 2), dtype=np.int32)

        matches = self.matcher.match(desc1, desc2)

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Take top matches
        good_matches = matches[: min(50, len(matches))]

        if len(good_matches) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in good_matches], dtype=np.int32)

        return match_indices

    def estimate_relative_pose(
        self,
        kp1: np.ndarray,
        kp2: np.ndarray,
        matches: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate relative pose from matched features using essential matrix.

        Args:
            kp1: Keypoints from first frame (N, 2)
            kp2: Keypoints from second frame (M, 2)
            matches: Matched indices (K, 2)
            K: Camera intrinsic matrix (3, 3)

        Returns:
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)
            inliers: Inlier mask (K,)
        """
        if len(matches) < 5:
            return np.eye(3), np.zeros(3), np.zeros(0, dtype=bool)

        pts1 = kp1[matches[:, 0]]
        pts2 = kp2[matches[:, 1]]

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            return np.eye(3), np.zeros(3), np.zeros(len(matches), dtype=bool)

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        inliers = (
            mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
        )

        return R, t.ravel(), inliers


class SDFConsistencyFactor(th.CostFunction):
    """Cost function that penalizes points not lying on the SDF surface.

    For each observed 3D point, transform it by the object pose and query
    the neural SDF. The error is the predicted SDF value (should be ~0 for surface).
    """

    def __init__(
        self,
        pose: th.SE3,
        points: th.Variable,
        sdf_model: nn.Module,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        """Initialize SDF consistency factor.

        Args:
            pose: SE3 pose variable to optimize
            points: Point3 variable containing observed 3D points
            sdf_model: Neural SDF model (maps 3D points to SDF values)
            cost_weight: Cost weight for this factor
            name: Optional name for the cost function
        """
        super().__init__(cost_weight, name=name)
        self.pose = pose
        self.points = points
        self.sdf_model = sdf_model

        self.register_optim_vars(["pose"])
        self.register_aux_vars(["points"])

    def error(self) -> torch.Tensor:
        """Compute SDF error for all points.

        Returns:
            error: (batch_size, num_points) SDF values
        """
        # Transform points from world to object frame
        # pose is world_T_object, so we need object_T_world = pose.inverse()
        pose_inv = self.pose.inverse()

        # points is (batch, N, 3)
        points_world = self.points.tensor
        batch_size = points_world.shape[0]
        num_points = points_world.shape[1]

        # Transform each point
        # pose_inv.transform_from expects Point3, but we have raw tensor
        # Use matrix multiplication: p_obj = R^T @ (p_world - t)
        R = self.pose.tensor[:, :3, :3]  # (batch, 3, 3)
        t = self.pose.tensor[:, :3, 3:4]  # (batch, 3, 1)

        # points_world: (batch, N, 3) -> (batch, 3, N)
        points_t = points_world.transpose(1, 2)

        # R^T @ (p - t) for each point
        points_obj = torch.bmm(R.transpose(1, 2), points_t - t)  # (batch, 3, N)
        points_obj = points_obj.transpose(1, 2)  # (batch, N, 3)

        # Query SDF for each point
        # Reshape to (batch * N, 3) for SDF model
        points_flat = points_obj.reshape(-1, 3)

        with torch.no_grad():
            sdf_values = self.sdf_model(points_flat)  # (batch * N,)

        sdf_values = sdf_values.reshape(batch_size, num_points)

        return sdf_values

    def jacobians(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Compute Jacobians for the error.

        For now, use numerical differentiation via autograd.
        """
        # Use autodiff for Jacobian computation
        return super().jacobians()

    def dim(self) -> int:
        """Return the dimension of the error."""
        return self.points.tensor.shape[1]

    def _copy_impl(self, new_name: Optional[str] = None) -> "SDFConsistencyFactor":
        return SDFConsistencyFactor(
            self.pose.copy(),
            self.points.copy(),
            self.sdf_model,
            self.weight.copy(),
            name=new_name,
        )


class PoseGraphOptimizer:
    """Sliding window pose graph optimizer using Theseus.

    Maintains a window of recent keyframes and optimizes object poses
    using visual odometry, SDF consistency, and regularization factors.
    """

    def __init__(
        self,
        config: PoseTrackingConfig,
        sdf_model: Optional[nn.Module] = None,
        camera_K: Optional[np.ndarray] = None,
    ):
        """Initialize pose graph optimizer.

        Args:
            config: Pose tracking configuration
            sdf_model: Neural SDF model for consistency factors
            camera_K: Camera intrinsic matrix (3, 3)
        """
        self.config = config
        self.sdf_model = sdf_model
        self.camera_K = camera_K if camera_K is not None else np.eye(3)

        self.device = torch.device(config.device)
        self.keyframes: list[Keyframe] = []
        self.visual_odom = VisualOdometry()

        # Current object pose estimate
        self._current_pose: Optional[th.SE3] = None

    @property
    def current_pose(self) -> Optional[np.ndarray]:
        """Get current object pose as 4x4 matrix."""
        if self._current_pose is None:
            return None

        # Convert to 4x4 homogeneous matrix
        mat = self._current_pose.tensor[0].cpu().numpy()  # (3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = mat
        return T

    def initialize(self, initial_pose: Optional[np.ndarray] = None):
        """Initialize the tracker with an initial pose.

        Args:
            initial_pose: Initial 4x4 object pose matrix (default: identity)
        """
        if initial_pose is None:
            initial_pose = np.eye(4, dtype=np.float32)

        # Convert to Theseus SE3
        mat_tensor = torch.tensor(initial_pose[:3, :], dtype=torch.float32).unsqueeze(0)
        self._current_pose = th.SE3(tensor=mat_tensor.to(self.device))
        self.keyframes.clear()

    def add_frame(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        timestamp: int = 0,
    ) -> np.ndarray:
        """Add a new frame and update pose estimate.

        Args:
            rgb: RGB image (H, W, 3)
            depth: Optional depth map (H, W)
            points: Optional 3D points in camera frame (N, 3)
            timestamp: Frame timestamp

        Returns:
            Updated object pose as 4x4 matrix
        """
        if self._current_pose is None:
            self.initialize()

        # Detect features
        keypoints, descriptors = self.visual_odom.detect_features(rgb)

        # Create new keyframe
        new_keyframe = Keyframe(
            timestamp=timestamp,
            pose=self._current_pose.copy(),
            rgb=rgb.copy(),
            depth=depth.copy() if depth is not None else None,
            points=points.copy() if points is not None else None,
            descriptors=descriptors,
        )

        # If we have previous keyframes, estimate relative motion
        if len(self.keyframes) > 0:
            prev_kf = self.keyframes[-1]

            if prev_kf.descriptors is not None and len(prev_kf.descriptors) > 0:
                # Match features
                prev_kps, _ = self.visual_odom.detect_features(prev_kf.rgb)
                matches = self.visual_odom.match_features(prev_kf.descriptors, descriptors)

                if len(matches) >= 5:
                    # Estimate relative pose
                    R, t, inliers = self.visual_odom.estimate_relative_pose(
                        prev_kps, keypoints, matches, self.camera_K
                    )

                    # Apply relative motion to current pose
                    if inliers.sum() >= 5:
                        self._apply_relative_motion(R, t)

        # Add keyframe and maintain window
        self.keyframes.append(new_keyframe)
        if len(self.keyframes) > self.config.window_size:
            self.keyframes.pop(0)

        # Optimize pose graph if we have SDF model and points
        if self.sdf_model is not None and points is not None and len(points) > 10:
            self._optimize_with_sdf(points)

        return self.current_pose

    def _apply_relative_motion(self, R: np.ndarray, t: np.ndarray):
        """Apply relative camera motion to object pose.

        Since camera moved, object pose in camera frame changes inversely.
        """
        # Create relative transformation
        rel_T = np.eye(4, dtype=np.float32)
        rel_T[:3, :3] = R
        rel_T[:3, 3] = t

        # Current pose
        current_T = self.current_pose

        # New pose = rel_T^-1 @ current_T (camera motion -> inverse for object)
        rel_T_inv = np.linalg.inv(rel_T)
        new_T = rel_T_inv @ current_T

        # Update Theseus SE3
        mat_tensor = torch.tensor(new_T[:3, :], dtype=torch.float32).unsqueeze(0)
        self._current_pose = th.SE3(tensor=mat_tensor.to(self.device))

    def _optimize_with_sdf(self, points: np.ndarray):
        """Optimize current pose using SDF consistency.

        Args:
            points: Observed 3D points (N, 3) in world frame
        """
        if self.sdf_model is None:
            return

        # Sample a subset of points for efficiency
        num_points = min(100, len(points))
        indices = np.random.choice(len(points), num_points, replace=False)
        sampled_points = points[indices]

        # Create Theseus optimization problem
        objective = th.Objective()

        # Create pose variable (to optimize) - on CPU for Theseus
        pose_var = self._current_pose.copy()
        pose_var.name = "object_pose"

        # Create points auxiliary variable - on CPU for Theseus
        points_tensor = torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0)
        points_var = th.Variable(tensor=points_tensor.to(self.device), name="observed_points")

        # Add SDF consistency factor
        sdf_weight = th.ScaleCostWeight(self.config.sdf_weight)
        sdf_weight.to(self.device)

        # Get SDF model device for moving tensors during query
        sdf_device = next(self.sdf_model.parameters()).device

        # Simple approach: use autodiff cost function
        def sdf_error_fn(optim_vars, aux_vars):
            pose = optim_vars[0]
            pts = aux_vars[0].tensor  # (1, N, 3)

            # Transform points to object frame
            R = pose.tensor[:, :3, :3]
            t = pose.tensor[:, :3, 3:4]

            pts_t = pts.transpose(1, 2)
            pts_obj = torch.bmm(R.transpose(1, 2), pts_t - t).transpose(1, 2)

            # Query SDF - move points to SDF device, then result back to CPU
            pts_flat = pts_obj.reshape(-1, 3).to(sdf_device)
            with torch.no_grad():
                sdf_vals = self.sdf_model(pts_flat)
            sdf_vals = sdf_vals.to(self.device)

            return sdf_vals.reshape(1, -1)

        sdf_factor = th.AutoDiffCostFunction(
            optim_vars=[pose_var],
            err_fn=sdf_error_fn,
            dim=num_points,
            aux_vars=[points_var],
            cost_weight=sdf_weight,
            name="sdf_consistency",
        )
        objective.add(sdf_factor)

        # Add regularization (prior on current pose)
        if len(self.keyframes) > 1:
            prev_pose = self.keyframes[-2].pose.copy()
            prev_pose.name = "prev_pose"

            reg_weight = th.ScaleCostWeight(self.config.reg_weight)
            reg_weight.to(self.device)

            reg_factor = th.Difference(
                pose_var,
                prev_pose,
                reg_weight,
                name="motion_prior",
            )
            objective.add(reg_factor)

        # Create optimizer and run
        try:
            optimizer = th.LevenbergMarquardt(
                objective,
                max_iterations=self.config.max_lm_iterations,
                step_size=self.config.lm_step_size,
            )

            with torch.no_grad():
                optimizer.optimize()

            # Update current pose
            self._current_pose = pose_var

        except Exception as e:
            # If optimization fails, keep current pose
            print(f"Pose optimization failed: {e}")

    def get_trajectory(self) -> list[np.ndarray]:
        """Get the trajectory of optimized poses.

        Returns:
            List of 4x4 pose matrices for each keyframe
        """
        trajectory = []
        for kf in self.keyframes:
            mat = kf.pose.tensor[0].cpu().numpy()
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = mat
            trajectory.append(T)
        return trajectory


def create_pose_tracker(
    sdf_model: Optional[nn.Module] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    window_size: int = 5,
) -> PoseGraphOptimizer:
    """Factory function to create a pose tracker.

    Args:
        sdf_model: Neural SDF model for consistency optimization
        camera_intrinsics: Camera K matrix (3, 3)
        window_size: Sliding window size

    Returns:
        Configured PoseGraphOptimizer
    """
    config = PoseTrackingConfig(
        window_size=window_size,
        device="cpu",  # Force CPU for Theseus (XPU not natively supported)
    )

    tracker = PoseGraphOptimizer(
        config=config,
        sdf_model=sdf_model,
        camera_K=camera_intrinsics,
    )

    return tracker
