"""Tests for pose tracking module."""

import numpy as np
import pytest
import torch


class TestPoseTrackingConfig:
    """Tests for PoseTrackingConfig."""

    def test_default_config(self):
        from perception.pose_tracking import PoseTrackingConfig

        config = PoseTrackingConfig()

        assert config.window_size == 5
        assert config.max_lm_iterations == 20
        assert config.device == "cpu"


class TestVisualOdometry:
    """Tests for VisualOdometry class."""

    def test_detect_features(self):
        from perception.pose_tracking import VisualOdometry

        vo = VisualOdometry(num_features=100)

        # Create a test image with some texture
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        keypoints, descriptors = vo.detect_features(rgb)

        assert keypoints.ndim == 2
        assert keypoints.shape[1] == 2  # (x, y) coordinates
        assert descriptors.ndim == 2
        assert descriptors.shape[1] == 32  # ORB descriptor size

    def test_detect_features_blank_image(self):
        from perception.pose_tracking import VisualOdometry

        vo = VisualOdometry()

        # Blank image should have few/no features
        rgb = np.ones((224, 224, 3), dtype=np.uint8) * 128

        keypoints, descriptors = vo.detect_features(rgb)

        # Should return arrays (possibly empty)
        assert keypoints.ndim == 2
        assert descriptors.ndim == 2

    def test_match_features(self):
        from perception.pose_tracking import VisualOdometry

        vo = VisualOdometry()

        # Create two similar random descriptor sets
        desc1 = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
        desc2 = desc1.copy()  # Identical descriptors should match well

        matches = vo.match_features(desc1, desc2)

        assert matches.ndim == 2
        assert matches.shape[1] == 2  # [idx1, idx2]

    def test_match_features_empty(self):
        from perception.pose_tracking import VisualOdometry

        vo = VisualOdometry()

        desc1 = np.zeros((0, 32), dtype=np.uint8)
        desc2 = np.zeros((0, 32), dtype=np.uint8)

        matches = vo.match_features(desc1, desc2)

        assert len(matches) == 0


class TestPoseGraphOptimizer:
    """Tests for PoseGraphOptimizer class."""

    def test_creation(self):
        from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig

        config = PoseTrackingConfig()
        optimizer = PoseGraphOptimizer(config)

        assert optimizer.config == config
        assert len(optimizer.keyframes) == 0

    def test_initialize(self):
        from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig

        config = PoseTrackingConfig()
        optimizer = PoseGraphOptimizer(config)

        optimizer.initialize()

        pose = optimizer.current_pose
        assert pose is not None
        assert pose.shape == (4, 4)
        # Should be identity
        assert np.allclose(pose, np.eye(4), atol=1e-6)

    def test_initialize_with_pose(self):
        from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig

        config = PoseTrackingConfig()
        optimizer = PoseGraphOptimizer(config)

        # Custom initial pose
        initial_pose = np.eye(4, dtype=np.float32)
        initial_pose[:3, 3] = [0.1, 0.2, 0.3]  # Translation

        optimizer.initialize(initial_pose)

        pose = optimizer.current_pose
        assert np.allclose(pose[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)

    def test_add_frame(self):
        from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig

        config = PoseTrackingConfig(window_size=3)
        optimizer = PoseGraphOptimizer(config)
        optimizer.initialize()

        # Add frames
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        for i in range(5):
            pose = optimizer.add_frame(rgb, timestamp=i)
            assert pose is not None
            assert pose.shape == (4, 4)

        # Window should be maintained
        assert len(optimizer.keyframes) == config.window_size

    def test_get_trajectory(self):
        from perception.pose_tracking import PoseGraphOptimizer, PoseTrackingConfig

        config = PoseTrackingConfig(window_size=5)
        optimizer = PoseGraphOptimizer(config)
        optimizer.initialize()

        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        for i in range(3):
            optimizer.add_frame(rgb, timestamp=i)

        trajectory = optimizer.get_trajectory()

        assert len(trajectory) == 3
        for T in trajectory:
            assert T.shape == (4, 4)


class TestCreatePoseTracker:
    """Tests for factory function."""

    def test_create_without_sdf(self):
        from perception.pose_tracking import create_pose_tracker

        tracker = create_pose_tracker()

        assert tracker is not None
        assert tracker.sdf_model is None

    def test_create_with_camera_intrinsics(self):
        from perception.pose_tracking import create_pose_tracker

        K = np.array([[200, 0, 112], [0, 200, 112], [0, 0, 1]], dtype=np.float32)

        tracker = create_pose_tracker(camera_intrinsics=K)

        assert np.allclose(tracker.camera_K, K)


class TestTheseusIntegration:
    """Tests for Theseus SE3 integration."""

    def test_theseus_se3_creation(self):
        import theseus as th

        # Identity pose
        pose = th.SE3()
        assert pose.tensor.shape == (1, 3, 4)

        # From matrix
        mat = torch.eye(3, 4).unsqueeze(0)
        pose = th.SE3(tensor=mat)
        assert torch.allclose(pose.tensor, mat)

    def test_theseus_se3_compose(self):
        import theseus as th

        pose1 = th.SE3()
        pose2 = th.SE3()

        composed = pose1.compose(pose2)
        assert composed.tensor.shape == (1, 3, 4)

    def test_theseus_se3_inverse(self):
        import theseus as th

        pose = th.SE3()
        pose_inv = pose.inverse()

        # Identity inverse should be identity
        assert torch.allclose(pose_inv.tensor, pose.tensor, atol=1e-6)
