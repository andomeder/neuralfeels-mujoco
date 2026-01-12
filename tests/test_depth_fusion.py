"""Tests for depth fusion module."""

import numpy as np


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics dataclass."""

    def test_to_matrix(self):
        from perception.depth_fusion import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            fx=200.0,
            fy=200.0,
            cx=112.0,
            cy=112.0,
            width=224,
            height=224,
        )

        K = intrinsics.to_matrix()

        assert K.shape == (3, 3)
        assert K[0, 0] == 200.0  # fx
        assert K[1, 1] == 200.0  # fy
        assert K[0, 2] == 112.0  # cx
        assert K[1, 2] == 112.0  # cy
        assert K[2, 2] == 1.0


class TestSE3:
    """Tests for SE3 pose representation."""

    def test_to_matrix(self):
        from perception.depth_fusion import SE3

        rotation = np.eye(3, dtype=np.float32)
        translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pose = SE3(rotation=rotation, translation=translation)

        T = pose.to_matrix()

        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], rotation)
        assert np.allclose(T[:3, 3], translation)
        assert T[3, 3] == 1.0

    def test_inverse(self):
        from perception.depth_fusion import SE3

        # Simple translation
        rotation = np.eye(3, dtype=np.float32)
        translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pose = SE3(rotation=rotation, translation=translation)

        inverse = pose.inverse()

        # T * T^-1 should equal identity
        T = pose.to_matrix()
        T_inv = inverse.to_matrix()
        result = T @ T_inv

        assert np.allclose(result, np.eye(4), atol=1e-6)


class TestTactileProjector:
    """Tests for TactileProjector class."""

    def test_empty_tactile(self):
        from perception.depth_fusion import (
            SE3,
            CameraIntrinsics,
            TactileProjector,
        )

        intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=112.0, cy=112.0, width=224, height=224)
        projector = TactileProjector(intrinsics)

        # Empty tactile depth (no contact)
        tactile_depth = np.zeros((32, 32), dtype=np.float32)
        fingertip_pose = SE3(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        )
        camera_pose = SE3(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        pixel_coords, depths, mask = projector.project_tactile_to_camera(
            tactile_depth, fingertip_pose, camera_pose
        )

        assert len(pixel_coords) == 0
        assert len(depths) == 0
        assert mask.shape == (224, 224)
        assert mask.sum() == 0

    def test_single_contact(self):
        from perception.depth_fusion import (
            SE3,
            CameraIntrinsics,
            TactileProjector,
        )

        intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=112.0, cy=112.0, width=224, height=224)
        projector = TactileProjector(intrinsics)

        # Single contact point at center
        tactile_depth = np.zeros((32, 32), dtype=np.float32)
        tactile_depth[16, 16] = 0.5

        fingertip_pose = SE3(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        )
        camera_pose = SE3(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        pixel_coords, depths, mask = projector.project_tactile_to_camera(
            tactile_depth, fingertip_pose, camera_pose
        )

        assert len(pixel_coords) >= 1
        assert len(depths) >= 1
        assert mask.sum() >= 1


class TestCreateDefaultCameraIntrinsics:
    """Tests for camera intrinsics factory function."""

    def test_default_intrinsics(self):
        from perception.depth_fusion import create_default_camera_intrinsics

        intrinsics = create_default_camera_intrinsics()

        assert intrinsics.width == 224
        assert intrinsics.height == 224
        assert intrinsics.cx == 112.0
        assert intrinsics.cy == 112.0
        # For 45 degree FOV: fy = 224 / (2 * tan(22.5 deg)) ≈ 270.2
        assert 260 < intrinsics.fy < 280

    def test_custom_fov(self):
        from perception.depth_fusion import create_default_camera_intrinsics

        intrinsics = create_default_camera_intrinsics(fov_degrees=60.0)

        # Wider FOV = smaller focal length
        intrinsics_45 = create_default_camera_intrinsics(fov_degrees=45.0)
        assert intrinsics.fy < intrinsics_45.fy


class TestUnprojectDepthToPoints:
    """Tests for depth unprojection."""

    def test_unproject_single_point(self):
        from perception.depth_fusion import (
            CameraIntrinsics,
            unproject_depth_to_points,
        )

        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)

        # Depth map with single point at center
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[50, 50] = 1.0  # 1 meter depth at center

        points = unproject_depth_to_points(depth, intrinsics)

        assert points.shape == (1, 3)
        # At center, x and y should be 0
        assert np.allclose(points[0, :2], [0.0, 0.0], atol=0.01)
        assert points[0, 2] == 1.0


class TestVisuotactileDepthFusion:
    """Tests for VisuotactileDepthFusion class (without loading DPT model)."""

    def test_fusion_no_tactile(self):
        from perception.depth_fusion import (
            SE3,
            CameraIntrinsics,
            VisuotactileDepthFusion,
        )

        intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=112.0, cy=112.0, width=224, height=224)

        # Create fusion without depth model (for faster testing)
        fusion = VisuotactileDepthFusion(
            camera_intrinsics=intrinsics,
            depth_model=None,
        )

        # Visual depth
        visual_depth = np.random.rand(224, 224).astype(np.float32)

        # Empty tactile
        tactile_depths = [np.zeros((32, 32), dtype=np.float32) for _ in range(4)]
        fingertip_poses = [
            SE3(
                rotation=np.eye(3, dtype=np.float32),
                translation=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            )
            for _ in range(4)
        ]
        camera_pose = SE3(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        fused_depth, confidence = fusion.fuse(
            visual_depth, tactile_depths, fingertip_poses, camera_pose
        )

        # With no tactile, fused should equal visual
        assert fused_depth.shape == (224, 224)
        assert np.allclose(fused_depth, visual_depth)
        # Confidence should be visual weight everywhere
        assert np.allclose(confidence, fusion.visual_weight)


class TestVisualizeDepthFusion:
    """Tests for visualization function."""

    def test_visualization_shape(self):
        from perception.depth_fusion import visualize_depth_fusion

        visual_depth = np.random.rand(224, 224).astype(np.float32)
        fused_depth = np.random.rand(224, 224).astype(np.float32)
        confidence = np.random.rand(224, 224).astype(np.float32)

        # Without RGB
        vis = visualize_depth_fusion(visual_depth, fused_depth, confidence)
        assert vis.shape[0] == 224
        assert vis.shape[2] == 3  # BGR color

        # With RGB
        rgb = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        vis = visualize_depth_fusion(visual_depth, fused_depth, confidence, rgb)
        assert vis.shape[0] == 224
        assert vis.shape[2] == 3
