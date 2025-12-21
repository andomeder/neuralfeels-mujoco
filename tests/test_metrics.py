"""Tests for evaluation metrics."""

import numpy as np
import pytest


class TestFScore:
    """Tests for F-score metric."""

    def test_perfect_match(self):
        from perception.metrics import f_score

        points = np.random.rand(100, 3).astype(np.float32)

        result = f_score(points, points, threshold=0.01)

        assert result["f_score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_overlap(self):
        from perception.metrics import f_score

        pred_points = np.zeros((100, 3), dtype=np.float32)
        gt_points = np.ones((100, 3), dtype=np.float32)

        result = f_score(pred_points, gt_points, threshold=0.01)

        assert result["f_score"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_partial_overlap(self):
        from perception.metrics import f_score

        pred_points = np.random.rand(100, 3).astype(np.float32)
        # Shift half the points
        gt_points = pred_points.copy()
        gt_points[50:] += 1.0

        result = f_score(pred_points, gt_points, threshold=0.01)

        # Should have partial precision/recall
        assert 0.0 < result["f_score"] < 1.0

    def test_empty_points(self):
        from perception.metrics import f_score

        pred_points = np.zeros((0, 3), dtype=np.float32)
        gt_points = np.random.rand(100, 3).astype(np.float32)

        result = f_score(pred_points, gt_points, threshold=0.01)

        assert result["f_score"] == 0.0


class TestChamferDistance:
    """Tests for chamfer distance computation."""

    def test_same_points(self):
        from perception.metrics import chamfer_distance

        points = np.random.rand(100, 3).astype(np.float32)

        dist_a_to_b, dist_b_to_a = chamfer_distance(points, points)

        assert np.allclose(dist_a_to_b, 0.0)
        assert np.allclose(dist_b_to_a, 0.0)

    def test_translated_points(self):
        from perception.metrics import chamfer_distance

        points_a = np.zeros((10, 3), dtype=np.float32)
        points_b = np.ones((10, 3), dtype=np.float32)

        dist_a_to_b, dist_b_to_a = chamfer_distance(points_a, points_b)

        expected_dist = np.sqrt(3)  # Distance from (0,0,0) to (1,1,1)
        assert np.allclose(dist_a_to_b, expected_dist, atol=0.01)


class TestADDS:
    """Tests for ADD-S metric."""

    def test_identity_pose(self):
        from perception.metrics import add_s

        pose = np.eye(4, dtype=np.float32)
        model_points = np.random.rand(100, 3).astype(np.float32)

        result = add_s(pose, pose, model_points)

        assert result < 1e-6

    def test_translated_pose(self):
        from perception.metrics import add_s

        pred_pose = np.eye(4, dtype=np.float32)
        pred_pose[:3, 3] = [0.01, 0, 0]  # 10mm translation

        gt_pose = np.eye(4, dtype=np.float32)

        model_points = np.array([[0, 0, 0]], dtype=np.float32)

        result = add_s(pred_pose, gt_pose, model_points)

        assert np.isclose(result, 0.01, atol=1e-6)


class TestADD:
    """Tests for ADD metric."""

    def test_identity_pose(self):
        from perception.metrics import add

        pose = np.eye(4, dtype=np.float32)
        model_points = np.random.rand(100, 3).astype(np.float32)

        result = add(pose, pose, model_points)

        assert result < 1e-6

    def test_translated_pose(self):
        from perception.metrics import add

        pred_pose = np.eye(4, dtype=np.float32)
        pred_pose[:3, 3] = [0.01, 0, 0]

        gt_pose = np.eye(4, dtype=np.float32)

        model_points = np.array([[0, 0, 0]], dtype=np.float32)

        result = add(pred_pose, gt_pose, model_points)

        assert np.isclose(result, 0.01, atol=1e-6)


class TestRotationError:
    """Tests for rotation error computation."""

    def test_identity_rotation(self):
        from perception.metrics import rotation_error

        pose = np.eye(4, dtype=np.float32)

        result = rotation_error(pose, pose)

        assert result < 1e-6

    def test_90_degree_rotation(self):
        from perception.metrics import rotation_error

        pred_pose = np.eye(4, dtype=np.float32)
        # Rotate 90 degrees around z-axis
        pred_pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        gt_pose = np.eye(4, dtype=np.float32)

        result = rotation_error(pred_pose, gt_pose)

        assert np.isclose(result, 90.0, atol=1.0)


class TestTranslationError:
    """Tests for translation error computation."""

    def test_no_translation(self):
        from perception.metrics import translation_error

        pose = np.eye(4, dtype=np.float32)

        result = translation_error(pose, pose)

        assert result < 1e-6

    def test_known_translation(self):
        from perception.metrics import translation_error

        pred_pose = np.eye(4, dtype=np.float32)
        pred_pose[:3, 3] = [0.003, 0.004, 0]  # 5mm (3-4-5 triangle)

        gt_pose = np.eye(4, dtype=np.float32)

        result = translation_error(pred_pose, gt_pose)

        assert np.isclose(result, 0.005, atol=1e-6)


class TestPoseMetrics:
    """Tests for comprehensive pose metrics."""

    def test_pose_metrics_output(self):
        from perception.metrics import pose_metrics

        pred_pose = np.eye(4, dtype=np.float32)
        gt_pose = np.eye(4, dtype=np.float32)
        model_points = np.random.rand(100, 3).astype(np.float32)

        result = pose_metrics(pred_pose, gt_pose, model_points)

        assert "add" in result
        assert "add_s" in result
        assert "rotation_error_deg" in result
        assert "translation_error_m" in result
        assert "translation_error_mm" in result


class TestEvaluateTrajectory:
    """Tests for trajectory evaluation."""

    def test_perfect_trajectory(self):
        from perception.metrics import evaluate_trajectory

        poses = [np.eye(4, dtype=np.float32) for _ in range(10)]
        model_points = np.random.rand(100, 3).astype(np.float32)

        result = evaluate_trajectory(poses, poses, model_points)

        assert result["mean_add"] < 1e-6
        assert result["mean_add_s"] < 1e-6
        assert result["mean_translation_error_mm"] < 1e-3

    def test_mismatched_length(self):
        from perception.metrics import evaluate_trajectory

        pred_poses = [np.eye(4) for _ in range(5)]
        gt_poses = [np.eye(4) for _ in range(10)]
        model_points = np.random.rand(100, 3).astype(np.float32)

        with pytest.raises(ValueError):
            evaluate_trajectory(pred_poses, gt_poses, model_points)


class TestGenerateSpherePoints:
    """Tests for sphere point generation."""

    def test_sphere_radius(self):
        from perception.metrics import generate_sphere_points

        radius = 0.025
        points = generate_sphere_points(radius=radius, num_points=1000)

        # All points should be at the specified radius
        distances = np.linalg.norm(points, axis=1)
        assert np.allclose(distances, radius, atol=1e-6)

    def test_sphere_num_points(self):
        from perception.metrics import generate_sphere_points

        num_points = 500
        points = generate_sphere_points(num_points=num_points)

        assert len(points) == num_points


class TestSampleMeshSurface:
    """Tests for mesh surface sampling."""

    def test_simple_triangle(self):
        from perception.metrics import sample_mesh_surface

        # Simple triangle
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        points = sample_mesh_surface(vertices, faces, num_samples=100, seed=42)

        assert len(points) == 100
        # All points should be on z=0 plane
        assert np.allclose(points[:, 2], 0.0)
        # All points should be within triangle bounds
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1.01)  # Allow small tolerance
