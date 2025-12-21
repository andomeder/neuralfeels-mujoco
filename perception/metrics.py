"""Evaluation metrics for visuotactile perception.

Implements:
- F-Score: Harmonic mean of precision and recall for mesh reconstruction
- ADD-S: Symmetric average distance for pose estimation

References:
- NeuralFeels: Uses F-score @ 5mm for shape, ADD-S for pose
- BOP Challenge: Standard pose estimation metrics
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample points uniformly on mesh surface.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Mesh face indices (F, 3)
        num_samples: Number of points to sample
        seed: Random seed for reproducibility

    Returns:
        points: Sampled surface points (N, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product for area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Sample faces proportional to area
    probabilities = areas / areas.sum()
    face_indices = np.random.choice(len(faces), size=num_samples, p=probabilities)

    # Sample random barycentric coordinates
    r1 = np.random.random(num_samples)
    r2 = np.random.random(num_samples)

    # Ensure points are inside triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Compute points
    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]

    points = (
        u[:, np.newaxis] * sampled_v0
        + v[:, np.newaxis] * sampled_v1
        + w[:, np.newaxis] * sampled_v2
    )

    return points.astype(np.float32)


def chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bidirectional chamfer distances.

    Args:
        points_a: First point set (N, 3)
        points_b: Second point set (M, 3)

    Returns:
        dist_a_to_b: Distance from each point in A to nearest in B (N,)
        dist_b_to_a: Distance from each point in B to nearest in A (M,)
    """
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    return dist_a_to_b, dist_b_to_a


def f_score(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.005,  # 5mm default
) -> dict:
    """Compute F-score for mesh reconstruction quality.

    F-score is the harmonic mean of precision and recall:
    - Precision: Fraction of predicted points within threshold of GT
    - Recall: Fraction of GT points within threshold of predicted

    Args:
        pred_points: Predicted mesh surface points (N, 3)
        gt_points: Ground truth mesh surface points (M, 3)
        threshold: Distance threshold in meters (default 5mm)

    Returns:
        Dictionary with:
            - f_score: Harmonic mean of precision and recall
            - precision: Fraction of pred within threshold of GT
            - recall: Fraction of GT within threshold of pred
            - mean_dist_pred_to_gt: Mean distance from pred to GT
            - mean_dist_gt_to_pred: Mean distance from GT to pred
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {
            "f_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_dist_pred_to_gt": float("inf"),
            "mean_dist_gt_to_pred": float("inf"),
        }

    dist_pred_to_gt, dist_gt_to_pred = chamfer_distance(pred_points, gt_points)

    precision = (dist_pred_to_gt < threshold).mean()
    recall = (dist_gt_to_pred < threshold).mean()

    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = 0.0

    return {
        "f_score": float(f),
        "precision": float(precision),
        "recall": float(recall),
        "mean_dist_pred_to_gt": float(dist_pred_to_gt.mean()),
        "mean_dist_gt_to_pred": float(dist_gt_to_pred.mean()),
    }


def f_score_from_meshes(
    pred_vertices: np.ndarray,
    pred_faces: np.ndarray,
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
    threshold: float = 0.005,
    num_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute F-score between two meshes.

    Args:
        pred_vertices: Predicted mesh vertices (V1, 3)
        pred_faces: Predicted mesh faces (F1, 3)
        gt_vertices: Ground truth mesh vertices (V2, 3)
        gt_faces: Ground truth mesh faces (F2, 3)
        threshold: Distance threshold in meters
        num_samples: Number of surface samples per mesh
        seed: Random seed for sampling

    Returns:
        F-score metrics dictionary
    """
    pred_points = sample_mesh_surface(pred_vertices, pred_faces, num_samples, seed)
    gt_points = sample_mesh_surface(gt_vertices, gt_faces, num_samples, seed)

    return f_score(pred_points, gt_points, threshold)


def add_s(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """Compute ADD-S (Average Distance of Symmetric) metric.

    ADD-S measures the average distance between the predicted and ground truth
    poses by finding the closest point correspondence. This is symmetric and
    works for objects with symmetries.

    ADD-S = (1/N) * sum_i min_j ||R_pred @ p_i + t_pred - (R_gt @ p_j + t_gt)||

    Args:
        pred_pose: Predicted 4x4 pose matrix
        gt_pose: Ground truth 4x4 pose matrix
        model_points: Object model points (N, 3)

    Returns:
        ADD-S distance in meters
    """
    if len(model_points) == 0:
        return float("inf")

    # Transform points by predicted pose
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    pred_transformed = (pred_R @ model_points.T).T + pred_t

    # Transform points by ground truth pose
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    gt_transformed = (gt_R @ model_points.T).T + gt_t

    # For each predicted point, find distance to closest GT point
    tree_gt = KDTree(gt_transformed)
    distances, _ = tree_gt.query(pred_transformed)

    return float(distances.mean())


def add(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """Compute ADD (Average Distance) metric.

    Standard ADD measures the average distance between corresponding points
    after applying the predicted and ground truth poses.

    ADD = (1/N) * sum_i ||R_pred @ p_i + t_pred - (R_gt @ p_i + t_gt)||

    Args:
        pred_pose: Predicted 4x4 pose matrix
        gt_pose: Ground truth 4x4 pose matrix
        model_points: Object model points (N, 3)

    Returns:
        ADD distance in meters
    """
    if len(model_points) == 0:
        return float("inf")

    # Transform points by predicted pose
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    pred_transformed = (pred_R @ model_points.T).T + pred_t

    # Transform points by ground truth pose
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    gt_transformed = (gt_R @ model_points.T).T + gt_t

    # Compute point-wise distances
    distances = np.linalg.norm(pred_transformed - gt_transformed, axis=1)

    return float(distances.mean())


def rotation_error(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
) -> float:
    """Compute rotation error between poses in degrees.

    Args:
        pred_pose: Predicted 4x4 pose matrix
        gt_pose: Ground truth 4x4 pose matrix

    Returns:
        Rotation error in degrees
    """
    pred_R = pred_pose[:3, :3]
    gt_R = gt_pose[:3, :3]

    # Relative rotation
    R_rel = pred_R @ gt_R.T

    # Extract angle from rotation matrix
    trace = np.trace(R_rel)
    # Clamp for numerical stability
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1) / 2)

    return float(np.degrees(angle_rad))


def translation_error(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
) -> float:
    """Compute translation error between poses in meters.

    Args:
        pred_pose: Predicted 4x4 pose matrix
        gt_pose: Ground truth 4x4 pose matrix

    Returns:
        Translation error in meters
    """
    pred_t = pred_pose[:3, 3]
    gt_t = gt_pose[:3, 3]

    return float(np.linalg.norm(pred_t - gt_t))


def pose_metrics(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    model_points: np.ndarray,
) -> dict:
    """Compute comprehensive pose metrics.

    Args:
        pred_pose: Predicted 4x4 pose matrix
        gt_pose: Ground truth 4x4 pose matrix
        model_points: Object model points (N, 3)

    Returns:
        Dictionary with all pose metrics
    """
    return {
        "add": add(pred_pose, gt_pose, model_points),
        "add_s": add_s(pred_pose, gt_pose, model_points),
        "rotation_error_deg": rotation_error(pred_pose, gt_pose),
        "translation_error_m": translation_error(pred_pose, gt_pose),
        "translation_error_mm": translation_error(pred_pose, gt_pose) * 1000,
    }


def evaluate_trajectory(
    pred_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    model_points: np.ndarray,
) -> dict:
    """Evaluate pose tracking over a trajectory.

    Args:
        pred_poses: List of predicted 4x4 pose matrices
        gt_poses: List of ground truth 4x4 pose matrices
        model_points: Object model points (N, 3)

    Returns:
        Dictionary with trajectory metrics
    """
    if len(pred_poses) != len(gt_poses):
        raise ValueError("Pose lists must have same length")

    if len(pred_poses) == 0:
        return {
            "mean_add": float("inf"),
            "mean_add_s": float("inf"),
            "mean_rotation_error_deg": float("inf"),
            "mean_translation_error_mm": float("inf"),
            "max_translation_error_mm": float("inf"),
        }

    metrics_list = [pose_metrics(pred, gt, model_points) for pred, gt in zip(pred_poses, gt_poses)]

    return {
        "mean_add": np.mean([m["add"] for m in metrics_list]),
        "mean_add_s": np.mean([m["add_s"] for m in metrics_list]),
        "mean_rotation_error_deg": np.mean([m["rotation_error_deg"] for m in metrics_list]),
        "mean_translation_error_mm": np.mean([m["translation_error_mm"] for m in metrics_list]),
        "max_translation_error_mm": np.max([m["translation_error_mm"] for m in metrics_list]),
        "final_add_s": metrics_list[-1]["add_s"],
        "final_translation_error_mm": metrics_list[-1]["translation_error_mm"],
    }


def generate_sphere_points(
    radius: float = 0.025,
    num_points: int = 1000,
) -> np.ndarray:
    """Generate points on a sphere surface for testing.

    Args:
        radius: Sphere radius in meters
        num_points: Number of points to generate

    Returns:
        points: Surface points (N, 3)
    """
    # Fibonacci sphere sampling for uniform distribution
    indices = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return np.stack([x, y, z], axis=1).astype(np.float32)
