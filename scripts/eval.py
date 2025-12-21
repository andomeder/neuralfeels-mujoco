"""Evaluation script for visuotactile perception.

Evaluates reconstruction quality (F-score) and pose tracking accuracy (ADD-S)
on collected episodes.

Usage:
    python scripts/eval.py --checkpoint outputs/checkpoints/final.pt --episodes datasets
    python scripts/eval.py --mesh outputs/final_mesh.npz --gt-mesh meshes/object.obj
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from perception.metrics import (
    add_s,
    chamfer_distance,
    evaluate_trajectory,
    f_score,
    sample_mesh_surface,
)


def load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from npz file.

    Args:
        path: Path to npz file

    Returns:
        Tuple of (vertices, faces)
    """
    data = np.load(path)
    return data["vertices"], data["faces"]


def load_mesh_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from OBJ file.

    Args:
        path: Path to OBJ file

    Returns:
        Tuple of (vertices, faces)
    """
    vertices = []
    faces = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Handle face indices (OBJ is 1-indexed)
                face = []
                for p in parts[1:]:
                    # Handle format: v, v/vt, v/vt/vn, v//vn
                    idx = int(p.split("/")[0]) - 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face[:3])  # Triangulate if needed

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from file (auto-detect format).

    Args:
        path: Path to mesh file

    Returns:
        Tuple of (vertices, faces)
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_mesh_npz(path)
    elif suffix == ".obj":
        return load_mesh_obj(path)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")


def evaluate_reconstruction(
    pred_mesh_path: Path,
    gt_mesh_path: Optional[Path] = None,
    num_samples: int = 10000,
    threshold: float = 0.005,  # 5mm
) -> dict:
    """Evaluate mesh reconstruction quality.

    Args:
        pred_mesh_path: Path to predicted mesh
        gt_mesh_path: Path to ground truth mesh (optional)
        num_samples: Number of points to sample
        threshold: Distance threshold for F-score

    Returns:
        Dictionary of metrics
    """
    # Load predicted mesh
    pred_verts, pred_faces = load_mesh(pred_mesh_path)
    print(f"Predicted mesh: {len(pred_verts)} vertices, {len(pred_faces)} faces")

    if len(pred_verts) == 0:
        return {
            "error": "Empty predicted mesh",
            "f_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Sample points from predicted mesh
    pred_points = sample_mesh_surface(pred_verts, pred_faces, num_samples)

    if gt_mesh_path is not None and gt_mesh_path.exists():
        # Load ground truth mesh
        gt_verts, gt_faces = load_mesh(gt_mesh_path)
        print(f"Ground truth mesh: {len(gt_verts)} vertices, {len(gt_faces)} faces")

        gt_points = sample_mesh_surface(gt_verts, gt_faces, num_samples)

        # Compute F-score
        precision, recall, fscore = f_score(pred_points, gt_points, threshold)

        # Compute Chamfer distance
        chamfer = chamfer_distance(pred_points, gt_points)

        return {
            "f_score": float(fscore),
            "precision": float(precision),
            "recall": float(recall),
            "chamfer_distance": float(chamfer),
            "pred_vertices": len(pred_verts),
            "pred_faces": len(pred_faces),
            "gt_vertices": len(gt_verts),
            "gt_faces": len(gt_faces),
            "threshold_mm": threshold * 1000,
        }
    else:
        # No ground truth - just report mesh stats
        return {
            "pred_vertices": len(pred_verts),
            "pred_faces": len(pred_faces),
            "note": "No ground truth mesh provided",
        }


def evaluate_pose_tracking(
    pred_poses_path: Path,
    gt_poses_path: Path,
    mesh_path: Optional[Path] = None,
) -> dict:
    """Evaluate pose tracking accuracy.

    Args:
        pred_poses_path: Path to predicted poses (N, 4, 4) npz
        gt_poses_path: Path to ground truth poses
        mesh_path: Optional path to object mesh for ADD-S

    Returns:
        Dictionary of metrics
    """
    # Load poses
    pred_data = np.load(pred_poses_path)
    gt_data = np.load(gt_poses_path)

    pred_poses = pred_data["poses"] if "poses" in pred_data else pred_data["arr_0"]
    gt_poses = gt_data["poses"] if "poses" in gt_data else gt_data["arr_0"]

    print(f"Predicted poses: {len(pred_poses)}")
    print(f"Ground truth poses: {len(gt_poses)}")

    # Compute metrics using evaluate_trajectory
    num_frames = min(len(pred_poses), len(gt_poses))
    if num_frames == 0:
        return {"error": "No poses to evaluate"}

    # Convert to list format expected by evaluate_trajectory
    pred_list = [pred_poses[i] for i in range(num_frames)]
    gt_list = [gt_poses[i] for i in range(num_frames)]

    if mesh_path is not None and mesh_path.exists():
        verts, faces = load_mesh(mesh_path)
        points = sample_mesh_surface(verts, faces, 1000)
    else:
        # Use unit sphere points as default
        theta = np.random.uniform(0, 2 * np.pi, 1000)
        phi = np.random.uniform(0, np.pi, 1000)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.stack([x, y, z], axis=-1).astype(np.float32) * 0.03  # 3cm radius

    metrics = evaluate_trajectory(pred_list, gt_list, points)

    return {
        "add_s_mean_mm": float(metrics["add_s_mean"]) * 1000,
        "add_s_median_mm": float(metrics["add_s_median"]) * 1000,
        "translation_error_mean_mm": float(metrics["translation_error_mean"]) * 1000,
        "translation_error_max_mm": float(metrics["translation_error_max"]) * 1000,
        "rotation_error_mean_deg": float(metrics["rotation_error_mean"]),
        "rotation_error_max_deg": float(metrics["rotation_error_max"]),
        "num_frames": num_frames,
    }


def create_evaluation_report(
    output_dir: Path,
    reconstruction_metrics: Optional[dict] = None,
    tracking_metrics: Optional[dict] = None,
) -> str:
    """Create a formatted evaluation report.

    Args:
        output_dir: Directory to save report
        reconstruction_metrics: Reconstruction evaluation results
        tracking_metrics: Pose tracking evaluation results

    Returns:
        Report string
    """
    report_lines = [
        "=" * 60,
        "NEURALFEELS-MUJOCO EVALUATION REPORT",
        "=" * 60,
        "",
    ]

    if reconstruction_metrics:
        report_lines.extend(
            [
                "RECONSTRUCTION QUALITY",
                "-" * 40,
            ]
        )

        if "f_score" in reconstruction_metrics:
            report_lines.extend(
                [
                    f"  F-score @ {reconstruction_metrics.get('threshold_mm', 5)}mm: "
                    f"{reconstruction_metrics['f_score']:.1%}",
                    f"  Precision: {reconstruction_metrics['precision']:.1%}",
                    f"  Recall: {reconstruction_metrics['recall']:.1%}",
                ]
            )

        if "chamfer_distance" in reconstruction_metrics:
            report_lines.append(
                f"  Chamfer Distance: {reconstruction_metrics['chamfer_distance'] * 1000:.2f} mm"
            )

        report_lines.extend(
            [
                f"  Mesh Vertices: {reconstruction_metrics.get('pred_vertices', 'N/A')}",
                f"  Mesh Faces: {reconstruction_metrics.get('pred_faces', 'N/A')}",
                "",
            ]
        )

    if tracking_metrics:
        report_lines.extend(
            [
                "POSE TRACKING ACCURACY",
                "-" * 40,
                f"  ADD-S Mean: {tracking_metrics['add_s_mean_mm']:.2f} mm",
                f"  ADD-S Median: {tracking_metrics['add_s_median_mm']:.2f} mm",
                f"  Translation Error Mean: {tracking_metrics['translation_error_mean_mm']:.2f} mm",
                f"  Translation Error Max: {tracking_metrics['translation_error_max_mm']:.2f} mm",
                f"  Rotation Error Mean: {tracking_metrics['rotation_error_mean_deg']:.2f}°",
                f"  Rotation Error Max: {tracking_metrics['rotation_error_max_deg']:.2f}°",
                f"  Frames Evaluated: {tracking_metrics['num_frames']}",
                "",
            ]
        )

    # Success criteria check
    report_lines.extend(
        [
            "SUCCESS CRITERIA CHECK",
            "-" * 40,
        ]
    )

    if reconstruction_metrics and "f_score" in reconstruction_metrics:
        fscore_pass = reconstruction_metrics["f_score"] >= 0.6
        report_lines.append(
            f"  [{'✓' if fscore_pass else '✗'}] F-score ≥ 60%: "
            f"{reconstruction_metrics['f_score']:.1%}"
        )

    if tracking_metrics:
        drift_pass = tracking_metrics["add_s_mean_mm"] < 10
        report_lines.append(
            f"  [{'✓' if drift_pass else '✗'}] Pose drift < 10mm: "
            f"{tracking_metrics['add_s_mean_mm']:.2f} mm"
        )

    report_lines.extend(
        [
            "",
            "=" * 60,
        ]
    )

    report = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return report


def run_full_evaluation(
    checkpoint_path: Optional[Path] = None,
    mesh_path: Optional[Path] = None,
    gt_mesh_path: Optional[Path] = None,
    poses_path: Optional[Path] = None,
    gt_poses_path: Optional[Path] = None,
    output_dir: Path = Path("outputs/metrics"),
):
    """Run full evaluation pipeline.

    Args:
        checkpoint_path: Path to perception checkpoint
        mesh_path: Path to predicted mesh
        gt_mesh_path: Path to ground truth mesh
        poses_path: Path to predicted poses
        gt_poses_path: Path to ground truth poses
        output_dir: Output directory for results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_metrics = None
    tracking_metrics = None

    # Evaluate reconstruction
    if mesh_path is not None and mesh_path.exists():
        print("\nEvaluating reconstruction quality...")
        reconstruction_metrics = evaluate_reconstruction(
            pred_mesh_path=mesh_path,
            gt_mesh_path=gt_mesh_path,
        )
        print(f"  F-score: {reconstruction_metrics.get('f_score', 'N/A')}")

    # Evaluate pose tracking
    if poses_path is not None and gt_poses_path is not None:
        if poses_path.exists() and gt_poses_path.exists():
            print("\nEvaluating pose tracking...")
            tracking_metrics = evaluate_pose_tracking(
                pred_poses_path=poses_path,
                gt_poses_path=gt_poses_path,
                mesh_path=gt_mesh_path,
            )
            print(f"  ADD-S: {tracking_metrics.get('add_s_mean_mm', 'N/A')} mm")

    # Generate report
    report = create_evaluation_report(
        output_dir=output_dir,
        reconstruction_metrics=reconstruction_metrics,
        tracking_metrics=tracking_metrics,
    )
    print("\n" + report)

    # Save all metrics as JSON
    all_metrics = {
        "reconstruction": reconstruction_metrics,
        "tracking": tracking_metrics,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate visuotactile perception")
    parser.add_argument(
        "--mesh",
        type=str,
        default="outputs/final_mesh.npz",
        help="Path to predicted mesh",
    )
    parser.add_argument(
        "--gt-mesh",
        type=str,
        default=None,
        help="Path to ground truth mesh",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Path to predicted poses (.npz)",
    )
    parser.add_argument(
        "--gt-poses",
        type=str,
        default=None,
        help="Path to ground truth poses (.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metrics",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to perception checkpoint (optional)",
    )
    args = parser.parse_args()

    run_full_evaluation(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        mesh_path=Path(args.mesh) if args.mesh else None,
        gt_mesh_path=Path(args.gt_mesh) if args.gt_mesh else None,
        poses_path=Path(args.poses) if args.poses else None,
        gt_poses_path=Path(args.gt_poses) if args.gt_poses else None,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
