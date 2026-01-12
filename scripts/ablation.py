"""Ablation study script for visuotactile perception modalities.

Compares:
- Vision-only: Uses monocular depth, no tactile fusion
- Tactile-only: Uses tactile depth, no visual depth estimation
- Visuotactile: Full fusion (baseline)

Evaluates reconstruction quality (F-score @ 5mm) on 3 objects.
Generates comparison plots and markdown table.

Usage:
    python scripts/ablation.py --steps 100 --output outputs/ablation
    python scripts/ablation.py --objects sphere box cylinder --steps 50
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from envs.allegro_hand_env import AllegroHandEnv
from perception.depth_fusion import SE3, create_default_camera_intrinsics
from perception.metrics import f_score, sample_mesh_surface
from perception.pipeline import PerceptionConfig, VisuotactilePerception

matplotlib.use("Agg")  # Non-interactive backend


def generate_primitive_mesh(
    object_name: str, resolution: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground truth mesh for primitive objects.

    Args:
        object_name: "sphere", "box", or "cylinder"
        resolution: Mesh resolution

    Returns:
        Tuple of (vertices, faces)
    """
    if object_name == "sphere":
        # Sphere radius from assets/objects/sphere.xml
        radius = 0.025  # 2.5cm
        return _generate_sphere_mesh(radius, resolution)
    elif object_name == "box":
        # Box dimensions from assets/objects/box.xml
        size = np.array([0.03, 0.05, 0.01])  # 6x10x2 cm (half-sizes)
        return _generate_box_mesh(size)
    elif object_name == "cylinder":
        # Cylinder dimensions from assets/objects/cylinder.xml
        radius = 0.035  # 7cm diameter
        height = 0.10  # 10cm height
        return _generate_cylinder_mesh(radius, height, resolution)
    else:
        raise ValueError(f"Unknown object: {object_name}")


def _generate_sphere_mesh(radius: float, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sphere mesh using UV parameterization."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Generate faces (triangulation of grid)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Two triangles per quad
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)

            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return vertices.astype(np.float32), np.array(faces, dtype=np.int32)


def _generate_box_mesh(half_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate box mesh."""
    x, y, z = half_size

    # 8 vertices of box
    vertices = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],  # Bottom
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],  # Top
        ],
        dtype=np.float32,
    )

    # 12 triangular faces (2 per side)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom
            [4, 6, 5],
            [4, 7, 6],  # Top
            [0, 5, 1],
            [0, 4, 5],  # Front
            [2, 7, 3],
            [2, 6, 7],  # Back
            [0, 7, 4],
            [0, 3, 7],  # Left
            [1, 6, 2],
            [1, 5, 6],  # Right
        ],
        dtype=np.int32,
    )

    return vertices, faces


def _generate_cylinder_mesh(
    radius: float, height: float, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cylinder mesh."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.array([-height / 2, height / 2])

    # Side vertices
    vertices = []
    for z_val in z:
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            vertices.append([x, y, z_val])

    # Center vertices for caps
    cap_bottom = [0, 0, -height / 2]
    cap_top = [0, 0, height / 2]
    vertices.extend([cap_bottom, cap_top])

    vertices = np.array(vertices, dtype=np.float32)

    # Faces
    faces = []
    n = resolution

    # Side faces
    for i in range(n):
        next_i = (i + 1) % n
        # Two triangles per side quad
        faces.append([i, next_i, i + n])
        faces.append([next_i, next_i + n, i + n])

    # Bottom cap
    cap_bottom_idx = 2 * n
    for i in range(n):
        next_i = (i + 1) % n
        faces.append([cap_bottom_idx, next_i, i])

    # Top cap
    cap_top_idx = 2 * n + 1
    for i in range(n):
        next_i = (i + 1) % n
        faces.append([cap_top_idx, i + n, next_i + n])

    return vertices, np.array(faces, dtype=np.int32)


def get_fingertip_poses(env: AllegroHandEnv) -> list[SE3]:
    """Extract fingertip poses from environment.

    Args:
        env: AllegroHandEnv instance

    Returns:
        List of 4 SE3 fingertip poses in world frame
    """
    fingertip_poses = []

    for geom_id in env.fingertip_geom_ids:
        # Get geom position and rotation
        pos = env.data.geom_xpos[geom_id].copy()
        mat = env.data.geom_xmat[geom_id].reshape(3, 3).copy()

        fingertip_poses.append(SE3(rotation=mat, translation=pos))

    return fingertip_poses


def get_camera_pose() -> SE3:
    """Get camera pose (fixed in environment).

    Returns:
        SE3 camera pose in world frame
    """
    # Camera is at fixed position looking at hand
    # From envs/assets/scene.xml camera position
    # This is a simplification - in reality would extract from MuJoCo
    return SE3(
        rotation=np.eye(3, dtype=np.float32),
        translation=np.array([0.0, 0.0, 0.5], dtype=np.float32),
    )


def run_perception_trial(
    object_name: str,
    modality: str,
    steps: int,
    device: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run perception pipeline for one modality.

    Args:
        object_name: Object to manipulate
        modality: "vision_only", "tactile_only", or "visuotactile"
        steps: Number of simulation steps
        device: Torch device

    Returns:
        Tuple of (vertices, faces) or (None, None) if failed
    """
    print(f"\n[{modality}] Running on {object_name} for {steps} steps...")

    # Configure modality
    if modality == "vision_only":
        use_visual = True
        use_tactile = False
        depth_model = "Intel/dpt-hybrid-midas"
    elif modality == "tactile_only":
        use_visual = False
        use_tactile = True
        depth_model = None  # No DPT model needed
    elif modality == "visuotactile":
        use_visual = True
        depth_model = "Intel/dpt-hybrid-midas"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Create environment
    env = AllegroHandEnv(object_name=object_name, use_grasp_stabilizer=True)

    # Create perception pipeline
    config = PerceptionConfig(
        sdf_update_freq=5,
        keyframe_min_interval=5,
        mesh_resolution=64,  # Lower resolution for speed
        device=device,
        depth_model=depth_model if use_visual else None,
    )

    perception = VisuotactilePerception(config)
    camera_intrinsics = create_default_camera_intrinsics(width=224, height=224, fov_degrees=45.0)
    perception.initialize(camera_intrinsics=camera_intrinsics)

    # Override depth fusion settings based on modality
    if perception.depth_fusion is not None:
        if modality == "vision_only":
            perception.depth_fusion.visual_weight = 1.0
            perception.depth_fusion.tactile_weight = 0.0
        elif modality == "tactile_only":
            perception.depth_fusion.visual_weight = 0.0
            perception.depth_fusion.tactile_weight = 1.0
        # visuotactile uses default weights (0.3, 1.0)

    # Run simulation
    obs, info = env.reset()
    camera_pose = get_camera_pose()

    for step in range(steps):
        # Neutral action (let grasp stabilizer handle control)
        action = np.zeros(16, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Get fingertip poses
        fingertip_poses = get_fingertip_poses(env)

        # Process frame through perception
        try:
            state = perception.process_frame(
                rgb=obs["rgb"],
                tactile=obs["tactile"],
                fingertip_poses=fingertip_poses,
                camera_pose=camera_pose,
            )
        except Exception as e:
            print(f"  Warning: Frame {step} failed: {e}")
            continue

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{steps}, loss: {state.total_loss:.4f}")

    env.close()

    # Extract final mesh
    vertices, faces = perception.get_mesh()

    if vertices is None or len(vertices) == 0:
        print(f"  [{modality}] Failed to extract mesh")
        return None, None

    print(f"  [{modality}] Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def evaluate_reconstruction(
    pred_verts: np.ndarray,
    pred_faces: np.ndarray,
    gt_verts: np.ndarray,
    gt_faces: np.ndarray,
    threshold: float = 0.005,
) -> dict:
    """Compute F-score between predicted and ground truth mesh.

    Args:
        pred_verts: Predicted vertices
        pred_faces: Predicted faces
        gt_verts: Ground truth vertices
        gt_faces: Ground truth faces
        threshold: Distance threshold (5mm default)

    Returns:
        Metrics dictionary
    """
    # Sample points
    num_samples = 10000
    pred_points = sample_mesh_surface(pred_verts, pred_faces, num_samples, seed=42)
    gt_points = sample_mesh_surface(gt_verts, gt_faces, num_samples, seed=42)

    # Compute F-score
    metrics = f_score(pred_points, gt_points, threshold)

    return metrics


def run_ablation_study(
    objects: list[str],
    steps: int,
    output_dir: Path,
    device: str = "auto",
):
    """Run full ablation study.

    Args:
        objects: List of object names to evaluate
        steps: Number of simulation steps per trial
        output_dir: Output directory for results
        device: Torch device
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["vision_only", "tactile_only", "visuotactile"]

    # Store all results
    results = {
        "objects": objects,
        "steps": steps,
        "modalities": {},
    }

    for modality in modalities:
        results["modalities"][modality] = {}

    # Run trials
    for object_name in objects:
        print(f"\n{'=' * 60}")
        print(f"OBJECT: {object_name}")
        print(f"{'=' * 60}")

        # Generate ground truth mesh
        gt_verts, gt_faces = generate_primitive_mesh(object_name, resolution=100)
        print(f"Ground truth: {len(gt_verts)} vertices, {len(gt_faces)} faces")

        for modality in modalities:
            # Run perception
            pred_verts, pred_faces = run_perception_trial(
                object_name=object_name,
                modality=modality,
                steps=steps,
                device=device,
            )

            # Evaluate
            if pred_verts is not None:
                metrics = evaluate_reconstruction(pred_verts, pred_faces, gt_verts, gt_faces)

                print(f"  [{modality}] F-score: {metrics['f_score']:.1%}")
                print(f"  [{modality}] Precision: {metrics['precision']:.1%}")
                print(f"  [{modality}] Recall: {metrics['recall']:.1%}")

                results["modalities"][modality][object_name] = metrics
            else:
                # Failed trial
                results["modalities"][modality][object_name] = {
                    "f_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "error": "mesh_extraction_failed",
                }

    # Save raw metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Generate comparison plot
    plot_path = output_dir / "comparison.png"
    generate_comparison_plot(results, plot_path)
    print(f"Comparison plot saved to: {plot_path}")

    # Generate markdown table
    table_path = output_dir / "table.md"
    generate_markdown_table(results, table_path)
    print(f"Markdown table saved to: {table_path}")

    # Print summary
    print_summary(results)


def generate_comparison_plot(results: dict, output_path: Path):
    """Generate bar chart comparing F-scores across modalities.

    Args:
        results: Results dictionary
        output_path: Path to save plot
    """
    objects = results["objects"]
    modalities = list(results["modalities"].keys())

    # Extract F-scores
    data = []
    for modality in modalities:
        scores = []
        for obj in objects:
            metrics = results["modalities"][modality].get(obj, {})
            scores.append(metrics.get("f_score", 0.0) * 100)  # Convert to percentage
        data.append(scores)

    # Plot
    x = np.arange(len(objects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3498db", "#e74c3c", "#2ecc71"]  # Blue, Red, Green
    labels = ["Vision-only", "Tactile-only", "Visuotactile"]

    for i, (scores, color, label) in enumerate(zip(data, colors, labels)):
        offset = (i - 1) * width
        ax.bar(x + offset, scores, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Object", fontsize=12, fontweight="bold")
    ax.set_ylabel("F-Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Visuotactile Perception Ablation Study\nF-Score @ 5mm Threshold",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([obj.capitalize() for obj in objects])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for i, scores in enumerate(data):
        offset = (i - 1) * width
        for j, score in enumerate(scores):
            if score > 0:
                ax.text(
                    x[j] + offset, score + 2, f"{score:.1f}", ha="center", va="bottom", fontsize=9
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_markdown_table(results: dict, output_path: Path):
    """Generate markdown table for README.

    Args:
        results: Results dictionary
        output_path: Path to save table
    """
    objects = results["objects"]
    modalities = list(results["modalities"].keys())

    # Calculate averages
    averages = {}
    for modality in modalities:
        scores = []
        for obj in objects:
            metrics = results["modalities"][modality].get(obj, {})
            scores.append(metrics.get("f_score", 0.0) * 100)
        averages[modality] = np.mean(scores)

    # Build table
    lines = []
    lines.append("# Ablation Study Results\n")
    lines.append(f"**Configuration**: {results['steps']} simulation steps per object\n")
    lines.append("**Metric**: F-Score @ 5mm threshold\n")
    lines.append("")
    lines.append(
        "| Modality | " + " | ".join([obj.capitalize() for obj in objects]) + " | Average |"
    )
    lines.append("|" + "|".join(["-" * 10 for _ in range(len(objects) + 2)]) + "|")

    modality_names = {
        "vision_only": "Vision-only",
        "tactile_only": "Tactile-only",
        "visuotactile": "Visuotactile",
    }

    for modality in modalities:
        row = [modality_names[modality]]
        for obj in objects:
            metrics = results["modalities"][modality].get(obj, {})
            score = metrics.get("f_score", 0.0) * 100
            row.append(f"{score:.1f}%")
        row.append(f"**{averages[modality]:.1f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Winner")
    best_modality = max(averages, key=averages.get)
    lines.append(
        f"**{modality_names[best_modality]}** achieves the highest average F-score: "
        f"**{averages[best_modality]:.1f}%**"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def print_summary(results: dict):
    """Print summary to console.

    Args:
        results: Results dictionary
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    objects = results["objects"]
    modalities = list(results["modalities"].keys())

    modality_names = {
        "vision_only": "Vision-only",
        "tactile_only": "Tactile-only",
        "visuotactile": "Visuotactile",
    }

    # Calculate averages
    averages = {}
    for modality in modalities:
        scores = []
        for obj in objects:
            metrics = results["modalities"][modality].get(obj, {})
            scores.append(metrics.get("f_score", 0.0) * 100)
        averages[modality] = np.mean(scores)

    # Print
    for modality in modalities:
        print(f"\n{modality_names[modality]}:")
        for obj in objects:
            metrics = results["modalities"][modality].get(obj, {})
            score = metrics.get("f_score", 0.0) * 100
            print(f"  {obj.capitalize()}: {score:.1f}%")
        print(f"  Average: {averages[modality]:.1f}%")

    # Winner
    best_modality = max(averages, key=averages.get)
    print(f"\n{'=' * 60}")
    print(f"WINNER: {modality_names[best_modality]} ({averages[best_modality]:.1f}%)")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for visuotactile perception modalities"
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["sphere", "box", "cylinder"],
        help="Objects to evaluate (default: sphere box cylinder)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps per trial (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ablation",
        help="Output directory for results (default: outputs/ablation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (auto, cpu, cuda, xpu) (default: auto)",
    )

    args = parser.parse_args()

    run_ablation_study(
        objects=args.objects,
        steps=args.steps,
        output_dir=Path(args.output),
        device=args.device,
    )


if __name__ == "__main__":
    main()
