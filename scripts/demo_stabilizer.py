#!/usr/bin/env python3
"""
Demo script showing grasp stabilizer in action.

Compares two scenarios:
1. Without stabilizer: Object likely drops during rotation
2. With stabilizer: Object maintained in hand via tactile feedback

Usage:
    python scripts/demo_stabilizer.py --mode side_by_side
    python scripts/demo_stabilizer.py --mode stabilizer_only --steps 600
"""

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np

from envs.allegro_hand_env import AllegroHandEnv
from envs.tactile_sim import visualize_tactile


def rotation_policy(t: int, amplitude: float = 0.3, freq: float = 0.1) -> np.ndarray:
    action = np.zeros(16)
    phase = 2 * np.pi * freq * t

    for finger in range(4):
        base_idx = finger * 4
        action[base_idx + 1] = amplitude * np.sin(phase)
        action[base_idx + 2] = amplitude * np.sin(phase + np.pi / 4)
        action[base_idx + 3] = amplitude * np.sin(phase + np.pi / 2)

    return action


def run_episode(
    env: AllegroHandEnv,
    steps: int,
    policy_fn=rotation_policy,
) -> tuple[list, list, list]:
    obs, info = env.reset()

    frames = []
    tactile_frames = []
    contact_forces_history = []

    for t in range(steps):
        action = policy_fn(t)
        obs, reward, terminated, truncated, info = env.step(action)

        rgb = obs["rgb"]
        tactile = obs["tactile"]

        contact_forces = env.get_contact_forces()
        contact_forces_history.append(contact_forces.copy())

        tactile_vis = visualize_tactile(
            tactile,
            finger_names=["Index", "Middle", "Ring", "Thumb"],
            contact_forces=(
                contact_forces / contact_forces.max() if contact_forces.max() > 0 else None
            ),
        )

        frames.append(rgb)
        tactile_frames.append(tactile_vis)

        if terminated or truncated:
            break

    env.close()
    return frames, tactile_frames, contact_forces_history


def create_side_by_side_comparison(
    steps: int = 300,
    output_path: str = "outputs/stabilizer_comparison.mp4",
):
    print("Running WITHOUT stabilizer...")
    env_no_stab = AllegroHandEnv(use_grasp_stabilizer=False)
    frames_no, tactile_no, forces_no = run_episode(env_no_stab, steps)

    print("Running WITH stabilizer...")
    env_with_stab = AllegroHandEnv(use_grasp_stabilizer=True, target_force=0.4, stabilizer_kp=3.0)
    frames_with, tactile_with, forces_with = run_episode(env_with_stab, steps)

    print(f"Creating comparison video at {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=20)

    for i in range(min(len(frames_no), len(frames_with))):
        rgb_no = cv2.cvtColor(frames_no[i], cv2.COLOR_RGB2BGR)
        rgb_with = cv2.cvtColor(frames_with[i], cv2.COLOR_RGB2BGR)
        tactile_no_bgr = tactile_no[i]
        tactile_with_bgr = tactile_with[i]

        cv2.putText(
            rgb_no,
            "Without Stabilizer",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            rgb_with,
            "With Stabilizer",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        top_row = np.hstack([rgb_no, rgb_with])
        bottom_row = np.hstack([tactile_no_bgr, tactile_with_bgr])

        top_row_resized = cv2.resize(top_row, (bottom_row.shape[1], top_row.shape[0]))

        combined = np.vstack([top_row_resized, bottom_row])

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)

    writer.close()

    avg_force_no = np.mean([f.mean() for f in forces_no])
    avg_force_with = np.mean([f.mean() for f in forces_with])

    print(f"\n{'=' * 60}")
    print(f"Results ({len(frames_no)} frames):")
    print(f"  Without Stabilizer - Avg contact force: {avg_force_no:.3f}")
    print(f"  With Stabilizer    - Avg contact force: {avg_force_with:.3f}")
    print(f"  Improvement: {(avg_force_with / avg_force_no - 1) * 100:.1f}%")
    print(f"{'=' * 60}")
    print(f"\nVideo saved to: {output_path}")


def create_stabilizer_demo(
    steps: int = 600,
    output_path: str = "outputs/stabilizer_demo.mp4",
):
    print("Running demo with grasp stabilizer...")
    env = AllegroHandEnv(use_grasp_stabilizer=True, target_force=0.4, stabilizer_kp=3.0)
    frames, tactile_frames, contact_forces_history = run_episode(env, steps)

    print(f"Creating demo video at {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=20)

    for i, (rgb, tactile_vis) in enumerate(zip(frames, tactile_frames)):
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        forces = contact_forces_history[i]
        avg_force = forces.mean()
        max_force = forces.max()

        cv2.putText(
            rgb_bgr,
            "Grasp Stabilizer Active",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            rgb_bgr,
            f"Avg Force: {avg_force:.3f}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            rgb_bgr,
            f"Max Force: {max_force:.3f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        rgb_resized = cv2.resize(rgb_bgr, (tactile_vis.shape[1], rgb_bgr.shape[0]))

        combined = np.vstack([rgb_resized, tactile_vis])

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)

    writer.close()

    avg_force = np.mean([f.mean() for f in contact_forces_history])
    min_force = np.min([f.min() for f in contact_forces_history])
    max_force = np.max([f.max() for f in contact_forces_history])

    print(f"\n{'=' * 60}")
    print(f"Stabilizer Demo ({len(frames)} frames, {len(frames) / 20:.1f}s):")
    print(f"  Avg contact force: {avg_force:.3f}")
    print(f"  Min contact force: {min_force:.3f}")
    print(f"  Max contact force: {max_force:.3f}")
    print(f"{'=' * 60}")
    print(f"\nVideo saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grasp stabilizer demo")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["side_by_side", "stabilizer_only"],
        default="side_by_side",
        help="Demo mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path",
    )

    args = parser.parse_args()

    if args.mode == "side_by_side":
        output = args.output or "outputs/stabilizer_comparison.mp4"
        create_side_by_side_comparison(steps=args.steps, output_path=output)
    else:
        output = args.output or "outputs/stabilizer_demo.mp4"
        create_stabilizer_demo(steps=args.steps, output_path=output)


if __name__ == "__main__":
    main()
