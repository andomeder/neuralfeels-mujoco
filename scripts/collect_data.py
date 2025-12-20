import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from envs.allegro_hand_env import AllegroHandEnv
from envs.tactile_sim import visualize_tactile


@dataclass
class RotationPolicy:
    frequency: float = 0.5
    amplitude: np.ndarray = None
    phase_offset: np.ndarray = None
    base_position: np.ndarray = None

    def __post_init__(self):
        if self.amplitude is None:
            self.amplitude = np.array(
                [
                    0.1,
                    0.3,
                    0.3,
                    0.3,
                    0.1,
                    0.3,
                    0.3,
                    0.3,
                    0.1,
                    0.3,
                    0.3,
                    0.3,
                    0.2,
                    0.2,
                    0.3,
                    0.3,
                ]
            )
        if self.phase_offset is None:
            self.phase_offset = (
                np.array(
                    [
                        0,
                        0,
                        0.5,
                        1.0,
                        0.25,
                        0.25,
                        0.75,
                        1.25,
                        0.5,
                        0.5,
                        1.0,
                        1.5,
                        0,
                        0.5,
                        0.5,
                        1.0,
                    ]
                )
                * np.pi
            )
        if self.base_position is None:
            self.base_position = np.array(
                [
                    0,
                    0.7,
                    0.7,
                    0.5,
                    0,
                    0.7,
                    0.7,
                    0.5,
                    0,
                    0.7,
                    0.7,
                    0.5,
                    0.8,
                    0.5,
                    0.7,
                    0.5,
                ]
            )

    def get_action(self, t: float) -> np.ndarray:
        omega = 2 * np.pi * self.frequency
        positions = self.base_position + self.amplitude * np.sin(omega * t + self.phase_offset)
        return positions


def collect_episode(
    env: AllegroHandEnv,
    policy: RotationPolicy,
    episode_dir: Path,
    max_steps: int = 200,
    save_images: bool = True,
) -> dict:
    episode_dir.mkdir(parents=True, exist_ok=True)

    if save_images:
        (episode_dir / "rgb").mkdir(exist_ok=True)
        (episode_dir / "tactile").mkdir(exist_ok=True)

    obs, info = env.reset()

    rgb_frames = []
    tactile_frames = []
    qpos_history = []
    qvel_history = []
    object_pos_history = []
    object_quat_history = []
    rewards = []

    for step in range(max_steps):
        t = step / env.control_freq
        action = policy.get_action(t)

        action_normalized = (action - 0.5) * 2

        obs, reward, terminated, truncated, info = env.step(action_normalized)

        rgb_frames.append(obs["rgb"])
        tactile_frames.append(obs["tactile"])
        qpos_history.append(obs["qpos"].copy())
        qvel_history.append(obs["qvel"].copy())
        object_pos_history.append(info["object_pos"].copy())
        object_quat_history.append(info["object_quat"].copy())
        rewards.append(reward)

        if save_images:
            cv2.imwrite(
                str(episode_dir / "rgb" / f"{step:04d}.png"),
                cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR),
            )
            np.save(episode_dir / "tactile" / f"{step:04d}.npy", obs["tactile"])

        if terminated:
            break

    np.save(episode_dir / "qpos.npy", np.array(qpos_history))
    np.save(episode_dir / "qvel.npy", np.array(qvel_history))
    np.save(episode_dir / "object_pos.npy", np.array(object_pos_history))
    np.save(episode_dir / "object_quat.npy", np.array(object_quat_history))
    np.save(episode_dir / "rewards.npy", np.array(rewards))

    metadata = {
        "num_steps": len(qpos_history),
        "total_reward": float(sum(rewards)),
        "control_freq": env.control_freq,
        "policy_frequency": policy.frequency,
    }

    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = AllegroHandEnv(render_mode="rgb_array")
    policy = RotationPolicy()

    print(f"Collecting {args.num_episodes} episodes...")

    for ep_idx in range(args.num_episodes):
        episode_dir = output_dir / f"episode_{ep_idx:03d}"

        print(f"Episode {ep_idx + 1}/{args.num_episodes}...", end=" ")

        metadata = collect_episode(
            env=env,
            policy=policy,
            episode_dir=episode_dir,
            max_steps=args.steps_per_episode,
            save_images=True,
        )

        print(f"steps={metadata['num_steps']}, reward={metadata['total_reward']:.2f}")

        if args.visualize and ep_idx == 0:
            tactile = np.load(episode_dir / "tactile" / "0100.npy")
            viz = visualize_tactile(tactile, ["Index", "Middle", "Ring", "Thumb"])
            cv2.imwrite(str(output_dir / "tactile_sample.png"), viz)

    env.close()
    print(f"\nDone! Episodes saved to {output_dir}")


if __name__ == "__main__":
    main()
