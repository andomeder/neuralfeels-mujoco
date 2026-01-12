#!/usr/bin/env python
"""Visualize all available objects in the Allegro Hand environment."""

import argparse
import cv2
import numpy as np
from envs.allegro_hand_env import AllegroHandEnv


def create_grid_visualization(images, labels, grid_size=(3, 3)):
    n_images = len(images)
    rows, cols = grid_size

    h, w = images[0].shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols

        img_with_label = img.copy()
        cv2.putText(
            img_with_label, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = img_with_label

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize object diversity")
    parser.add_argument(
        "--output", type=str, default="outputs/objects_showcase.png", help="Output image path"
    )
    args = parser.parse_args()

    print("Generating object showcase...")

    images = []
    labels = []

    for obj_name in AllegroHandEnv.OBJECT_SET:
        print(f"  Rendering {obj_name}...")
        env = AllegroHandEnv(object_name=obj_name)
        obs, _ = env.reset()

        for _ in range(5):
            obs, _, _, _, _ = env.step(np.zeros(16))

        images.append(obs["rgb"])
        labels.append(obj_name)
        env.close()

    grid = create_grid_visualization(images, labels, grid_size=(3, 3))

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved showcase to {args.output}")


if __name__ == "__main__":
    main()
