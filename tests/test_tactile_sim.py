import numpy as np


def test_tactile_depth_empty():
    from envs.tactile_sim import get_tactile_depth

    depth = get_tactile_depth(
        contact_positions=[],
        contact_forces=[],
        fingertip_pos=np.zeros(3),
        fingertip_mat=np.eye(3),
    )

    assert depth.shape == (32, 32)
    assert depth.sum() == 0


def test_tactile_depth_single_contact():
    from envs.tactile_sim import get_tactile_depth

    fingertip_pos = np.array([0, 0, 0])
    fingertip_mat = np.eye(3)
    contact_pos = np.array([0, 0, 0])

    depth = get_tactile_depth(
        contact_positions=[contact_pos],
        contact_forces=[5.0],
        fingertip_pos=fingertip_pos,
        fingertip_mat=fingertip_mat,
        apply_blur=False,
    )

    assert depth.shape == (32, 32)
    assert depth.max() == 0.5
    assert depth[16, 16] == 0.5


def test_tactile_depth_with_blur():
    from envs.tactile_sim import get_tactile_depth

    fingertip_pos = np.array([0, 0, 0])
    fingertip_mat = np.eye(3)
    contact_pos = np.array([0, 0, 0])

    depth = get_tactile_depth(
        contact_positions=[contact_pos],
        contact_forces=[5.0],
        fingertip_pos=fingertip_pos,
        fingertip_mat=fingertip_mat,
        apply_blur=True,
    )

    assert depth.shape == (32, 32)
    assert depth.max() == 1.0
    assert depth.sum() > 0


def test_visualize_tactile():
    from envs.tactile_sim import visualize_tactile

    tactile = np.random.rand(4, 32, 32).astype(np.float32)
    grid = visualize_tactile(tactile)

    assert grid.shape[0] == 96
    assert grid.shape[1] == 96 * 4
    assert grid.shape[2] == 3


if __name__ == "__main__":
    test_tactile_depth_empty()
    print("✓ test_tactile_depth_empty passed")

    test_tactile_depth_single_contact()
    print("✓ test_tactile_depth_single_contact passed")

    test_tactile_depth_with_blur()
    print("✓ test_tactile_depth_with_blur passed")

    test_visualize_tactile()
    print("✓ test_visualize_tactile passed")

    print("\nAll tactile tests passed!")
