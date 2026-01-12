import numpy as np
import pytest

from envs.grasp_controller import GraspStabilizer


@pytest.fixture
def stabilizer():
    return GraspStabilizer(
        num_fingers=4,
        target_force=0.3,
        kp=2.0,
        kd=0.1,
        force_threshold_low=0.1,
        force_threshold_high=0.8,
    )


def test_grasp_stabilizer_init():
    stabilizer = GraspStabilizer()

    assert stabilizer.num_fingers == 4
    assert stabilizer.target_force == 0.3
    assert stabilizer.kp == 2.0
    assert stabilizer.kd == 0.1
    assert stabilizer.prev_forces.shape == (4,)
    assert stabilizer.no_contact_frames == 0
    assert stabilizer.drop_detected is False


def test_get_finger_forces_no_contact(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)

    forces = stabilizer.get_finger_forces(tactile)

    assert forces.shape == (4,)
    assert np.all(forces == 0.0)


def test_get_finger_forces_with_contact(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)
    tactile[0, 15:17, 15:17] = 0.8
    tactile[2, 10:12, 10:12] = 0.5

    forces = stabilizer.get_finger_forces(tactile)

    assert forces.shape == (4,)
    assert forces[0] > 0
    assert forces[1] == 0
    assert forces[2] > 0
    assert forces[3] == 0


def test_compute_grip_adjustment_shape(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)

    adjustment = stabilizer.compute_grip_adjustment(tactile)

    assert adjustment.shape == (16,)
    assert adjustment.dtype == np.float32


def test_compute_grip_adjustment_no_contact(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)

    adjustment = stabilizer.compute_grip_adjustment(tactile)

    for finger in range(4):
        base_idx = finger * 4
        curl_values = [adjustment[base_idx + 1], adjustment[base_idx + 2], adjustment[base_idx + 3]]
        assert all(v > 0 for v in curl_values)


def test_compute_grip_adjustment_high_contact(stabilizer):
    tactile = np.ones((4, 32, 32), dtype=np.float32)

    adjustment = stabilizer.compute_grip_adjustment(tactile)

    for finger in range(4):
        base_idx = finger * 4
        curl_values = [adjustment[base_idx + 1], adjustment[base_idx + 2], adjustment[base_idx + 3]]
        assert all(v < 0 for v in curl_values)


def test_drop_detection(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)

    for _ in range(3):
        stabilizer.compute_grip_adjustment(tactile)
        assert stabilizer.is_drop_detected() is False

    for _ in range(5):
        stabilizer.compute_grip_adjustment(tactile)

    assert stabilizer.is_drop_detected() is True


def test_slip_detection(stabilizer):
    tactile1 = np.zeros((4, 32, 32), dtype=np.float32)
    tactile1[0, 10:20, 10:20] = 0.8

    tactile2 = np.zeros((4, 32, 32), dtype=np.float32)
    tactile2[0, 20:30, 20:30] = 0.8

    stabilizer.compute_grip_adjustment(tactile1, enable_slip_detection=True)

    slip = stabilizer.detect_slip(tactile2)

    assert slip.shape == (4,)
    assert slip[0]


def test_reset(stabilizer):
    tactile = np.ones((4, 32, 32), dtype=np.float32)

    stabilizer.compute_grip_adjustment(tactile)
    assert np.any(stabilizer.prev_forces > 0)

    stabilizer.reset()

    assert np.all(stabilizer.prev_forces == 0)
    assert stabilizer.prev_tactile is None
    assert stabilizer.no_contact_frames == 0
    assert stabilizer.drop_detected is False


def test_get_state(stabilizer):
    tactile = np.ones((4, 32, 32), dtype=np.float32) * 0.5

    stabilizer.compute_grip_adjustment(tactile)

    state = stabilizer.get_state()

    assert "forces" in state
    assert "no_contact_frames" in state
    assert "drop_detected" in state
    assert state["forces"].shape == (4,)


def test_joint_mapping_correct(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)
    tactile[1] = 0.1

    adjustment = stabilizer.compute_grip_adjustment(tactile)

    finger_1_curl_joints = [adjustment[5], adjustment[6], adjustment[7]]
    assert all(v != 0 for v in finger_1_curl_joints)

    finger_1_spread_joint = adjustment[4]
    assert finger_1_spread_joint == 0


def test_max_adjustment_clipping(stabilizer):
    tactile = np.zeros((4, 32, 32), dtype=np.float32)

    adjustment = stabilizer.compute_grip_adjustment(tactile)

    assert np.all(np.abs(adjustment) <= stabilizer.max_adjustment)
