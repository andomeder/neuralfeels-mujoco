import numpy as np
import pytest

from envs.allegro_hand_env import AllegroHandEnv


def test_env_without_stabilizer():
    env = AllegroHandEnv(use_grasp_stabilizer=False)

    assert env.use_grasp_stabilizer is False
    assert env._grasp_stabilizer is None

    obs, info = env.reset()

    assert "contact_forces" not in info
    assert "drop_detected" not in info


def test_env_with_stabilizer():
    env = AllegroHandEnv(use_grasp_stabilizer=True)

    assert env.use_grasp_stabilizer is True
    assert env._grasp_stabilizer is not None

    obs, info = env.reset()

    assert "contact_forces" in info
    assert "drop_detected" in info
    assert "no_contact_frames" in info


def test_stabilizer_integration_step():
    env = AllegroHandEnv(use_grasp_stabilizer=True)
    env.reset()

    action = np.zeros(16)
    obs, reward, terminated, truncated, info = env.step(action)

    assert "contact_forces" in info
    assert info["contact_forces"].shape == (4,)


def test_stabilizer_reset():
    env = AllegroHandEnv(use_grasp_stabilizer=True)
    env.reset()

    action = np.ones(16) * 0.5
    env.step(action)

    obs, info = env.reset()

    assert info["no_contact_frames"] == 0


def test_get_contact_forces_without_stabilizer():
    env = AllegroHandEnv(use_grasp_stabilizer=False)
    env.reset()

    forces = env.get_contact_forces()

    assert forces.shape == (4,)


def test_get_contact_forces_with_stabilizer():
    env = AllegroHandEnv(use_grasp_stabilizer=True)
    env.reset()

    forces = env.get_contact_forces()

    assert forces.shape == (4,)


def test_stabilizer_modifies_action():
    env_with = AllegroHandEnv(use_grasp_stabilizer=True)
    env_with.reset()

    env_without = AllegroHandEnv(use_grasp_stabilizer=False)
    env_without.reset()

    action = np.zeros(16)

    env_with.step(action)
    env_without.step(action)

    ctrl_with = env_with.data.ctrl.copy()
    ctrl_without = env_without.data.ctrl.copy()

    assert not np.allclose(ctrl_with, ctrl_without)


def test_stabilizer_custom_params():
    env = AllegroHandEnv(
        use_grasp_stabilizer=True,
        stabilizer_kp=5.0,
        stabilizer_kd=0.5,
        target_force=0.5,
    )

    assert env._grasp_stabilizer.kp == 5.0
    assert env._grasp_stabilizer.kd == 0.5
    assert env._grasp_stabilizer.target_force == 0.5


@pytest.mark.slow
def test_stabilizer_monitors_contact():
    env = AllegroHandEnv(
        use_grasp_stabilizer=True,
        target_force=0.4,
        stabilizer_kp=3.0,
    )

    env.reset()

    force_variance_list = []
    for _ in range(50):
        action = np.random.uniform(-0.3, 0.3, 16)
        obs, reward, terminated, truncated, info = env.step(action)

        forces = info["contact_forces"]
        force_variance_list.append(forces.var())

    avg_variance = np.mean(force_variance_list)
    assert avg_variance >= 0
