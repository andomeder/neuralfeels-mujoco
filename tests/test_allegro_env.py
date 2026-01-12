def test_env_creation():
    from envs.allegro_hand_env import AllegroHandEnv

    env = AllegroHandEnv()
    assert env is not None
    assert env.NUM_JOINTS == 16
    assert env.NUM_FINGERS == 4
    env.close()


def test_env_reset():
    from envs.allegro_hand_env import AllegroHandEnv

    env = AllegroHandEnv()
    obs, info = env.reset()

    assert obs["rgb"].shape == (224, 224, 3)
    assert obs["tactile"].shape == (4, 32, 32)
    assert obs["qpos"].shape == (16,)
    assert obs["qvel"].shape == (16,)
    assert "object_pos" in info
    assert "object_quat" in info

    env.close()


def test_env_step():
    from envs.allegro_hand_env import AllegroHandEnv

    env = AllegroHandEnv()
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_count"] == 1

    env.close()


def test_env_multiple_steps():
    from envs.allegro_hand_env import AllegroHandEnv

    env = AllegroHandEnv()
    env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    assert info["step_count"] == 10
    env.close()


def test_env_render():
    from envs.allegro_hand_env import AllegroHandEnv

    env = AllegroHandEnv(render_mode="rgb_array")
    env.reset()

    img = env.render()
    assert img is not None
    assert img.shape == (224, 224, 3)

    env.close()


if __name__ == "__main__":
    test_env_creation()
    print("✓ test_env_creation passed")

    test_env_reset()
    print("✓ test_env_reset passed")

    test_env_step()
    print("✓ test_env_step passed")

    test_env_multiple_steps()
    print("✓ test_env_multiple_steps passed")

    test_env_render()
    print("✓ test_env_render passed")

    print("\nAll tests passed!")
