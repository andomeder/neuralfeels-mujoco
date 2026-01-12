import numpy as np
import pytest

from envs.allegro_hand_env import AllegroHandEnv


class TestObjectSupport:
    def test_all_objects_load(self):
        for obj_name in AllegroHandEnv.OBJECT_SET:
            env = AllegroHandEnv(object_name=obj_name)
            obs, info = env.reset()

            assert obs["rgb"].shape == (224, 224, 3)
            assert obs["tactile"].shape == (4, 32, 32)
            assert obs["qpos"].shape == (16,)
            assert obs["qvel"].shape == (16,)

            assert "object_pos" in info
            assert "object_quat" in info
            assert "object_name" in info
            assert info["object_name"] == obj_name

            env.close()

    def test_invalid_object_name(self):
        with pytest.raises(ValueError, match="Unknown object"):
            AllegroHandEnv(object_name="invalid_object")

    def test_object_randomization(self):
        env = AllegroHandEnv(object_name="sphere", randomize_object=True)

        objects_seen = set()
        for _ in range(20):
            obs, info = env.reset()
            objects_seen.add(info["object_name"])

        assert len(objects_seen) >= 3, f"Expected at least 3 different objects, got {objects_seen}"
        assert all(obj in AllegroHandEnv.OBJECT_SET for obj in objects_seen)

        env.close()

    def test_object_pose_randomization(self):
        env = AllegroHandEnv(object_name="box")

        positions = []
        quaternions = []
        for _ in range(10):
            obs, info = env.reset()
            positions.append(info["object_pos"])
            quaternions.append(info["object_quat"])

        positions = np.array(positions)
        quaternions = np.array(quaternions)

        pos_std = positions.std(axis=0)
        assert pos_std[0] > 0.001, "X position should vary"
        assert pos_std[1] > 0.001, "Y position should vary"

        quat_std = quaternions.std(axis=0)
        assert quat_std.sum() > 0.1, "Orientation should vary"

        env.close()

    def test_step_with_different_objects(self):
        for obj_name in ["sphere", "box", "cylinder"]:
            env = AllegroHandEnv(object_name=obj_name)
            obs, _ = env.reset()

            action = np.zeros(16)
            obs, reward, terminated, truncated, info = env.step(action)

            assert obs["rgb"].shape == (224, 224, 3)
            assert obs["tactile"].shape == (4, 32, 32)
            assert info["object_name"] == obj_name

            env.close()

    def test_object_geom_exists(self):
        for obj_name in AllegroHandEnv.OBJECT_SET:
            env = AllegroHandEnv(object_name=obj_name)
            env.reset()

            import mujoco

            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
            assert geom_id >= 0, f"Object geom not found for {obj_name}"

            env.close()
