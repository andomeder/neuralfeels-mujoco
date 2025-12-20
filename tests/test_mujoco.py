import mujoco
import numpy as np


def test_scene_loads():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    data = mujoco.MjData(model)

    assert model.nbody == 24
    assert model.njnt == 17
    assert model.nu == 16
    assert model.ncam == 3


def test_allegro_hand_loads():
    model = mujoco.MjModel.from_xml_path("envs/assets/allegro/allegro_hand_right.xml")
    data = mujoco.MjData(model)

    assert model.njnt == 16
    assert model.nu == 16


def test_tactile_geoms_exist():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")

    tactile_names = [
        "fingertip_0_tactile",
        "fingertip_1_tactile",
        "fingertip_2_tactile",
        "fingertip_3_tactile",
    ]

    for name in tactile_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0, f"Geom {name} not found"


def test_simulation_step():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    data = mujoco.MjData(model)

    initial_time = data.time
    mujoco.mj_step(model, data)

    assert data.time > initial_time


def test_rendering():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 224, 224)
    renderer.update_scene(data, camera="main_cam")
    img = renderer.render()

    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8

    renderer.close()


if __name__ == "__main__":
    test_scene_loads()
    print("✓ test_scene_loads passed")

    test_allegro_hand_loads()
    print("✓ test_allegro_hand_loads passed")

    test_tactile_geoms_exist()
    print("✓ test_tactile_geoms_exist passed")

    test_simulation_step()
    print("✓ test_simulation_step passed")

    test_rendering()
    print("✓ test_rendering passed")

    print("\nAll MuJoCo tests passed!")
