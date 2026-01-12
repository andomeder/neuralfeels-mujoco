import mujoco
import numpy as np


def test_scene_loads():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    mujoco.MjData(model)

    assert model.nbody == 24
    assert model.njnt == 17
    assert model.nu == 16
    assert model.ncam == 3


def test_allegro_hand_loads():
    model = mujoco.MjModel.from_xml_path("envs/assets/allegro/allegro_hand_right.xml")
    mujoco.MjData(model)

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

    for _ in range(10):
        mujoco.mj_step(model, data)

    assert data.time > 0


def test_rendering():
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 224, 224)
    renderer.update_scene(data, camera="main_cam")
    img = renderer.render()

    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8

    renderer.close()
