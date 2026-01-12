from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.grasp_controller import GraspStabilizer


class AllegroHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    NUM_JOINTS = 16
    NUM_FINGERS = 4
    TACTILE_RES = 32
    IMG_SIZE = 224

    TACTILE_GEOM_NAMES = [
        "fingertip_0_tactile",
        "fingertip_1_tactile",
        "fingertip_2_tactile",
        "fingertip_3_tactile",
    ]

    # Available object types (primitives mimicking YCB shapes)
    OBJECT_SET = [
        "sphere",  # Tennis ball-like
        "box",  # Cracker box-like
        "cylinder",  # Soup can-like
        "capsule",  # Banana-like
        "ellipsoid",  # Apple-like
        "mug",  # Mug with handle
        "hammer",  # Tool-like
    ]

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        control_freq: int = 20,
        sim_freq: int = 500,
        max_episode_steps: int = 200,
        use_grasp_stabilizer: bool = False,
        stabilizer_kp: float = 2.0,
        stabilizer_kd: float = 0.1,
        target_force: float = 0.3,
        object_name: str = "sphere",
        randomize_object: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.frame_skip = sim_freq // control_freq
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        if object_name not in self.OBJECT_SET:
            raise ValueError(f"Unknown object '{object_name}'. Choose from {self.OBJECT_SET}")
        self.object_name = object_name
        self.randomize_object = randomize_object
        self._current_object = object_name

        self.use_grasp_stabilizer = use_grasp_stabilizer
        self._grasp_stabilizer = None
        if use_grasp_stabilizer:
            self._grasp_stabilizer = GraspStabilizer(
                num_fingers=self.NUM_FINGERS,
                target_force=target_force,
                kp=stabilizer_kp,
                kd=stabilizer_kd,
            )

        self.assets_path = Path(__file__).parent / "assets"
        self.model, self.data = self._load_scene_with_object(self._current_object)

        self.model.opt.timestep = 1.0 / sim_freq

        self._tactile_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.TACTILE_GEOM_NAMES
        ]

        self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=(self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8
                ),
                "tactile": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.NUM_FINGERS, self.TACTILE_RES, self.TACTILE_RES),
                    dtype=np.float32,
                ),
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.NUM_JOINTS,), dtype=np.float32
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.NUM_JOINTS,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_JOINTS,), dtype=np.float32
        )

        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, self.IMG_SIZE, self.IMG_SIZE)

    def _load_scene_with_object(self, object_name: str) -> tuple:
        import xml.etree.ElementTree as ET

        scene_path = self.assets_path / "scene.xml"
        object_path = self.assets_path / "objects" / f"{object_name}.xml"

        scene_tree = ET.parse(scene_path)
        scene_root = scene_tree.getroot()

        object_tree = ET.parse(object_path)
        object_worldbody = object_tree.find("worldbody")

        scene_worldbody = scene_root.find("worldbody")

        existing_object = scene_worldbody.find("./body[@name='object']")
        if existing_object is not None:
            scene_worldbody.remove(existing_object)

        for object_body in object_worldbody.findall("body"):
            scene_worldbody.append(object_body)

        tmp_scene_path = self.assets_path / f"_tmp_scene_{object_name}.xml"
        scene_tree.write(str(tmp_scene_path), encoding="unicode")

        model = mujoco.MjModel.from_xml_path(str(tmp_scene_path))
        data = mujoco.MjData(model)

        tmp_scene_path.unlink()

        return model, data

    def _get_obs(self) -> dict[str, np.ndarray]:
        qpos = self.data.qpos[: self.NUM_JOINTS].astype(np.float32)
        qvel = self.data.qvel[: self.NUM_JOINTS].astype(np.float32)

        tactile = self._get_tactile_obs()

        if self._renderer is not None:
            self._renderer.update_scene(self.data, camera="main_cam")
            rgb = self._renderer.render()
        else:
            rgb = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)

        return {
            "rgb": rgb,
            "tactile": tactile,
            "qpos": qpos,
            "qvel": qvel,
        }

    def _get_tactile_obs(self) -> np.ndarray:
        tactile = np.zeros((self.NUM_FINGERS, self.TACTILE_RES, self.TACTILE_RES), dtype=np.float32)

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1, geom2 = contact.geom1, contact.geom2

            for finger_idx, tactile_geom_id in enumerate(self._tactile_geom_ids):
                if geom1 == tactile_geom_id or geom2 == tactile_geom_id:
                    contact_pos = contact.pos
                    fingertip_body_name = f"fingertip_{finger_idx}"
                    fingertip_body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, fingertip_body_name
                    )

                    fingertip_pos = self.data.xpos[fingertip_body_id]
                    fingertip_mat = self.data.xmat[fingertip_body_id].reshape(3, 3)

                    local_pos = fingertip_mat.T @ (contact_pos - fingertip_pos)

                    u = int((local_pos[0] / 0.02 + 0.5) * self.TACTILE_RES)
                    v = int((local_pos[1] / 0.02 + 0.5) * self.TACTILE_RES)

                    u = np.clip(u, 0, self.TACTILE_RES - 1)
                    v = np.clip(v, 0, self.TACTILE_RES - 1)

                    # Get contact force from MuJoCo contact solver
                    # contact.frame is the contact frame basis, not force
                    # Use mj_contactForce to get actual contact forces
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_idx, force)
                    force_magnitude = np.linalg.norm(force[:3])  # Normal + tangent forces
                    depth = min(force_magnitude / 10.0, 1.0)

                    tactile[finger_idx, v, u] = max(tactile[finger_idx, v, u], depth)

        return tactile

    def _get_info(self) -> dict[str, Any]:
        object_pos = self.data.xpos[self._object_body_id].copy()
        object_quat = self.data.xquat[self._object_body_id].copy()

        info = {
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_name": self._current_object,
            "step_count": self._step_count,
        }

        if self._grasp_stabilizer is not None:
            stabilizer_state = self._grasp_stabilizer.get_state()
            info.update(
                {
                    "contact_forces": stabilizer_state["forces"],
                    "drop_detected": stabilizer_state["drop_detected"],
                    "no_contact_frames": stabilizer_state["no_contact_frames"],
                }
            )

        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        if self.randomize_object:
            self._current_object = self.np_random.choice(self.OBJECT_SET)
            self.model, self.data = self._load_scene_with_object(self._current_object)
            self.model.opt.timestep = 1.0 / self.sim_freq
            self._tactile_geom_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in self.TACTILE_GEOM_NAMES
            ]
            self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = mujoco.Renderer(self.model, self.IMG_SIZE, self.IMG_SIZE)

        mujoco.mj_resetData(self.model, self.data)

        init_qpos = np.array(
            [
                0,
                0.5,
                0.5,
                0.5,
                0,
                0.5,
                0.5,
                0.5,
                0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )
        self.data.qpos[: self.NUM_JOINTS] = init_qpos

        object_qpos_start = self.NUM_JOINTS
        object_pos = np.array([0, 0, 0.1])
        object_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
        self.data.qpos[object_qpos_start : object_qpos_start + 3] = object_pos

        random_quat = self._random_quaternion()
        self.data.qpos[object_qpos_start + 3 : object_qpos_start + 7] = random_quat

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        if self._grasp_stabilizer is not None:
            self._grasp_stabilizer.reset()

        return self._get_obs(), self._get_info()

    def _random_quaternion(self) -> np.ndarray:
        u1, u2, u3 = self.np_random.uniform(0, 1, size=3)
        sqrt1_u1 = np.sqrt(1 - u1)
        sqrtu1 = np.sqrt(u1)
        return np.array(
            [
                sqrt1_u1 * np.sin(2 * np.pi * u2),
                sqrt1_u1 * np.cos(2 * np.pi * u2),
                sqrtu1 * np.sin(2 * np.pi * u3),
                sqrtu1 * np.cos(2 * np.pi * u3),
            ]
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        if self.use_grasp_stabilizer and self._grasp_stabilizer is not None:
            tactile = self._get_tactile_obs()
            grip_adjustment = self._grasp_stabilizer.compute_grip_adjustment(tactile)
            action = action + grip_adjustment
            action = np.clip(action, -1.0, 1.0)

        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        object_pos = self.data.xpos[self._object_body_id]
        palm_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")]

        dist = np.linalg.norm(object_pos - palm_pos)
        reward = -dist

        if object_pos[2] < 0:
            reward -= 10.0

        return float(reward)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data, camera="main_cam")
            return self._renderer.render()
        return None

    def get_contact_forces(self) -> np.ndarray:
        """Get per-finger contact forces from tactile sensors."""
        if self._grasp_stabilizer is None:
            tactile = self._get_tactile_obs()
            return np.array([tactile[i].sum() for i in range(self.NUM_FINGERS)])
        return self._grasp_stabilizer.get_state()["forces"]

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
