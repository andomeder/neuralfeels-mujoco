from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


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

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        control_freq: int = 20,
        sim_freq: int = 500,
        max_episode_steps: int = 200,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.frame_skip = sim_freq // control_freq
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        assets_path = Path(__file__).parent / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(assets_path))
        self.data = mujoco.MjData(self.model)

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

                    force_magnitude = np.linalg.norm(contact.frame[:3])
                    depth = min(force_magnitude / 10.0, 1.0)

                    tactile[finger_idx, v, u] = max(tactile[finger_idx, v, u], depth)

        return tactile

    def _get_info(self) -> dict[str, Any]:
        object_pos = self.data.xpos[self._object_body_id].copy()
        object_quat = self.data.xquat[self._object_body_id].copy()

        return {
            "object_pos": object_pos,
            "object_quat": object_quat,
            "step_count": self._step_count,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

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
        self.data.qpos[object_qpos_start : object_qpos_start + 3] = [0, 0, 0.1]
        self.data.qpos[object_qpos_start + 3 : object_qpos_start + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
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

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
