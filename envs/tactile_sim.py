import numpy as np
from scipy.ndimage import gaussian_filter

TACTILE_RES = 32
SENSOR_SIZE = 0.02
MAX_FORCE = 10.0
GAUSSIAN_SIGMA = 1.5


def get_tactile_depth(
    contact_positions: list[np.ndarray],
    contact_forces: list[float],
    fingertip_pos: np.ndarray,
    fingertip_mat: np.ndarray,
    resolution: int = TACTILE_RES,
    sensor_size: float = SENSOR_SIZE,
    max_force: float = MAX_FORCE,
    apply_blur: bool = True,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    depth_map = np.zeros((resolution, resolution), dtype=np.float32)

    if not contact_positions:
        return depth_map

    for pos, force in zip(contact_positions, contact_forces):
        local_pos = fingertip_mat.T @ (pos - fingertip_pos)

        u = (local_pos[0] / sensor_size + 0.5) * resolution
        v = (local_pos[1] / sensor_size + 0.5) * resolution

        u_int = int(np.clip(u, 0, resolution - 1))
        v_int = int(np.clip(v, 0, resolution - 1))

        depth = min(force / max_force, 1.0)

        depth_map[v_int, u_int] = max(depth_map[v_int, u_int], depth)

    if apply_blur and depth_map.sum() > 0:
        depth_map = gaussian_filter(depth_map, sigma=sigma)
        if depth_map.max() > 0:
            depth_map = depth_map / depth_map.max()

    return depth_map


def splat_contact(
    depth_map: np.ndarray,
    u: float,
    v: float,
    depth: float,
    splat_radius: int = 2,
) -> None:
    resolution = depth_map.shape[0]
    u_int, v_int = int(u), int(v)

    for du in range(-splat_radius, splat_radius + 1):
        for dv in range(-splat_radius, splat_radius + 1):
            nu, nv = u_int + du, v_int + dv
            if 0 <= nu < resolution and 0 <= nv < resolution:
                dist = np.sqrt(du**2 + dv**2)
                weight = max(0, 1 - dist / (splat_radius + 1))
                depth_map[nv, nu] = max(depth_map[nv, nu], depth * weight)


def get_tactile_from_mujoco(
    data,
    model,
    tactile_geom_ids: list[int],
    fingertip_body_ids: list[int],
    resolution: int = TACTILE_RES,
    apply_blur: bool = True,
) -> np.ndarray:
    num_fingers = len(tactile_geom_ids)
    tactile = np.zeros((num_fingers, resolution, resolution), dtype=np.float32)

    finger_contacts: list[tuple[list[np.ndarray], list[float]]] = [
        ([], []) for _ in range(num_fingers)
    ]

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1, geom2 = contact.geom1, contact.geom2

        for finger_idx, tactile_geom_id in enumerate(tactile_geom_ids):
            if geom1 == tactile_geom_id or geom2 == tactile_geom_id:
                contact_pos = contact.pos.copy()
                # Get contact force from MuJoCo contact solver
                # contact.frame is the contact frame basis, not force
                import mujoco

                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_idx, force)
                force_magnitude = np.linalg.norm(force[:3])  # Normal + tangent forces

                finger_contacts[finger_idx][0].append(contact_pos)
                finger_contacts[finger_idx][1].append(force_magnitude)

    for finger_idx in range(num_fingers):
        positions, forces = finger_contacts[finger_idx]
        if positions:
            body_id = fingertip_body_ids[finger_idx]
            fingertip_pos = data.xpos[body_id]
            fingertip_mat = data.xmat[body_id].reshape(3, 3)

            tactile[finger_idx] = get_tactile_depth(
                positions,
                forces,
                fingertip_pos,
                fingertip_mat,
                resolution=resolution,
                apply_blur=apply_blur,
            )

    return tactile


def visualize_tactile(
    tactile: np.ndarray,
    finger_names: list[str] | None = None,
    contact_forces: np.ndarray | None = None,
) -> np.ndarray:
    import cv2

    num_fingers = tactile.shape[0]
    resolution = tactile.shape[1]

    if finger_names is None:
        finger_names = [f"F{i}" for i in range(num_fingers)]

    cell_size = resolution * 3
    grid = np.zeros((cell_size, cell_size * num_fingers, 3), dtype=np.uint8)

    for i in range(num_fingers):
        depth = tactile[i]
        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colored, (cell_size, cell_size))

        has_contact = depth.max() > 0.05
        if has_contact:
            cv2.rectangle(depth_resized, (0, 0), (cell_size - 1, cell_size - 1), (0, 255, 255), 3)

        cv2.putText(
            depth_resized,
            finger_names[i],
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if contact_forces is not None and i < len(contact_forces):
            force = min(1.0, contact_forces[i])
            bar_height = int(force * (cell_size - 10))
            bar_x = cell_size - 12
            cv2.rectangle(
                depth_resized,
                (bar_x, cell_size - 5),
                (bar_x + 8, cell_size - 5 - bar_height),
                (0, 255, 0),
                -1,
            )
            cv2.rectangle(depth_resized, (bar_x, 5), (bar_x + 8, cell_size - 5), (100, 100, 100), 1)

        grid[:, i * cell_size : (i + 1) * cell_size] = depth_resized

    return grid
