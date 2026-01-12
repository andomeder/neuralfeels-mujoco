"""
Closed-loop grasp stabilizer using tactile feedback.

Maintains stable grasps by monitoring contact forces from tactile sensors
and adjusting grip strength to prevent object dropping.
"""

import numpy as np


class GraspStabilizer:
    """
    PD controller for grasp stabilization using tactile feedback.

    Monitors contact forces on each fingertip and adjusts grip strength
    to maintain target force levels. Detects object drops and triggers
    recovery by increasing grip force.

    Args:
        num_fingers: Number of fingers (default: 4 for Allegro Hand)
        target_force: Target normalized force per finger (0-1 scale)
        kp: Proportional gain for PD controller
        kd: Derivative gain for PD controller
        force_threshold_low: Below this force, increase grip
        force_threshold_high: Above this force, decrease grip
        drop_threshold: Consecutive frames with zero contact to trigger drop detection
        max_adjustment: Maximum grip adjustment per step (safety limit)
    """

    def __init__(
        self,
        num_fingers: int = 4,
        target_force: float = 0.3,
        kp: float = 2.0,
        kd: float = 0.1,
        force_threshold_low: float = 0.1,
        force_threshold_high: float = 0.8,
        drop_threshold: int = 5,
        max_adjustment: float = 0.5,
    ):
        self.num_fingers = num_fingers
        self.target_force = target_force
        self.kp = kp
        self.kd = kd
        self.force_threshold_low = force_threshold_low
        self.force_threshold_high = force_threshold_high
        self.drop_threshold = drop_threshold
        self.max_adjustment = max_adjustment

        self.prev_forces = np.zeros(num_fingers, dtype=np.float32)
        self.prev_tactile = None
        self.no_contact_frames = 0
        self.drop_detected = False

        # Allegro Hand: 4 fingers × 4 joints (0=spread, 1-3=curl)
        self.curl_joint_indices = []
        for finger in range(num_fingers):
            base_idx = finger * 4
            self.curl_joint_indices.extend([base_idx + 1, base_idx + 2, base_idx + 3])

    def get_finger_forces(self, tactile: np.ndarray) -> np.ndarray:
        """
        Extract contact force proxy from tactile depth maps.

        Args:
            tactile: (num_fingers, H, W) tactile depth maps (0-1 normalized)

        Returns:
            (num_fingers,) array of force estimates
        """
        forces = np.array(
            [
                tactile[i].sum() / (tactile.shape[1] * tactile.shape[2])
                for i in range(self.num_fingers)
            ],
            dtype=np.float32,
        )

        forces = forces * 20.0  # Empirical scaling for force proxy

        return np.clip(forces, 0.0, 1.0)

    def detect_slip(self, tactile: np.ndarray) -> np.ndarray:
        """
        Detect slip by computing frame-to-frame tactile changes.

        Args:
            tactile: Current tactile observation

        Returns:
            (num_fingers,) boolean array indicating slip per finger
        """
        if self.prev_tactile is None:
            self.prev_tactile = tactile.copy()
            return np.zeros(self.num_fingers, dtype=bool)

        diff = np.abs(tactile - self.prev_tactile)

        slip_scores = np.array([diff[i].sum() for i in range(self.num_fingers)])
        slip_threshold = 0.5

        slipping = slip_scores > slip_threshold

        self.prev_tactile = tactile.copy()

        return slipping

    def compute_grip_adjustment(
        self,
        tactile: np.ndarray,
        enable_slip_detection: bool = True,
    ) -> np.ndarray:
        """
        Compute grip adjustment using PD control on tactile forces.

        Args:
            tactile: (num_fingers, H, W) tactile depth maps
            enable_slip_detection: Whether to use slip detection

        Returns:
            (16,) array of joint adjustments to add to action
        """
        forces = self.get_finger_forces(tactile)

        slip_detected = np.zeros(self.num_fingers, dtype=bool)
        if enable_slip_detection:
            slip_detected = self.detect_slip(tactile)

        if np.all(forces < 0.01):
            self.no_contact_frames += 1
        else:
            self.no_contact_frames = 0
            self.drop_detected = False

        if self.no_contact_frames >= self.drop_threshold:
            self.drop_detected = True

        errors = self.target_force - forces
        d_errors = forces - self.prev_forces

        adjustments = self.kp * errors - self.kd * d_errors

        adjustments = np.where(slip_detected, adjustments * 2.0, adjustments)

        if self.drop_detected:
            adjustments = np.full(self.num_fingers, 0.3)

        adjustments = np.clip(adjustments, -self.max_adjustment, self.max_adjustment)

        joint_adjustments = np.zeros(16, dtype=np.float32)

        for finger in range(self.num_fingers):
            curl_adjustment = adjustments[finger]
            base_idx = finger * 4
            for j in [1, 2, 3]:
                joint_adjustments[base_idx + j] = curl_adjustment

        self.prev_forces = forces.copy()

        return joint_adjustments

    def get_contact_forces(self, tactile: np.ndarray) -> np.ndarray:
        """
        Get current contact forces for monitoring/debugging.

        Args:
            tactile: (num_fingers, H, W) tactile depth maps

        Returns:
            (num_fingers,) array of force estimates
        """
        return self.get_finger_forces(tactile)

    def reset(self):
        """Reset internal state (call on environment reset)."""
        self.prev_forces = np.zeros(self.num_fingers, dtype=np.float32)
        self.prev_tactile = None
        self.no_contact_frames = 0
        self.drop_detected = False

    def is_drop_detected(self) -> bool:
        """Check if object drop has been detected."""
        return self.drop_detected

    def get_state(self) -> dict:
        """
        Get internal state for debugging/logging.

        Returns:
            Dictionary with current forces, drop status, etc.
        """
        return {
            "forces": self.prev_forces.copy(),
            "no_contact_frames": self.no_contact_frames,
            "drop_detected": self.drop_detected,
        }
