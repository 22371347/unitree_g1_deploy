from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

import whole_body_tracking.tasks.tracking.mdp as mdp


class G1PunchGetUpEnv(ManagerBasedRLEnv):
    """Single-policy WBT punch + self-get-up environment.

    This class changes RESET ORDER only.

    IsaacLab's normal ManagerBasedRLEnv reset executes the action/reward
    managers and then command_manager.reset().  WBT's MotionCommand writes the
    reference root/joint state during command reset.  Therefore the fallen pose
    must be written AFTER super()._reset_idx() has completely finished.

    This is training initialization only:
        one 154-D policy
        no policy switch
        no deployment state machine
    """

    def _reset_idx(
        self,
        env_ids: Sequence[int],
    ):
        # Run the original complete IsaacLab + WBT reset first.
        super()._reset_idx(env_ids)

        # Then overwrite only a subset with fallen starts.
        fallen_ids = (
            mdp.reset_mixed_tracking_and_fallen_after_command(
                self,
                env_ids,
                fallen_probability=(
                    self.cfg.fallen_reset_probability
                ),
                root_height_range=(
                    self.cfg.fallen_root_height_range
                ),
                xy_range=(
                    self.cfg.fallen_xy_range
                ),
                joint_noise_range=(
                    self.cfg.fallen_joint_noise_range
                ),
                orientation_noise=(
                    self.cfg.fallen_orientation_noise
                ),
            )
        )

        # super()._reset_idx() already resets ActionManager, but reset it once
        # more for the environments whose physical pose was overwritten after
        # the normal reset. This guarantees action and prev_action histories
        # begin cleanly for get-up samples.
        if fallen_ids.numel() > 0:
            self.action_manager.reset(
                fallen_ids
            )
