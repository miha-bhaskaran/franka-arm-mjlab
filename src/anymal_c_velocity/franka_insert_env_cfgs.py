"""Franka Panda peg-in-hole insertion environment with arm effort control."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp

from anymal_c_velocity.franka_emika_panda.franka_constants import get_franka_robot_cfg


def get_fixture_spec() -> mujoco.MjSpec:
    """Static square fixture with a 22-mm square through-hole."""
    spec = mujoco.MjSpec()
    outer_width = 0.10
    hole_width = 0.022
    thickness = 0.03
    wall_width = (outer_width - hole_width) / 2

    fixture = spec.worldbody.add_body(
        name="insertion_fixture", pos=(0.5595, 0.0, thickness / 2)
    )

    for name, x in (
        ("left_wall", -(hole_width + wall_width) / 2),
        ("right_wall", (hole_width + wall_width) / 2),
    ):
        fixture.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0.0, 0.0),
            size=(wall_width / 2, outer_width / 2, thickness / 2),
            rgba=(0.6, 0.6, 0.6, 1.0),
        )

    for name, y in (
        ("front_wall", -(hole_width + wall_width) / 2),
        ("back_wall", (hole_width + wall_width) / 2),
    ):
        fixture.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(0.0, y, 0.0),
            size=(hole_width / 2, wall_width / 2, thickness / 2),
            rgba=(0.6, 0.6, 0.6, 1.0),
        )
    return spec


def franka_insertion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_lift_cube_env_cfg()
    cfg.scene.entities = {
        "robot": get_franka_robot_cfg(),
        "fixture": EntityCfg(spec_fn=get_fixture_spec),
    }

    # Ordered first seven torque actions, followed by one gripper position action.
    cfg.actions = {
        "arm_effort": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=tuple(f"joint{i}" for i in range(1, 8)),
            scale=1.0,
            preserve_order=True,
        ),
        "gripper_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("finger_joint1",),
            scale=0.04,
            use_default_offset=True,
            preserve_order=True,
        ),
    }

    for group in cfg.observations.values():
        group.terms = {
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
        }

    cfg.sim.dt = 0.005
    cfg.decimation = 1
    cfg.commands = {}
    cfg.events = {
        "reset_base": cfg.events["reset_base"],
        "reset_robot_joints": cfg.events["reset_robot_joints"],
    }
    cfg.rewards = {
        "action_rate_l2": cfg.rewards["action_rate_l2"],
        "joint_pos_limits": cfg.rewards["joint_pos_limits"],
    }
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    cfg.scene.sensors = ()
    cfg.curriculum = {}
    cfg.viewer.body_name = "hand"

    if play:
        cfg.episode_length_s = int(1e9)
        for group in cfg.observations.values():
            group.enable_corruption = False
    return cfg


if __name__ == "__main__":
    import mujoco.viewer

    spec = get_fixture_spec()
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(1.0, 1.0, 0.1),
        rgba=(0.9, 0.9, 0.9, 1.0),
    )
    mujoco.viewer.launch(spec.compile())
