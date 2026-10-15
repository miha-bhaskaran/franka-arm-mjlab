"""Franka Emika Panda robot and torque-actuator configuration."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

_HERE = Path(__file__).parent
FRANKA_XML = _HERE / "panda.xml"
if not FRANKA_XML.exists():
    raise FileNotFoundError(FRANKA_XML)


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, FRANKA_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(FRANKA_XML))
    spec.assets = get_assets(spec.meshdir)

    for name in (f"actuator{i}" for i in range(1, 9)):
        spec.delete(spec.actuator(name))
    spec.delete(spec.key("home"))

    hand = spec.body("hand")
    hand.add_site(
        name="grasp_site", pos=(0.0, 0.0, 0.1034),
        size=(0.01,), rgba=(1.0, 0.0, 0.0, 1.0),
    )
    peg = hand.add_body(name="insertion_peg", pos=(0.0, 0.0, 0.145))
    peg.add_geom(
        name="peg_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.01, 0.01, 0.04), mass=0.05,
        rgba=(0.2, 0.6, 0.9, 1.0),
    )
    peg.add_site(
        name="peg_tip", pos=(0.0, 0.0, 0.04),
        size=(0.005,), rgba=(0.0, 1.0, 0.0, 1.0),
    )
    return spec


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -1.57079,
        "joint5": 0.0,
        "joint6": 1.57079,
        "joint7": -0.7853,
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    },
    joint_vel={".*": 0.0},
)

FRANKA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinMotorActuatorCfg(
            target_names_expr=("joint1", "joint2", "joint3", "joint4"),
            effort_limit=87.0,
            armature=0.1,
        ),
        BuiltinMotorActuatorCfg(
            target_names_expr=("joint5", "joint6", "joint7"),
            effort_limit=12.0,
            armature=0.1,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("finger_joint1",),
            stiffness=100.0,
            damping=10.0,
            effort_limit=20.0,
            armature=0.001,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)

# Kept for compatibility with other task files that may still import it.
# It is NOT used by the insertion task's torque-controlled arm.
FRANKA_ACTION_SCALE = {f"joint{i}": 1.0 for i in range(1, 8)}
FRANKA_ACTION_SCALE["finger_joint1"] = 0.04


def get_franka_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=FRANKA_ARTICULATION,
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity
    import mujoco.viewer

    model = Entity(get_franka_robot_cfg()).spec.compile()
    print("Number of actuators:", model.nu)
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    mujoco.viewer.launch(model)
