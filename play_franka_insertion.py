"""Scripted Franka peg insertion using mjlab torque actions and bias-compensated PD.

Run: uv run python play_franka_insertion.py
Optional: uv run python play_franka_insertion.py --no-correction
"""

import argparse
import mujoco
import numpy as np
import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.rl import RslRlVecEnvWrapper
from anymal_c_velocity.franka_emika_panda.franka_constants import get_franka_robot_cfg
from anymal_c_velocity.franka_insert_env_cfgs import get_fixture_spec


def solve_ik(model, data, site_id, target_position, target_rotation,

             q_initial, qpos_ids, dof_ids, max_iterations=200):

    """Damped least-squares 6D IK, adapted from test_franka_insertion.py."""

    q = q_initial.copy()

    for _ in range(max_iterations):

        data.qpos[qpos_ids] = q

        mujoco.mj_forward(model, data)

        position_error = target_position - data.site_xpos[site_id]

        current_rotation = data.site_xmat[site_id].reshape(3, 3)

        orientation_error = 0.5 * sum(

            (np.cross(current_rotation[:, i], target_rotation[:, i])

             for i in range(3)), np.zeros(3)

        )

        if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(orientation_error) < 1e-3:

            return q.copy()

        jac_pos = np.zeros((3, model.nv))

        jac_rot = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)

        jac = np.vstack((jac_pos[:, dof_ids], jac_rot[:, dof_ids]))

        error = np.r_[position_error, orientation_error]

        q += 0.2 * jac.T @ np.linalg.solve(jac @ jac.T + 0.05**2 * np.eye(6), error)

    raise RuntimeError(f"IK did not converge for target {target_position}")





def build_trajectory():

    """Use a scratch MuJoCo model for IK; control is executed in mjlab."""

    robot_cfg = get_franka_robot_cfg()

    spec = Entity(robot_cfg).spec

    frame = spec.worldbody.add_frame(name="fixture_attachment")

    spec.attach(get_fixture_spec(), frame=frame, prefix="", suffix="")

    model = spec.compile()

    data = mujoco.MjData(model)

    home = robot_cfg.init_state.joint_pos

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]

    if any(j < 0 for j in joint_ids):

        raise RuntimeError("Franka joints joint1..joint7 were not found")

    qpos_ids = [int(model.jnt_qposadr[j]) for j in joint_ids]

    dof_ids = [int(model.jnt_dofadr[j]) for j in joint_ids]

    for name, position in home.items():

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if jid >= 0:

            data.qpos[model.jnt_qposadr[jid]] = position

    mujoco.mj_forward(model, data)

    home_q = data.qpos[qpos_ids].copy()

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")

    if tip_id < 0:

        raise RuntimeError("peg_tip site not found")

    rotation = data.site_xmat[tip_id].reshape(3, 3).copy()

    nominal = np.array([0.5545, 0.0, 0.05])

    corrected = np.array([0.5595, 0.0, 0.05])

    inserted = np.array([0.5595, 0.0, 0.01])

    approach_q = solve_ik(model, data, tip_id, nominal, rotation, home_q, qpos_ids, dof_ids)

    corrected_q = solve_ik(model, data, tip_id, corrected, rotation, approach_q, qpos_ids, dof_ids)

    waypoint_q = []

    seed = corrected_q.copy()

    for position in np.linspace(corrected, inserted, 81):

        seed = solve_ik(model, data, tip_id, position, rotation, seed, qpos_ids, dof_ids)

        waypoint_q.append(seed.copy())

    nominal_waypoints = []

    seed = approach_q.copy()

    for position in np.linspace(nominal, np.array([0.5545, 0.0, 0.01]), 81):

        seed = solve_ik(model, data, tip_id, position, rotation, seed, qpos_ids, dof_ids)

        nominal_waypoints.append(seed.copy())

    return model, home_q, approach_q, corrected_q, np.asarray(waypoint_q), np.asarray(nominal_waypoints)







class BiasCompensatedPD:
    """Joint-space PD plus MuJoCo's live bias forces (gravity + velocity terms)."""

    def __init__(self, model: mujoco.MjModel, device: torch.device):
        self.device = device
        self.kp = torch.tensor(
            [100, 100, 100, 100, 50, 50, 50],
            device=device,
            dtype=torch.float32,
        )

        self.kd = torch.tensor(
            [20, 20, 20, 20, 10, 10, 10],
            device=device,
            dtype=torch.float32,
        )
        # self.kp = torch.tensor([400, 400, 400, 400, 200, 200, 200], device=device, dtype=torch.float32)
        # self.kd = torch.tensor([40, 40, 40, 40, 20, 20, 20], device=device, dtype=torch.float32)
        self.limits = torch.tensor([87, 87, 87, 87, 12, 12, 12], device=device, dtype=torch.float32)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        if any(j < 0 for j in joint_ids):
            raise RuntimeError("Arm joints are missing from the reference model")
        self.dof_ids = [int(model.jnt_dofadr[j]) for j in joint_ids]
        self.qpos_ids = [int(model.jnt_qposadr[j]) for j in joint_ids]

    def compute(self, desired: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                live_qfrc_bias: torch.Tensor) -> torch.Tensor:
        # qfrc_bias is from the LIVE mjlab simulation. Do not call inverse() here.
        bias = live_qfrc_bias[:, self.dof_ids]
        torque = self.kp * (desired - q) - self.kd * qd + bias
        return torch.maximum(torch.minimum(torque, self.limits), -self.limits)


class InsertionPolicy:
    def __init__(self, env, trajectory, controller, correction=True):
        self.env = env
        self.controller = controller
        self.home, self.approach, self.corrected, self.waypoints, self.nominal_waypoints = trajectory
        self.correction = correction
        self.calls = 0
        self.step_dt = 0.005  # Match the existing insertion trajectory/control cadence.
        self.action_shape = env.unwrapped.action_space.shape
        if self.action_shape[-1] != 8:
            raise ValueError(f"Expected 8 actions (7 torques + gripper), got {self.action_shape}")

        term = env.unwrapped.action_manager.get_term("arm_effort")
        names = list(term.target_names)
        expected = [f"joint{i}" for i in range(1, 8)]
        if names != expected:
            raise RuntimeError(f"Arm action order {names} does not match {expected}")
        print("Arm effort action order:", names)
        print("Control step:", self.step_dt)

    def target_at(self, t):
        if t < 4:
            alpha = np.clip(t / 4, 0, 1)
            return (1 - alpha) * self.home + alpha * self.approach
        if t < 8:
            return self.approach
        if t < 10:
            alpha = np.clip((t - 8) / 2, 0, 1)
            return ((1 - alpha) * self.approach + alpha * self.corrected) if self.correction else self.approach
        if t < 18:
            alpha = np.clip((t - 10) / 8, 0, 1)
            index = min(int(alpha * 80), 80)
            return self.waypoints[index] if self.correction else self.nominal_waypoints[index]
        return self.waypoints[-1] if self.correction else self.nominal_waypoints[-1]

    def __call__(self, obs):
        del obs
        t = self.calls * self.step_dt
        self.calls += 1
        target_np = self.target_at(t)
        #target_np = self.home.copy()
        robot = self.env.unwrapped.scene["robot"]
        sim = self.env.unwrapped.sim
        device = self.controller.device

        q = robot.data.joint_pos[:, :7]
        qd = robot.data.joint_vel[:, :7]
        desired = torch.as_tensor(target_np, device=device, dtype=q.dtype).unsqueeze(0)

        # Validate the reference-model DOF/QPOS indexing before using bias torques.
        if self.calls == 1:
            sim_qpos = sim.data.qpos[:, self.controller.qpos_ids]
            if not torch.allclose(sim_qpos, q, atol=1e-3, rtol=1e-3):
                raise RuntimeError(
                    "Reference model qpos indexing does not match live mjlab scene; "
                    "do not apply qfrc_bias with these DOF IDs."
                )

        effort = self.controller.compute(desired, q, qd, sim.data.qfrc_bias)
        action = torch.zeros(self.action_shape, device=device, dtype=torch.float32)
        action[:, :7] = effort
        action[:, 7] = 0.0  # Gripper: default finger position (0.04 m).

        if self.calls % 50 == 0 or self.calls == 1:

            bias = sim.data.qfrc_bias[:, self.controller.dof_ids]

            position_term = self.controller.kp * (desired - q)
            damping_term = -self.controller.kd * qd

            print(f"\nTime: {t:.2f}s")

            print("Desired:", desired[0].detach().cpu().numpy().round(3))
            print("Actual: ", q[0].detach().cpu().numpy().round(3))
            print("Error:  ", (desired - q)[0].detach().cpu().numpy().round(3))

            print("\n--- CONTROLLER DEBUG ---")

            print("Joint velocities:",
                qd[0].detach().cpu().numpy().round(3))

            print("P torque:",
                position_term[0].detach().cpu().numpy().round(2))

            print("D torque:",
                damping_term[0].detach().cpu().numpy().round(2))

            print("Bias torque:",
                bias[0].detach().cpu().numpy().round(2))

            print("Final torque:",
                effort[0].detach().cpu().numpy().round(2))

            print("------------------------")
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-correction", action="store_true")
    args = parser.parse_args()

    reference_model, *trajectory = build_trajectory()
    home, approach, corrected, waypoints, _ = trajectory
    print("\n========== IK TRAJECTORY ==========")
    print("HOME:", home)
    print("APPROACH:", approach)
    print("CORRECTED:", corrected)
    print("FINAL INSERTION:", waypoints[-1])
    print("===================================\n")

    cfg = load_env_cfg("Mjlab-Franka-Peg-Insertion", play=True)
    cfg.scene.num_envs = 1
    cfg.terminations = {}
    base_env = ManagerBasedRlEnv(
        cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print("Gravity:", base_env.sim.model.opt.gravity)
    # The RSL-RL wrapper otherwise clips torque actions to the RL default (often [-1,1]).
    # 100 exceeds every configured arm torque limit (87 Nm).
    env = RslRlVecEnvWrapper(base_env, clip_actions=100.0)

    try:
        controller = BiasCompensatedPD(reference_model, device=torch.device(base_env.device))
        policy = InsertionPolicy(env, trajectory, controller, correction=not args.no_correction)
        NativeMujocoViewer(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
