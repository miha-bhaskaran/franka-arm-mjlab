# Franka Panda Peg-in-Hole Insertion in MJLab

A simulated peg-in-hole manipulation task using a Franka Emika Panda robot, implemented in MJLab with MuJoCo physics.

This project implements a scripted insertion pipeline combining damped least-squares inverse kinematics, joint-space trajectory generation, and torque-level PD control with bias-force compensation.

The goal is to establish a reliable nominal controller for investigating residual reinforcement learning on contact-rich manipulation tasks.

## Demo

![Franka Panda Insertion Demo](teaser.gif)

## Overview

Precise robotic insertion requires accurate end-effector positioning, stable joint control, and interaction with contact constraints.

This project implements a peg-in-hole environment in which a Franka Panda robot inserts a square peg into a fixed square opening.

The implementation includes:

- A custom Franka Panda model with a rigidly attached insertion peg.
- A static insertion fixture with a square opening.
- Six-dimensional damped least-squares inverse kinematics.
- Joint-space trajectories for approach, alignment, and insertion.
- A torque-controlled PD controller with MuJoCo bias-force compensation.
- A nominal-versus-corrected trajectory comparison.

The current implementation uses a scripted controller rather than a learned policy.

## Simulation Environment

### Robot

The environment uses a seven-degree-of-freedom Franka Emika Panda manipulator.

The robot configuration includes:

- Seven torque-controlled arm joints.
- A position-controlled gripper actuator.
- A rigidly attached insertion peg.
- A site at the peg tip for inverse kinematics.

The peg is modeled as a box with half-widths of 10 mm, giving it a square cross-section of 20 mm × 20 mm.

### Insertion Fixture

The fixture is constructed using four MuJoCo box geometries arranged around a square opening.

| Parameter | Value |
|-----------|-------|
| Fixture outer width | 100 mm |
| Hole width | 22 mm |
| Fixture thickness | 30 mm |
| Peg cross-section | 20 × 20 mm |
| Nominal lateral clearance | 1 mm per side |
| Simulation timestep | 0.005 s |
| Control frequency | 200 Hz |

The peg is rigidly attached to the robot's hand.

The fixture is positioned at approximately:

```python
[0.5595, 0.0, 0.015]
```

in world coordinates.

## Method

### 1. Inverse Kinematics

The robot's joint configurations are generated using a damped least-squares inverse kinematics solver.

The solver minimizes a six-dimensional end-effector pose error consisting of:

- Three-dimensional position error.
- Three-dimensional orientation error.

The iterative joint update is:

\[
q_{k+1}
=
q_k +
\alpha J^T
(JJ^T+\lambda^2 I)^{-1}e
\]

where:

- \(q\) represents the robot joint configuration.
- \(J\) is the 6D end-effector Jacobian.
- \(e\) is the position-and-orientation error.
- \(\alpha = 0.2\) is the update gain.
- \(\lambda = 0.05\) is the damping coefficient.

The solver terminates when both conditions are satisfied:

\[
\|e_p\| < 10^{-4}\ \text{m}
\]

\[
\|e_R\| < 10^{-3}
\]

The IK solver generates joint configurations for the approach, corrected alignment, and insertion waypoints.

### 2. Trajectory Generation

The scripted trajectory contains four main phases.

| Phase | Time | Description |
|-------|------|-------------|
| Approach | 0–4 s | Move from the home configuration to the nominal approach pose |
| Hold | 4–8 s | Maintain the approach configuration |
| Alignment | 8–10 s | Apply the lateral position correction |
| Insertion | 10–18 s | Follow the insertion trajectory into the fixture |

The nominal approach position is:

```python
[0.5545, 0.0, 0.05]
```

The corrected approach position is:

```python
[0.5595, 0.0, 0.05]
```

The final insertion target is:

```python
[0.5595, 0.0, 0.01]
```

The nominal approach is offset by 5 mm along the x-axis relative to the corrected target.

During the alignment phase, the robot transitions toward the corrected joint configuration.

The insertion trajectory is generated from 81 Cartesian waypoints, with inverse kinematics used to obtain the corresponding joint configurations.

### 3. Joint-Space PD Control

The robot tracks the desired joint configurations using a joint-space proportional-derivative controller.

The commanded torque is:

\[
\tau =
K_p(q_d-q)
-
K_d\dot{q}
+
\tau_{\mathrm{bias}}
\]

where:

- \(q_d\) is the desired joint configuration.
- \(q\) is the measured joint configuration.
- \(\dot{q}\) is the measured joint velocity.
- \(K_p\) is the proportional gain matrix.
- \(K_d\) is the derivative gain matrix.
- \(\tau_{\mathrm{bias}}\) represents MuJoCo's live bias forces.

The bias forces account for gravity and velocity-dependent effects.

The controller uses the following gains:

| Joints | Kp | Kd | Torque limit |
|--------|----|----|--------------|
| 1–4 | 100 | 20 | ±87 Nm |
| 5–7 | 50 | 10 | ±12 Nm |

The commanded torques are clipped to the configured actuator limits.

The controller operates at 200 Hz, matching the simulation timestep.

### 4. Nominal vs. Corrected Insertion

Two trajectory variants are provided.

**Corrected trajectory**

The robot approaches the nominal target, performs a 5 mm lateral correction, and then executes the insertion trajectory.

**Uncorrected trajectory**

The robot maintains the nominal lateral offset and follows an insertion trajectory without applying the alignment correction.

This provides a controlled comparison for studying the effect of target-position error on insertion behavior.

## Installation

### Requirements

- Python 3.10–3.13
- MJLab
- MuJoCo
- PyTorch
- NVIDIA Warp
- `uv` package manager

A compatible NVIDIA GPU and driver are recommended for GPU execution.

### Setup

Clone the repository:

```bash
git clone https://github.com/miha-bhaskaran/franka-peg-insertion.git
cd franka-peg-insertion
```

Install the project dependencies:

```bash
uv sync
```

**Dependency compatibility note:** The development environment uses a specific NVIDIA Warp version compatible with the available GPU driver. Reproducing the environment on another machine may require adjusting Warp and CUDA compatibility.

## Running the Simulation

### Corrected insertion

```bash
uv run python play_franka_insertion.py
```

### Uncorrected insertion

```bash
uv run python play_franka_insertion.py --no-correction
```

The simulation launches a MuJoCo viewer and executes the scripted joint-space trajectory.

The terminal displays:

- Computed IK configurations.
- Desired and actual joint positions.
- Joint tracking errors.
- Joint velocities.
- Proportional and derivative torque contributions.
- Bias-compensation torques.
- Final commanded torques.

## Project Structure

```text
franka-peg-insertion/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── play_franka_insertion.py
├── teaser.gif
└── src/
    └── franka_peg_insertion/
        ├── __init__.py
        ├── franka_insert_env_cfgs.py
        ├── franka_rl_cfg.py
        └── franka_emika_panda/
            ├── franka_constants.py
            ├── panda.xml
            ├── LICENSE
            └── assets/
```

### Key Files

**`play_franka_insertion.py`**

Implements inverse kinematics, trajectory generation, torque-level PD control, and simulation execution.

**`franka_insert_env_cfgs.py`**

Defines the insertion fixture, robot actions, observations, and simulation parameters.

**`franka_constants.py`**

Configures the Franka robot, actuators, initial joint positions, and insertion peg.

**`franka_rl_cfg.py`**

Contains PPO runner configuration for future learning-based experiments. The current scripted insertion demonstration does not train or execute a PPO policy.

## Current Status

The current implementation establishes a nominal insertion pipeline using inverse kinematics and torque-level control.

The environment supports comparisons between corrected and uncorrected insertion trajectories.

Quantitative evaluation of insertion success, tracking error, and robustness to randomized target offsets remains future work.

## Future Work

This project serves as a foundation for investigating residual reinforcement learning for contact-rich robotic manipulation.

Planned extensions include:

1. Introducing randomized fixture-position offsets.
2. Measuring nominal-controller insertion success and failure modes.
3. Training a residual reinforcement learning policy to modify the nominal controller's actions.
4. Evaluating whether learned spatial corrections improve robustness.
5. Investigating temporal adaptation when spatial corrections alone are insufficient.
6. Comparing learned policies against the scripted baseline.

The broader research objective is to investigate whether jointly adapting spatial corrections and trajectory progress can improve recovery from contact-related failures.

## Acknowledgments

Built using MJLab, MuJoCo, and NVIDIA Warp.

The Franka Panda robot model and associated assets retain their original licensing information in `src/franka_peg_insertion/franka_emika_panda/LICENSE`.