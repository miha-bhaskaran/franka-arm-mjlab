"""Franka Panda peg-in-hole insertion task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .franka_insert_env_cfgs import franka_insertion_env_cfg
from .franka_rl_cfg import franka_insertion_ppo_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Franka-Peg-Insertion",
    env_cfg=franka_insertion_env_cfg(),
    play_env_cfg=franka_insertion_env_cfg(play=True),
    rl_cfg=franka_insertion_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)