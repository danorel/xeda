import wandb
import typing as t

from constants import (
    POLICY_EPISODES,
    POLICY_EPISODE_STEPS_CONCENTRATED,
    POLICY_EPISODE_STEPS_SCATTERED,
    POLICY_WANDB_PROJECT,
    POLICY_WANDB_VERBOSE,
)
from constants.processes.policy_trainer import POLICY_BASE_CONFIG
from .A3C import Policy


def policy_trainer(target_set_id: str, target_set_items: t.List[int], mode: str):
    policy_name = f"{target_set_id}_{mode}"

    if mode == "scattered":
        policy_episode_steps = POLICY_EPISODE_STEPS_SCATTERED
    elif mode == "concentrated":
        policy_episode_steps = POLICY_EPISODE_STEPS_CONCENTRATED
    else:
        raise NotImplementedError(
            "Please, provide training mode: 'scattered' or 'concentrated'"
        )

    policy_config = {
        **POLICY_BASE_CONFIG,
        "target_set": target_set_id,
        "mode": mode,
        "episode_steps": policy_episode_steps,
    }

    if POLICY_WANDB_VERBOSE:
        wandb.init(
            project=POLICY_WANDB_PROJECT,
            id=wandb.util.generate_id(),
            name=policy_name,
            config=policy_config,
        )

    policy = Policy(
        "pipeline",
        policy_name,
        policy_config,
        target_set_id,
        target_set_items,
        mode,
    )
    policy_models = policy.train(POLICY_EPISODES)

    return policy_config, policy_models
