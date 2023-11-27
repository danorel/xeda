import multiprocessing
import wandb

from constants import (
    EPISODES,
    WANDB_VERBOSE
)
from .A3C import Policy

def policy_trainer(target_set_id: str, mode: str):
    policy_config = {
        "dataset": "galaxies",
        "workers": multiprocessing.cpu_count(),
        "target_set": target_set_id,
        "mode": mode,
        "gamma": 0.99,
        "update_interval": 20,
        "actor_lr": 0.00003,
        "critic_lr": 0.00003,
        "icm_lr": 0.05,
        "lstm_steps": 3,
        "eval_interval": 10,
        "curiosity_ratio": 0.0,
        "counter_curiosity_ratio": 0.0,
        "operators": ["by_facet", "by_superset", "by_neighbors", "by_distribution"],
        "utility_mode": None,
        "utility_weights": [0.333, 0.333, 0.334],
    }
    policy_name = f"{target_set_id}_{mode}"

    if WANDB_VERBOSE:
        wandb.init(
            project="xeda",
            id=wandb.util.generate_id(),
            name=policy_name,
            config=policy_config,
        )

    policy = Policy(
        "pipeline",
        policy_name,
        policy_config,
        target_set_id,
        mode,
    )
    policy_models = policy.train(EPISODES)

    return policy_config, policy_models
