import os
import multiprocessing

POLICY_WANDB_PROJECT = os.getenv("POLICY_WANDB_PROJECT", "xeda")
POLICY_WANDB_VERBOSE = os.getenv("POLICY_WANDB_API_KEY") is not None

POLICY_EPISODES = int(os.getenv("POLICY_EPISODES", "1"))
POLICY_EPISODE_STEPS_SCATTERED = int(os.getenv("POLICY_EPISODE_STEPS_SCATTERED", "1"))
POLICY_EPISODE_STEPS_CONCENTRATED = int(
    os.getenv("POLICY_EPISODE_STEPS_CONCENTRATED", "1")
)

POLICY_BASE_CONFIG = {
    "dataset": "galaxies",
    "workers": multiprocessing.cpu_count(),
    "gamma": 0.99,
    "save_interval": 250,
    "eval_interval": 5,
    "update_interval": 10,
    "actor_lr": 0.00003,
    "critic_lr": 0.00003,
    "icm_lr": 0.05,
    "lstm_steps": 3,
    "curiosity_ratio": 0.0,
    "counter_curiosity_ratio": 0.0,
    "operators": ["by_facet", "by_superset", "by_neighbors", "by_distribution"],
    "utility_mode": None,
    "utility_weights": [0.333, 0.333, 0.334],
}
