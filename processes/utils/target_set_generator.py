import json
import random

from constants import (
    SAMPLING_MIN_ITEM_SET_NODES,
    SAMPLING_MAX_ITEM_SET_NODES,
    TARGET_SETS_PATH,
)


class TargetSetGenerator:
    @staticmethod
    def get_diverse_target_set(number_of_samples=100):
        target_files = list(TARGET_SETS_PATH.glob("*.json"))
        initial_target_set = []
        for target_file in target_files:
            with target_file.open("r") as f:
                target_set = json.load(f)
            if len(target_set) > number_of_samples:
                initial_target_set += random.choices(target_set, k=number_of_samples)
            else:
                initial_target_set += target_set
        return set(initial_target_set)

    @staticmethod
    def get_concentrated_target_set():
        target_files = list(TARGET_SETS_PATH.glob("*.json"))
        while True:
            target_file = random.choice(target_files)
            with target_file.open("r") as f:
                target_set = json.load(f)
            if (
                len(target_set) >= SAMPLING_MIN_ITEM_SET_NODES
                and len(target_set) <= SAMPLING_MAX_ITEM_SET_NODES
            ):
                return set(target_set)
