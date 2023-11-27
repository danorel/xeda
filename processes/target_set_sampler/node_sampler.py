import random

from constants import (
    ATTEMPTS,
    MAX_ITEM_SET_NODES,
    MIN_ITEM_SET_NODES,
    SAMPLE_AMOUNT,
    SAMPLING_RATE,
)

def make_item_set_sampler(sampling_rate: float):
    def sample(item_set: set):
        k = round(len(item_set) * sampling_rate)
        return random.choices(list(item_set), k=k)

    return sample


def make_item_set_validator(min_item_set_nodes: int, max_item_set_nodes: int):
    def satisfies_conditions(item_set: set):
        return (
            len(item_set) >= min_item_set_nodes and len(item_set) <= max_item_set_nodes
        )

    return satisfies_conditions


def target_set_node_sampler(members_df):
    sample, satisfies_conditions = (
        make_item_set_sampler(SAMPLING_RATE),
        make_item_set_validator(MIN_ITEM_SET_NODES, MAX_ITEM_SET_NODES),
    )

    target_sets = []
    for _ in range(SAMPLE_AMOUNT):
        item_set = set()
        for _ in range(ATTEMPTS):
            input_set_id = random.randint(0, members_df.shape[0] - 1)
            item_set = set(
                [int(id) for id in members_df.iloc[input_set_id, 2][1:-1].split(", ")]
            )
            if satisfies_conditions(item_set):
                break
        if not len(item_set):
            return None
        target_sets.append(sample(item_set))

    return target_sets
