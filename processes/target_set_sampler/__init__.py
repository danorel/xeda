from .node_sampler import target_set_node_sampler


def target_set_sampler(groups_df):
    return {"node_sampling": target_set_node_sampler(groups_df)}
