import json
import numpy as np

from constants import DATA_NAME, TARGET_SETS_PATH, TARGET_SETS_MEAN_VECTORS_PATH


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def generate_mean_vectors(pipeline):
    mean_vectors = {}

    original_attributes = list(map(lambda x: x+"_original", list(pipeline.ordered_dimensions)))
    original_attributes += [x for x in pipeline.exploration_columns if not x in pipeline.ordered_dimensions]

    for attribute in original_attributes:
        mean = pipeline.initial_collection[attribute].mean()
        std = pipeline.initial_collection[attribute].std()
        pipeline.initial_collection[attribute] = (pipeline.initial_collection[attribute] - mean) / std

    for file in TARGET_SETS_PATH.rglob("*"):
        with file.open('r') as f:
            ids = json.load(f)
            galaxies = pipeline.initial_collection[pipeline.initial_collection[f"{DATA_NAME}.objID"].isin(ids)]
            means_vector = []
            for attribute in original_attributes:
                if attribute.replace("_original", "") in pipeline.ordered_dimensions:
                    mean = galaxies[attribute].mean()
                    means_vector.append(mean)
            mean_vectors[file.name.replace('.json', '')] = means_vector

    with TARGET_SETS_MEAN_VECTORS_PATH.open('w') as f:
        json.dump(mean_vectors, f, indent=1, default=np_encoder)
