import typing_extensions as te


class Annotation(te.TypedDict):
    total_length: int
    remaining_operators: dict
    current_operator: str

    current_novelty: float
    current_uniformity: float
    current_diversity: float
    current_score_galaxy: float
    current_utilityWeights: list

    delta_novelty: float
    delta_uniformity: float
    delta_diversity: float
    delta_score_galaxy: float
    delta_utilityWeights: list

    final_novelty: float
    final_uniformity: float
    final_diversity: float
    final_score_galaxy: float
    final_utilityWeights: list

    curiosity: float
    familiarity: float
