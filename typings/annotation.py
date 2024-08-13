import typing_extensions as te


class Annotation(te.TypedDict):
    total_length: int
    remaining_operators: dict
    current_operator: str

    remaining_dimensions: dict
    current_dimension: str

    current_novelty: float
    current_uniformity: float
    current_diversity: float
    current_utilityWeights: list

    delta_novelty: float
    delta_uniformity: float
    delta_diversity: float
    delta_utilityWeights: list

    final_novelty: float
    final_uniformity: float
    final_diversity: float
    final_utilityWeights: list

    curiosity: float
    familiarity: float


class PartialAnnotation(te.TypedDict):
    current_operator: str
    current_dimension: str

    current_novelty: float
    current_uniformity: float
    current_diversity: float
    current_utilityWeights: list

    delta_novelty: float
    delta_uniformity: float
    delta_diversity: float
    delta_utilityWeights: list

    curiosity: float
    familiarity: float
