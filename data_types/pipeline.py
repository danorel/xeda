import typing as t
import typing_extensions as te

from pydantic import BaseModel

from data_types.annotation import Annotation


"""
common variables for pipeline formats 
"""

ID: te.TypeAlias = str
Operator = te.Literal["by_facet", "by_neighbors", "by_superset"]
Dimension = te.Literal["i", "r", "z"]
TargetSet = te.Literal["Scattered"]


class Predicate(te.TypedDict):
    dimension: Dimension
    value: str


class RequestData(BaseModel):
    dataset_to_explore: str
    input_set_id: t.Optional[int] = None
    dimensions: t.Optional[t.List[str]] = None
    get_scores: t.Optional[bool] = False
    get_predicted_scores: t.Optional[bool] = False
    target_set: t.Optional[str] = None
    curiosity_weight: t.Optional[float] = None
    found_items_with_ratio: t.Optional[t.Dict[str, float]] = None
    target_items: t.Optional[t.List[str]] = None
    previous_set_states: t.Optional[t.List[t.List[float]]] = None
    previous_operation_states: t.Optional[t.List[t.List[float]]] = None
    seen_predicates: t.Optional[t.List[str]] = []
    dataset_ids: t.Optional[t.List[int]] = None
    seen_sets: t.Optional[t.List[int]] = []
    utility_weights: t.Optional[t.List[float]] = [0.333, 0.333, 0.334]
    previous_operations: t.Optional[t.List[str]] = []
    decreasing_gamma: t.Optional[bool] = False
    galaxy_class_scores: t.Optional[t.Dict[str, float]] = None
    weights_mode: t.Optional[str] = None


"""
dora pipeline format
"""


class InputSetDora(te.TypedDict):
    length: int
    id: int
    predicate: t.List[Predicate]
    silhouette: t.Optional[t.Any]
    novelty: t.Optional[t.Any]


class RequestDataDora(RequestData):
    pass


class PipelineItemDora(te.TypedDict):
    selectedSetId: t.Optional[str]
    operator: str
    checkedDimension: str
    url: str
    inputSet: t.Optional[InputSetDora]
    reward: float
    curiosityReward: float
    requestData: RequestDataDora


class AnnotatedPipelineItemDora(PipelineItemDora):
    annotation: Annotation


"""
eda4sum pipeline format
"""


class InputSetEda4Sum(te.TypedDict):
    length: int
    id: int
    predicate: t.List[Predicate]
    item_class: str
    uniformity: int


class RequestDataEda4Sum(RequestData):
    evolving_parameter: str
    evolution_type: str


class PipelineItemEda4Sum(te.TypedDict):
    selectedSetId: int
    operator: Operator
    checkedDimension: str
    url: str
    inputSet: t.Optional[InputSetEda4Sum]
    requestData: t.Optional[RequestDataEda4Sum]
    reward: int
    utility: float
    uniformity: float
    novelty: float
    distance: float
    utilityWeights: t.List[float]
    galaxy_class_score: float
    class_score_found_12: int
    class_score_found_15: int
    class_score_found_18: int
    class_score_found_21: int


class AnnotatedPipelineItemEda4Sum(PipelineItemEda4Sum):
    annotation: Annotation


PipelineType = te.Literal["dora", "eda4sum"]
PipelineKind = te.Literal["raw", "annotated"]

PipelineDora = t.List[PipelineItemDora]
PipelineEda4Sum = t.List[PipelineItemEda4Sum]

AnnotatedPipelineDora = t.List[AnnotatedPipelineItemDora]
AnnotatedPipelineEda4Sum = t.List[AnnotatedPipelineItemEda4Sum]

T = te.TypeVar("T")
K = te.TypeVar("K")

# TODO: te.Generic[T, K]
Pipeline = t.List
