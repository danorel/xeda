import typing as t

from random import randrange
from time import time

from constants import DATA_NAME, PIPELINE_MIN_SIZE, PIPELINE_MAX_SIZE
from typings.pipeline import OperatorRequestData
from ..utils.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from ..utils.model_manager import ModelManager
from .operators import by_distribution, by_facet, by_neighbors, by_superset


def _get_initial_request_data(database_pipeline_cache, info):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[DATA_NAME]
    request_data = OperatorRequestData(
        input_set_id=-1,
        dataset_to_explore=DATA_NAME,
        dataset_ids=[],
        weights_mode="custom",
        utility_weights=info.get("utility_weights"),
        found_items_with_ratio={},
        decreasing_gamma=False,
        galaxy_class_scores=None,
        dimensions=list(pipeline.ordered_dimensions.keys()),
        seen_sets=[],
        previous_operations=[],
        previous_operation_states=None,
        previous_set_states=None,
        get_scores=True,
        get_predicted_scores=True,
    )
    return request_data


def pipeline_to_request_data(pipeline) -> t.Optional[OperatorRequestData]:
    if not len(pipeline):
        return None
    last_node = pipeline[-1]
    last_request_data = last_node["requestData"]
    operator_request_data = OperatorRequestData(**last_request_data)
    return operator_request_data


def next_node_from_request_data(request_data: OperatorRequestData, args, **kwargs):
    return next_node(request_data, *args, **kwargs)


def next_node_from_pipeline(pipeline, *args, **kwargs):
    request_data = pipeline_to_request_data(pipeline)
    return next_node(request_data, *args, **kwargs)


def next_node(
    prev_request: OperatorRequestData,
    database_pipeline_cache,
    model_manager,
    operator: t.Optional[str] = None,
    dimension: t.Optional[str] = None
):
    if operator is None:
        if len(prev_request.previous_operations):
            operator = prev_request.previous_operations[-1]
        else:
            operator = "by_facet"
    else:
        if dimension is None:
            raise ValueError("Please, pass dimension as well")
        prev_request.dimensions[0] = dimension

    """
    prediction contains:
    {
        "predictedOperation": operation,
        "predictedDimension": dimension,
        "predictedSetId": set_id,
        "foundItemsWithRatio": state_encoder.found_items_with_ratio,
        "setStates": new_set_states,
        "operationStates": new_operation_states,
        "reward": reward
    }
    """
    start_time = time()
    if operator == "by_distribution":
        prediction = by_distribution(
            database_pipeline_cache, model_manager, prev_request
        )
    elif operator == "by_facet":
        prediction = by_facet(database_pipeline_cache, model_manager, prev_request)
    elif operator == "by_neighbors":
        prediction = by_neighbors(database_pipeline_cache, model_manager, prev_request)
    elif operator == "by_superset":
        prediction = by_superset(database_pipeline_cache, model_manager, prev_request)
    else:
        raise Exception("Operator not implemented")
    print(f"Finished applying operator in {time() - start_time}s")

    if not prediction:
        raise ValueError("Prediction failed")

    next_request = OperatorRequestData.parse_obj(prev_request.dict())
    next_set_id = prediction.get("predictedSetId")
    if next_set_id is not None:
        next_request.input_set_id = next_set_id
        next_request.seen_sets.append(next_set_id)
    next_request.found_items_with_ratio = prediction.get("foundItemsWithRatio")
    next_request.previous_operations.append(prediction.get("predictedOperation"))
    next_request.previous_set_states = prediction.get("setStates")
    next_request.previous_operation_states = prediction.get("operationStates")
    next_request.dataset_ids = list(map(lambda x: x.get("id"), prediction.get("sets")))

    next_node = {
        "selectedSetId": next_set_id,
        "operator": operator,
        "checkedDimension": prediction.get("predictedDimension"),
        "inputSet": prev_request.input_set_id,
        "reward": prediction.get("reward", 0),
        "requestData": next_request.dict(),
        "curiosityReward": prediction.get("curiosityReward"),
        "utility": prediction.get("utility"),
        "uniformity": prediction.get("uniformity"),
        "novelty": prediction.get("novelty"),
        "distance": prediction.get("distance"),
        "utilityWeights": prediction.get("utility_weights"),
        "galaxy_class_score": prediction.get("galaxy_class_score"),
        "class_score_found_12": prediction.get("class_score_found_12"),
        "class_score_found_15": prediction.get("class_score_found_15"),
        "class_score_found_18": prediction.get("class_score_found_18"),
        "class_score_found_21": prediction.get("class_score_found_21"),
    }

    return next_node, next_request


def sample_pipeline_from_models(models, database_pipeline_cache, info, logger):
    model_manager = ModelManager(database_pipeline_cache[DATA_NAME], models)
    request_data = _get_initial_request_data(database_pipeline_cache, info)
    pipeline = []
    pipeline_size = randrange(PIPELINE_MIN_SIZE, PIPELINE_MAX_SIZE)
    logger.info(f"Generating Pipeline of size {pipeline_size}")
    for i in range(pipeline_size):
        try:
            start_time = time()
            node, request_data = next_node_from_request_data(
                request_data, database_pipeline_cache, model_manager
            )
            print(f"Finished iteration in {time() - start_time}s")
            pipeline.append(node)
        except ValueError:
            logger.info(
                f"Unexpectedly exited from pipeline generation on step {i}. Saving pipeline as it is..."
            )
            break
    return pipeline
