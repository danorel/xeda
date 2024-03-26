import json
import traceback

from typings.pipeline import RequestData
from ...utils.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from ...utils.model_manager import ModelManager
from .get_items_sets import get_items_sets


def by_neighbors(
    database_pipeline_cache, model_manager: ModelManager, operator_request: RequestData
):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore
        ]
        dataset = pipeline.get_groups_as_datasets([operator_request.input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=operator_request.dimensions
        )
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]

        prediction_result = {}

        operation_identifier = (
            f"by_neighbors-{operator_request.dimensions[0]}-{dataset.set_id}"
        )
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        if operator_request.weights_mode != None:
            prediction_result = model_manager.get_prediction(
                result_sets,
                operator_request.target_items,
                operator_request.found_items_with_ratio,
                operator_request.previous_set_states,
                operator_request.previous_operation_states,
            )
        result = get_items_sets(
            result_sets,
            pipeline,
            operator_request.get_scores,
            operator_request.get_predicted_scores,
            operator_request.galaxy_class_scores,
            seen_sets=set(operator_request.seen_sets),
            previous_dataset_ids=set(operator_request.dataset_ids),
            utility_weights=operator_request.utility_weights,
            previous_operations=operator_request.previous_operations,
            decreasing_gamma=operator_request.decreasing_gamma,
        )
        result.update(prediction_result)
        return result
    except Exception as error:
        print(error)
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0
