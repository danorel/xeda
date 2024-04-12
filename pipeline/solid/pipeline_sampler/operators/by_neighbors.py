import json
import traceback

from typings.pipeline import OperatorRequestData
from ...utils.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from ...utils.model_manager import ModelManager
from .get_items_sets import get_items_sets


def by_neighbors(
    database_pipeline_cache, model_manager: ModelManager, request_data: OperatorRequestData
):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            request_data.dataset_to_explore
        ]
        dataset = pipeline.get_groups_as_datasets([request_data.input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=request_data.dimensions
        )
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]

        prediction_result = {}

        operation_identifier = (
            f"by_neighbors-{request_data.dimensions[0]}-{dataset.set_id}"
        )
        if not operation_identifier in request_data.previous_operations:
            request_data.previous_operations.append(operation_identifier)
        if request_data.weights_mode != None:
            prediction_result = model_manager.get_prediction(
                result_sets,
                request_data.target_items,
                request_data.found_items_with_ratio,
                request_data.previous_set_states,
                request_data.previous_operation_states,
            )
        result = get_items_sets(
            result_sets,
            pipeline,
            request_data.get_scores,
            request_data.get_predicted_scores,
            request_data.galaxy_class_scores,
            seen_sets=set(request_data.seen_sets),
            previous_dataset_ids=set(request_data.dataset_ids),
            utility_weights=request_data.utility_weights,
            previous_operations=request_data.previous_operations,
            decreasing_gamma=request_data.decreasing_gamma,
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
