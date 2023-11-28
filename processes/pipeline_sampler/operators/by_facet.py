import json
import traceback

from data_types.pipeline import RequestData
from ...utils.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from ...utils.model_manager import ModelManager
from .get_items_sets import get_items_sets


def by_facet(
    database_pipeline_cache, model_manager: ModelManager, operator_request: RequestData
):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore
        ]
        if operator_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets([operator_request.input_set_id])[
                0
            ]
        number_of_groups = 10 if len(operator_request.dimensions) == 1 else 5
        result_sets = pipeline.by_facet(
            dataset=dataset,
            attributes=operator_request.dimensions,
            number_of_groups=number_of_groups,
        )
        result_sets = [d for d in result_sets if d.set_id != None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        operation_identifier = (
            f"by_facet-{operator_request.dimensions[0]}-{dataset.set_id}"
        )
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        prediction_result = {}
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
        return None
