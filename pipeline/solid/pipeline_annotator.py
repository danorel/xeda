import pandas as pd
import typing as t

from typings.pipeline import (
    PipelineEda4Sum,
    PipelineItemEda4Sum,
    AnnotatedPipelineEda4Sum,
    AnnotatedPipelineItemEda4Sum
)
from typings.annotation import Annotation


def find_item_set(
    members: pd.DataFrame, pipeline_body_item: PipelineItemEda4Sum
) -> t.Set[str]:
    input_set_id = pipeline_body_item["inputSet"]["id"]
    members = members.loc[members["id"] == input_set_id]["members"].iloc[0]
    input_set = set(members[1:-1].split(", "))
    return input_set


def _find_remaining_operators(pipeline: PipelineEda4Sum) -> dict:
    operators = {}
    pipeline = pipeline[1:]
    for pipeline_item in pipeline:
        current_operator = pipeline_item["operator"]
        if current_operator in operators.keys():
            operators[current_operator] += 1
        else:
            operators[current_operator] = 1

    return operators


def _find_remaining_dimensions(pipeline: PipelineEda4Sum) -> dict:
    dimensions = {}
    pipeline = pipeline[1:]
    for pipeline_item in pipeline:
        current_dimension = pipeline_item["checkedDimension"]
        if current_dimension in dimensions.keys():
            dimensions[current_dimension] += 1
        else:
            dimensions[current_dimension] = 1

    return dimensions


def _find_delta_uniformity(
    pipeline_item_current: PipelineItemEda4Sum, pipeline_item_next: PipelineItemEda4Sum
) -> float:
    return pipeline_item_next["uniformity"] - pipeline_item_current["uniformity"]


def _find_delta_novelty(
    pipeline_item_current: PipelineItemEda4Sum, pipeline_item_next: PipelineItemEda4Sum
) -> float:
    return pipeline_item_next["novelty"] - pipeline_item_current["novelty"]


def _find_delta_diversity(
    pipeline_item_current: PipelineEda4Sum, pipeline_item_next: PipelineEda4Sum
) -> float:
    return pipeline_item_next["distance"] - pipeline_item_current["distance"]


def _find_utility_weights(
    pipeline_item_current: PipelineEda4Sum, pipeline_item_next: PipelineEda4Sum
) -> t.List[float]:
    return [
        pipeline_item_next["utilityWeights"][i]
        - pipeline_item_current["utilityWeights"][i]
        for i in range(3)
    ]


def _find_familiarity_curiosity(seen_galaxies, item_members) -> t.Tuple[float, float]:
    if len(seen_galaxies) == 0:
        return [0.0, 0.0]
    else:
        common_members_number = sum(1 for elem in item_members if elem in seen_galaxies)
        familiarity = common_members_number / (len(seen_galaxies))
        separate_members_number = sum(
            1 for elem in item_members if elem not in seen_galaxies
        )
        curiosity = separate_members_number / (len(seen_galaxies))
        return [familiarity, curiosity]


def annotate_pipeline(
    groups_df: pd.DataFrame,
    pipeline: PipelineEda4Sum, 
    logger = None
) -> AnnotatedPipelineEda4Sum:
    seen_galaxies = []

    length = len(pipeline)
    annotated_pipeline: AnnotatedPipelineEda4Sum = []

    for item in range(length):
        if item is not length - 1:
            delta_uniformity = _find_delta_uniformity(
                pipeline[item], pipeline[item + 1]
            )
            delta_novelty = _find_delta_novelty(pipeline[item], pipeline[item + 1])
            delta_diversity = _find_delta_diversity(pipeline[item], pipeline[item + 1])
            delta_utility_weights = _find_utility_weights(
                pipeline[item], pipeline[item + 1]
            )
        else:
            delta_uniformity = 0
            delta_novelty = 0
            delta_diversity = 0
            delta_utility_weights = [0.0, 0.0, 0.0]

        familiarity = 0.0
        curiosity = 0.0

        if "requestData" in pipeline[item].keys():
            input_set_id = pipeline[item]["selectedSetId"]
            item_members = groups_df.loc[groups_df["id"] == input_set_id]["members"]

            for i in item_members:
                list_members = i[1:-1].split(", ")
                result_members = [int(num) for num in list_members]

            if item_members.empty:
                if logger is not None:
                    logger.warn(f"Node[id={input_set_id}] is missing in .csv")
            else:
                familiarity, curiosity = _find_familiarity_curiosity(
                    seen_galaxies, result_members
                )
                seen_galaxies.extend(result_members)

        annotation = Annotation(
            total_length=length,
            remaining_operators=_find_remaining_operators(pipeline[item:]),
            remaining_dimensions=_find_remaining_dimensions(pipeline[item:]),
            current_operator=pipeline[item]["operator"],
            current_dimension=pipeline[item]["checkedDimension"],
            delta_uniformity=delta_uniformity,
            delta_novelty=delta_novelty,
            delta_diversity=delta_diversity,
            delta_utilityWeights=delta_utility_weights,
            current_uniformity=pipeline[item]["uniformity"],
            current_novelty=pipeline[item]["novelty"],
            current_diversity=pipeline[item]["distance"],
            current_utilityWeights=pipeline[item]["utilityWeights"],
            final_uniformity=pipeline[-1]["uniformity"],
            final_novelty=pipeline[-1]["novelty"],
            final_diversity=pipeline[-1]["distance"],
            final_utilityWeights=pipeline[-1]["utilityWeights"],
            familiarity=familiarity,
            curiosity=curiosity,
        )
        if 'annotation' in pipeline[item]:
            pipeline[item].pop('annotation')
        annotated_pipeline_item = AnnotatedPipelineItemEda4Sum(
            **pipeline[item],
            annotation=annotation
        )
        annotated_pipeline.append(annotated_pipeline_item)

    return annotated_pipeline
