import json
import pandas as pd
import typing as t
import openai

import constants.ai as ai_constants
import constants.dataset as dataset_constants
from typings.pipeline import (
    AnnotatedPartialPipelineEda4Sum, 
    AnnotatedPipelineItemEda4Sum,
    Pipeline,
    PipelineEda4Sum, 
    PipelineItemEda4Sum
)
from typings.annotation import PartialAnnotation
from utils.vector_store import MilvusVectorStore, SearchResult

groups_df = pd.read_csv(dataset_constants.GROUPS_CSV_PATH)

vector_store = MilvusVectorStore(
    host=ai_constants.VECTOR_STORE_HOST,
    port=ai_constants.VECTOR_STORE_PORT,
    collection_name=ai_constants.VECTOR_STORE_COLLECTION
)

embedding_client = openai.OpenAI(api_key=ai_constants.OPENAI_API_KEY)


def make_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model=ai_constants.OPENAI_EMBEDDINGS_MODEL
    )
    return response.data[0].embedding


def find_item_set(
    members: pd.DataFrame, pipeline_body_item: PipelineItemEda4Sum
) -> t.Set[str]:
    input_set_id = pipeline_body_item["inputSet"]["id"]
    members = members.loc[members["id"] == input_set_id]["members"].iloc[0]
    input_set = set(members[1:-1].split(", "))
    return input_set


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


def annotate_partial_pipeline(
    groups_df: pd.DataFrame, partial_pipeline: PipelineEda4Sum
) -> AnnotatedPartialPipelineEda4Sum:
    seen_galaxies = []

    length = len(partial_pipeline)
    annotated_partial_pipeline: AnnotatedPartialPipelineEda4Sum = []

    for item in range(length):
        if item is not length - 1:
            delta_uniformity = _find_delta_uniformity(
                partial_pipeline[item], partial_pipeline[item + 1]
            )
            delta_novelty = _find_delta_novelty(partial_pipeline[item], partial_pipeline[item + 1])
            delta_diversity = _find_delta_diversity(partial_pipeline[item], partial_pipeline[item + 1])
            delta_utility_weights = _find_utility_weights(
                partial_pipeline[item], partial_pipeline[item + 1]
            )
        else:
            delta_uniformity = 0
            delta_novelty = 0
            delta_diversity = 0
            delta_utility_weights = [0.0, 0.0, 0.0]

        familiarity = 0.0
        curiosity = 0.0

        if "requestData" in partial_pipeline[item].keys():
            input_set_id = partial_pipeline[item]["selectedSetId"]
            item_members = groups_df.loc[groups_df["id"] == input_set_id]["members"]

            for i in item_members:
                list_members = i[1:-1].split(", ")
                result_members = [int(num) for num in list_members]

            if item_members.empty:
                print(f"Node[id={input_set_id}] is missing in .csv")
            else:
                familiarity, curiosity = _find_familiarity_curiosity(
                    seen_galaxies, result_members
                )
                seen_galaxies.extend(result_members)

        partial_annotation = PartialAnnotation(
            current_operator=partial_pipeline[item]["operator"],
            current_dimension=partial_pipeline[item]["checkedDimension"],
            delta_uniformity=delta_uniformity,
            delta_novelty=delta_novelty,
            delta_diversity=delta_diversity,
            delta_utilityWeights=delta_utility_weights,
            current_uniformity=partial_pipeline[item]["uniformity"],
            current_novelty=partial_pipeline[item]["novelty"],
            current_diversity=partial_pipeline[item]["distance"],
            current_utilityWeights=partial_pipeline[item]["utilityWeights"],
            familiarity=familiarity,
            curiosity=curiosity,
        )
        annotated_pipeline_item = AnnotatedPipelineItemEda4Sum(
            **partial_pipeline[item], annotation=partial_annotation
        )
        annotated_partial_pipeline.append(annotated_pipeline_item)

    return annotated_partial_pipeline


def node_to_encoding(node):
    annotation = node["annotation"]
    node_encoding = []
    for k, v in annotation.items():
        if isinstance(v, dict):
            for key in v:
                node_encoding.append(f"{k}_{key} = {v[key]}")
        else:
            node_encoding.append(f"{k} = {v}")
    return ', '.join(node_encoding)


def pipeline_to_encoding(annotated_pipeline: PipelineEda4Sum):
    return ';'.join([node_to_encoding(node) for node in annotated_pipeline])


def pipeline_to_embedding(partial_pipeline: PipelineEda4Sum):
    partial_annotated_pipeline = annotate_partial_pipeline(groups_df, partial_pipeline)
    partial_pipeline_encoding = pipeline_to_encoding(partial_annotated_pipeline)
    partial_pipeline_embedding = make_embedding(partial_pipeline_encoding)
    return partial_pipeline_embedding


def results_to_pipelines(search_results: t.List[SearchResult]) -> t.List[Pipeline]:
    pipelines = []
    for search_result in search_results:
        try:
            pipeline = json.loads(search_result['document'])
            pipelines.append(pipeline)
        except:
            print(f"Failed to fetch pipeline by id {search_result['id']}")
    return pipelines


def results_to_scores(search_results: t.List[SearchResult]) -> t.List[float]:
    scores = []
    for search_result in search_results:
        try:
            score = float(search_result['score'])
            scores.append(score)
        except:
            print(f"Failed to fetch pipeline by id {search_result['id']}")
    return scores


def make_pipeline_snapshot(pipeline: Pipeline, score: float, current_step: int, lookup_recent_steps: int = 3):
    if current_step < len(pipeline):
        target_node = pipeline[current_step]
    else:
        target_node = pipeline[-1]
    target_annotation = target_node.get("annotation", {})

    total_length = target_annotation.get("total_length")
    remaining_operators = target_annotation.get("remaining_operators", {})
    if len(remaining_operators):
        remaining_operators = ', '.join([
            f"{operator_name}: {operator_count}"
            for operator_name, operator_count in remaining_operators.items()
        ])
    else:
        remaining_operators = "None"

    recent_nodes = pipeline[-(lookup_recent_steps+1):-1]
    recent_annotations = [node.get("annotation") for node in recent_nodes if node.get("annotation") is not None]
    if len(recent_annotations): 
        recent_operators = ', '.join([annotation.get("current_operator") for annotation in recent_annotations if annotation.get("current_operator") is not None])
        recent_dimensions = ', '.join([annotation.get("current_dimension") for annotation in recent_annotations if annotation.get("current_dimension") is not None])
    else:
        recent_operators = "None"
        recent_dimensions = "None"
    
    return f"cosine similarity score = {round(score, 3)}; total length = {total_length}; recent operators = {recent_operators}; recent dimensions = {recent_dimensions}; remaining operators = {remaining_operators}."


def make_explanation_details(neighbouring_pipelines: t.List[Pipeline], neighbouring_scores: t.List[float], current_step: int):
    return [
        {
            "order": f"{order}-th pipeline",
            "snapshot": make_pipeline_snapshot(pipeline, score, current_step)
        }
        for order, (pipeline, score) in enumerate(zip(neighbouring_pipelines, neighbouring_scores), 1)
    ]