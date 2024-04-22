import chromadb
import json
import typing as t
import pandas as pd
import typing as t

from chromadb.utils import embedding_functions
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

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
from utils.vector_store import MilvusVectorStore

pretrained_embeddings = embedding_functions.OpenAIEmbeddingFunction(
    api_key=ai_constants.OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

vector_store = MilvusVectorStore(
    host=ai_constants.VECTOR_STORE_HOST,
    port=ai_constants.VECTOR_STORE_PORT,
    collection_name=ai_constants.VECTOR_STORE_COLLECTION
)

summarization_prompt_template = """Write a concise summary of "{text}". CONCISE SUMMARY:"""
summarization_prompt = PromptTemplate.from_template(summarization_prompt_template)

summarization_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
summarization_chain = LLMChain(llm=summarization_llm, prompt=summarization_prompt)

stuff_chain = StuffDocumentsChain(llm_chain=summarization_chain, document_variable_name="text")

groups_df = pd.read_csv(dataset_constants.GROUPS_CSV_PATH)

#@Guidance utils: make natural language explanations 

def make_instruction(name, value):
    return f"{name} = {value}" if value else None


def make_natural_language_documents(docs: list):
    for doc in docs:
        pipeline = json.loads(doc)
        last_node = pipeline[-1]
        # Extract natural language properties
        total_length, operator, checked_dimension, remaining_operators = (
            last_node['annotation']['total_length'],
            last_node['operator'],
            last_node['checkedDimension'],
            last_node['annotation']['remaining_operators']
        )
        # Derive natural language guidance features
        remaining_operators_count = sum(v for v in remaining_operators.values())
        remaining_operators_distribution = ', '.join([f"{operator} = {operator_count / remaining_operators_count}%" for operator, operator_count in remaining_operators.items()])
        # Build natural language query for summarization
        natural_language_instructions = [instruction for instruction in [
            make_instruction(name="most_probable_pipeline_length", value=total_length),
            make_instruction(name="most_probable_operator", value=operator),
            make_instruction(name="reachable_attribute_by_operator", value=checked_dimension),
            make_instruction(name="operator_probability_distribution", value=remaining_operators_distribution),
        ] if instruction is not None]
        natural_language_document = Document(page_content=", ".join(natural_language_instructions))
        yield natural_language_document

#@Guidance utils: pipeline annotator

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


def explain(partial_pipeline: Pipeline):
    partial_annotated_pipeline = annotate_partial_pipeline(groups_df, partial_pipeline)
    partial_pipeline_partial_annotation = ';'.join([node_to_encoding(node) for node in partial_annotated_pipeline])
    partial_annotation_embeddings = pretrained_embeddings([partial_pipeline_partial_annotation])

    most_similar_responses = vector_store.search(
        partial_annotation_embeddings,
        k=3
    )
    
    if not len(most_similar_responses):
        return "Not found any similar pipelines"
    else:
        return stuff_chain.run(make_natural_language_documents([r['document'] for r in most_similar_responses]))