import re
import chromadb
import json
import typing as t
import pandas as pd
import typing as t
import heapq
import statistics

from collections import Counter, defaultdict
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

pretrained_embeddings = embedding_functions.OpenAIEmbeddingFunction(
    api_key=ai_constants.OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

vector_store = chromadb.HttpClient(
    host=ai_constants.VECTOR_STORE_HOST, 
    port=ai_constants.VECTOR_STORE_PORT
)

try:
    vector_collection = vector_store.create_collection(
        name=ai_constants.VECTOR_STORE_COLLECTION, 
        embedding_function=pretrained_embeddings,
        metadata={
            "hnsw:space": "cosine"
        }
    )
except:
    vector_collection = vector_store.get_collection(ai_constants.VECTOR_STORE_COLLECTION)


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
        pipeline = json.loads(doc[0])
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


def mine_dimensions(similar_pipeline_encodings):
  dimensions = ['remaining_dimensions_u', 'remaining_dimensions_g', 'remaining_dimensions_r', 'remaining_dimensions_i', 'remaining_dimensions_z', 'remaining_dimensions_petroRad_r', 'remaining_dimensions_redshift']
  dimensions_values = defaultdict(float)

  for dimension in dimensions:
      dimensions_values[dimension] += sum(encoding.get(dimension, 0.0) for encoding in similar_pipeline_encodings)

  total_count = sum(dimensions_values.values()) + 0.000001
  percentages_length = {key: value / total_count * 100 for key, value in dimensions_values.items()}

  top_two_values = heapq.nlargest(2, percentages_length.values())

  top_two_keys = [(key, value) for key, value in percentages_length.items() if value in top_two_values]
  return top_two_keys


def mine_operators(similar_pipeline_encodings):
  operators = ['remaining_operators_by_neighbors', 'remaining_operators_by_superset', 'remaining_operators_by_distribution', 'remaining_operators_by_facet']
  operator_values = defaultdict(float)

  for operator in operators:
      operator_values[operator] += sum(encoding.get(operator, 0.0) for encoding in similar_pipeline_encodings)

  total_count = sum(operator_values.values()) + 0.000001
  percentages_length = {key: value / total_count * 100 for key, value in operator_values.items()}

  top_two_values = heapq.nlargest(4, percentages_length.values())
  top_two_keys = [(key, value) for key, value in percentages_length.items() if value in top_two_values]
  return top_two_keys


def mine_length(similar_pipeline_encodings):
  counts_length = Counter(encoding['total_length'] for encoding in similar_pipeline_encodings)
  percentages_length = {key: count / len(counts_length) * 100 for key, count in counts_length.items()}

  max_length = max(percentages_length, key=lambda k: percentages_length[k])
  return int(max_length)


def mine_count_of_similar_pipelines(similar_pipeline_encodings):
  return len(similar_pipeline_encodings)


def mine_familiarity(similar_pipeline_encodings):
    counts = [
        encoding['familiarity'] 
        for encoding in similar_pipeline_encodings
    ]
    return (statistics.median(counts))


def terminal_node_to_encoding(node: dict, step: int):
    encoding = {
        "total_length": node["annotation"]["total_length"],
        "current_operator": node['operator'],
        "current_dimension": node['checkedDimension'],
        "current_utility": node['utility'],
        "current_novelty": node['novelty'],
        "current_diversity": node['distance'],
        "current_galaxy_class_score": node['galaxy_class_score'],
        "current_step": step
    }
    return encoding


def pipeline_to_encoding(pipeline: list, step: int):
    try:
        if not len(pipeline):
            return {}
        
        terminal_node = pipeline[-1]
        terminal_encoding = terminal_node_to_encoding(terminal_node, step) 

        return terminal_encoding
    except:
        return {}


def explain(partial_pipeline: Pipeline):
    partial_annotated_pipeline = annotate_partial_pipeline(groups_df, partial_pipeline)
    partial_pipeline_partial_annotation = ';'.join([node_to_encoding(node) for node in partial_annotated_pipeline])
    partial_annotation_embeddings = pretrained_embeddings([partial_pipeline_partial_annotation])

    most_similar_responses = vector_collection.query(
        query_embeddings=partial_annotation_embeddings,
        n_results=3,
        include=["documents", "distances"]
    )
    
    if not len(most_similar_responses['documents'][0]):
        return "Not found any similar pipelines"
    else:
        step = len(partial_pipeline)
        similar_pipeline_encodings = [
            pipeline_to_encoding(pipeline, step)
            for pipeline in most_similar_responses['documents']
        ]
        total_length, operator, dimension, familiarity, count_of_similar_pipelines = (
            mine_length(similar_pipeline_encodings),
            mine_operators(similar_pipeline_encodings),
            mine_dimensions(similar_pipeline_encodings),
            mine_familiarity(similar_pipeline_encodings),
            mine_count_of_similar_pipelines(similar_pipeline_encodings),
        )
        steps_left = total_length - step
        explanation = f"""
        On average \033[94m{steps_left}\033[0m step/s, you will reach a scattered/concentrated set with an expected final familiarity of \033[94m{familiarity}\033[0m. \n
        You are more likely to get there by focusing on the \033[94m{operator[0][0][20:]}\033[0m and \033[94m{operator[1][0][20:]}\033[0m operators and on \033[94m{dimension[0][0][21:]}\033[0m and \033[94m{dimension[1][0][21:]}\033[0m dimensions \n
        You will probably finish with total length of \033[94m{total_length}\033[0m. \n'
        You get this guidance because: in the \033[94m{count_of_similar_pipelines}\033[0m similar pipelines the following distribution of operator \033[94m{operator[0][0][20:]}\033[0m is \033[94m{round(operator[0][1], 2)}\033[0m, \033[94m{operator[1][0][20:]}\033[0m is \033[94m{round(operator[1][1], 2)}\033[0m, \033[94m{operator[2][0][20:]}\033[0m is \033[94m{round(operator[2][1], 2)}\033[0m, \033[94m{operator[3][0][20:]}\033[0m is \033[94m{round(operator[3][1], 2)}\033[0m; \n'
        the distribution of dimension \033[94m{dimension[0][0][21:]}\033[0m is \033[94m{round(dimension[0][1], 2)}\033[0m, \033[94m{dimension[1][0][21:]}\033[0m is \033[94m{round(dimension[1][1], 2)}\033[0m. ')
        """.lstrip()
        return explanation