import copy
import json
import pandas as pd
import typing as t
import uuid
import openai

from constants import (
    GROUPS_CSV_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDINGS_MODEL,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_HOST,
    VECTOR_STORE_PORT
)
from typings.pipeline import Pipeline
from utils.vector_store import MilvusVectorStore

embedding_client = openai.OpenAI(api_key=OPENAI_API_KEY)

vector_store = MilvusVectorStore(
    host=VECTOR_STORE_HOST,
    port=VECTOR_STORE_PORT,
    collection_name=VECTOR_STORE_COLLECTION,
    timeout=600
)

groups_df = pd.read_csv(GROUPS_CSV_PATH)

def make_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDINGS_MODEL
    )
    return response.data[0].embedding


def node_to_annotation_encoding(node):
    annotation = node["annotation"]
    node_encoding = []
    for k, v in annotation.items():
        if isinstance(v, dict):
            for key in v:
                node_encoding.append(f"{k}_{key} = {v[key]}")
        else:
            node_encoding.append(f"{k} = {v}")
    return ', '.join(node_encoding)


def annotation_subset_to_embedding(annotation_subset: t.List[str]):
    annotation_text = ';'.join(annotation_subset)
    annotation_embedding = make_embedding(annotation_text)
    return annotation_embedding


def pipeline_to_annotation_subsets(pipeline: Pipeline) -> t.List[Pipeline]:
    annotation_subsets = []
    partial_annotation = []
    for node in reversed(pipeline):
        encoded_annotation = node_to_annotation_encoding(node)
        partial_annotation.append(encoded_annotation)
        annotation_subsets.append(copy.deepcopy(partial_annotation))
    return annotation_subsets


def pipeline_to_annotation_payloads(pipeline: Pipeline):
    annotation_subsets = pipeline_to_annotation_subsets(pipeline)
    annotation_payloads = (
        [str(uuid.uuid4()) for _ in range(len(annotation_subsets))],
        [annotation_subset_to_embedding(annotation_subset) for annotation_subset in annotation_subsets]
    )
    return annotation_payloads


def pipeline_to_encoding(annotated_pipeline: Pipeline):
    return ';'.join([node_to_annotation_encoding(node) for node in annotated_pipeline])


def pipeline_to_embedding(annotated_pipeline: Pipeline):
    serialized_pipeline = json.dumps(serialized_pipeline)
    (
        annotation_ids,
        annotation_embeddings
    ) = pipeline_to_annotation_payloads(annotated_pipeline)
    if len(annotation_ids) > 0:
        vector_store.insert(
            ids=annotation_ids,
            documents=[serialized_pipeline for _ in range(len(annotation_ids))],
            embeddings=annotation_embeddings,
        )