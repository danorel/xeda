import chromadb
import copy
import uuid

from chromadb.utils import embedding_functions

from constants import (
    OPENAI_API_KEY,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_HOST,
    VECTOR_STORE_PORT
)
from data_types.pipeline import Pipeline

pretrained_embeddings = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

vector_store = chromadb.HttpClient(
    host=VECTOR_STORE_HOST, 
    port=VECTOR_STORE_PORT
)

vector_collection = vector_store.get_or_create_collection(
    name=VECTOR_STORE_COLLECTION, 
    embedding_function=pretrained_embeddings,
    metadata={
        "hnsw:space": "cosine"
    }
)

def pipeline_to_splits(pipeline: Pipeline) -> list[Pipeline]:
    splits = []
    pipeline_encoding = []
    for node in reversed(pipeline):
        annotation = node["annotation"]
        node_encoding = []
        for k, v in annotation.items():
            if isinstance(v, dict):
                for key in v:
                    node_encoding.append(f"{k}_{key} = {v[key]}")
            else:
                node_encoding.append(f"{k} = {v}")
        pipeline_encoding.append(', '.join(node_encoding))
        splits.append(copy.deepcopy(pipeline_encoding))
    return splits


def pipeline_to_embedding(pipeline: Pipeline):
    pipeline_splits = pipeline_to_splits(pipeline)
    pipeline_ids, pipeline_documents = (
        [str(uuid.uuid4()) for i in range(len(pipeline_splits))],
        [';'.join(pipeline_split) for pipeline_split in pipeline_splits]
    )
    vector_collection.add(
        ids=pipeline_ids,
        documents=pipeline_documents
    )
