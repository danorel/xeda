import pandas as pd

from dagster import (
    AssetExecutionContext,
    asset,
)

from constants import (
    AWS_S3_BUCKET_NAME
)
from utils.s3 import (
    pull_pipeline_json
)
from ..solid import pipeline_to_embedding
from ..resources import S3FSResource


@asset(io_manager_key="s3_io_manager")
def generate_embeddings(
    context: AssetExecutionContext,
    s3fs: S3FSResource,
    annotated_pipelines: pd.DataFrame,
):
    for i, annotated_pipeline in annotated_pipelines.iterrows():
        context.log.info(f"{annotated_pipeline['pipeline_name']} embedding creation has been started")

        annotated_pipeline = pull_pipeline_json(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            pipeline_folder="annotated_pipelines",
            pipeline_name=annotated_pipeline["pipeline_name"],
        )
        pipeline_to_embedding(annotated_pipeline)

        context.log.info(f"Embedding creation has finished")
