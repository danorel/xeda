from dagster import (
    AssetSelection,
    Definitions,
    EnvVar,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)
from dagster_aws.s3.io_manager import s3_pickle_io_manager
from dagster_aws.s3.resources import S3Resource

from . import assets

all_assets = load_assets_from_modules([assets])

offline_pipeline_annotation_schedule = ScheduleDefinition(
    job=define_asset_job("offline_pipeline_annotation", selection=AssetSelection.all()),
    cron_schedule="0 * * * *",
)

defs = Definitions(
    assets=all_assets,
    schedules=[offline_pipeline_annotation_schedule],
    resources={
        "s3_io_manager": s3_pickle_io_manager.configured(
            {"s3_bucket": EnvVar("AWS_S3_BUCKET")}
        ),
        "s3": S3Resource(
            aws_access_key_id=EnvVar("AWS_ACCESS_KEY"),
            aws_secret_access_key=EnvVar("AWS_SECRET_KEY"),
            region_name=EnvVar("AWS_REGION"),
        ),
    },
)
