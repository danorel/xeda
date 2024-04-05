from dagster import (
    AssetSelection,
    Definitions,
    EnvVar,
    ScheduleDefinition,
    DefaultScheduleStatus,
    define_asset_job,
    load_assets_from_modules,
)
from dagster_aws.s3 import s3_resource
from dagster_aws.s3.io_manager import s3_pickle_io_manager

from . import assets
from . import resources

all_assets = load_assets_from_modules([assets])

schedule = ScheduleDefinition(
    job=define_asset_job("offline_pipeline_annotation", selection=AssetSelection.all()),
    cron_schedule="*/30 * * * *",
    default_status=DefaultScheduleStatus.RUNNING
)

defs = Definitions(
    assets=all_assets,
    schedules=[schedule],
    resources={
        "s3_io_manager": s3_pickle_io_manager.configured(
            {"s3_bucket": {"env": "AWS_S3_BUCKET_NAME"}}
        ),
        "s3": s3_resource.configured(
            {
                "aws_access_key_id": {"env": "AWS_ACCESS_KEY_ID"},
                "aws_secret_access_key": {"env": "AWS_SECRET_ACCESS_KEY"},
                "region_name": {"env": "AWS_S3_REGION_NAME"},
                "endpoint_url": {"env": "AWS_S3_ENDPOINT_URL"},
            }
        ),
        "s3fs": resources.S3FSResource(
            key=EnvVar("AWS_ACCESS_KEY_ID"),
            secret=EnvVar("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=EnvVar("AWS_S3_ENDPOINT_URL"),
            use_ssl=EnvVar("AWS_S3_USE_SSL"),
            region_name=EnvVar("AWS_S3_REGION_NAME"),
        ),
    },
)
