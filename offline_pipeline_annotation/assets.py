import datetime as dt
import pandas as pd
import uuid

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from constants import BUCKET_NAME, GROUPS_PATH, TARGET_SET_PATH
from processes import policy_trainer, target_set_sampler, sample_pipeline_from_models
from processes.utils.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from utils.s3 import (
    pull_info_json, 
    pull_keras_model, 
    push_info_json, 
    push_keras_model, 
    push_pipeline_json
)


@asset
def groups():
    groups_df = pd.read_csv(GROUPS_PATH)

    return MaterializeResult(
        metadata={
            "num_records": len(groups_df),
            "preview": MetadataValue.md(groups_df.head(n=3).to_markdown()),
        }
    )


@asset(deps=[groups], required_resource_keys={"s3"})
def target_sets(context: AssetExecutionContext):
    groups_df = pd.read_csv(GROUPS_PATH)

    target_sets_by_sampling_method = target_set_sampler(groups_df)
    target_sets_df = pd.DataFrame(columns=["id", "item_set", "sampling_method"])
    for sampling_method, sampled_target_sets in target_sets_by_sampling_method.items():
        target_set_records = [
            {
                "id": str(uuid.uuid4()),
                "item_set": list(item_set),
                "sampling_method": sampling_method,
            }
            for item_set in sampled_target_sets
        ]
        sampled_target_sets_df = pd.DataFrame.from_records(target_set_records)
        target_sets_df = pd.concat([target_sets_df, sampled_target_sets_df])

    # Save target_sets.csv locally

    target_sets_df.to_csv(TARGET_SET_PATH, index=False)

    # Load target_sets.csv to S3 bucket

    target_sets_version = f"target_set_{dt.datetime.now().isoformat()}.csv"
    target_sets_key = f"target_sets/{target_sets_version}"

    context.resources.s3.upload_file(TARGET_SET_PATH, BUCKET_NAME, target_sets_key)

    return MaterializeResult(
        metadata={
            "num_records": len(target_sets_df),
            "preview": MetadataValue.md(target_sets_df.head(n=3).to_markdown()),
        }
    )


@asset(deps=[target_sets], required_resource_keys={"s3"})
def policies(context: AssetExecutionContext):
    target_sets_df = pd.read_csv(TARGET_SET_PATH)

    for _, target_set in target_sets_df.iterrows():
        target_set_id = target_set["id"]

        for mode in ["scattered", "concentrated"]:
            context.log.info(
                f"Training policy with TargetSet[id={target_set_id}, mode={mode}]"
            )

            policy_config, (operation_actor, set_actor, critic) = policy_trainer(
                target_set_id, mode
            )
            policy_name = f"policy_{target_set_id}_{mode}"

            push_info_json(
                BUCKET_NAME, 
                policy_name,
                policy_config
            )
            push_keras_model(
                BUCKET_NAME,
                policy_name,
                operation_actor.model,
                model_name="operation_actor",
            )
            push_keras_model(
                BUCKET_NAME,
                policy_name,
                set_actor.model,
                model_name="set_actor",
            )
            push_keras_model(
                BUCKET_NAME,
                policy_name,
                critic.model,
                model_name="critic",
            )

            context.log.info(
                f"Trained policy with TargetSet[id={target_set_id}, mode={mode}]"
            )


@asset(deps=[policies], required_resource_keys={"s3"})
def pipelines(context: AssetExecutionContext):
    target_sets_df = pd.read_csv(TARGET_SET_PATH)

    context.log.info(f"Reading precalculated dataset of pipelines")
    database_pipeline_cache = {}
    database_pipeline_cache["galaxies"] = PipelineWithPrecalculatedSets(
        database_name="sdss",
        initial_collection_names=["galaxies"],
        discrete_categories_count=10,
        min_set_size=10,
        exploration_columns=[
            "galaxies.u",
            "galaxies.g",
            "galaxies.r",
            "galaxies.i",
            "galaxies.z",
            "galaxies.petroRad_r",
            "galaxies.redshift",
        ],
        id_column="galaxies.objID",
    )
    context.log.info(f"Read precalculated dataset of pipelines")

    for _, target_set in target_sets_df.iterrows():
        target_set_id = target_set["id"]

        for mode in ["scattered", "concentrated"]:
            context.log.info(
                f"Generating pipeline using policy trained on with TargetSet[id={target_set_id}, mode={mode}]"
            )

            policy_name = f"policy_{target_set_id}_{mode}"

            models = {
                "set": pull_keras_model(
                    BUCKET_NAME,
                    policy_name,
                    model_name="set_actor",
                ),
                "operation": pull_keras_model(
                    BUCKET_NAME,
                    policy_name,
                    model_name="operation_actor",
                ),
                "set_op_counters": None,
            }
            info = pull_info_json(BUCKET_NAME, policy_name)

            pipeline_name = f"pipeline_{target_set_id}_{mode}"
            pipeline = sample_pipeline_from_models(
                context.log, 
                models, 
                database_pipeline_cache, 
                info
            )

            push_pipeline_json(
                BUCKET_NAME,
                pipeline_name,
                pipeline
            )

            context.log.info(
                f"Generated pipeline using policy trained on TargetSet[id={target_set_id}, mode={mode}]"
            )
