import pandas as pd
import uuid

from dagster import (
    AssetExecutionContext,
    MetadataValue,
    Output,
    asset,
)

from constants import (
    AWS_S3_BUCKET_NAME,
    DATA_NAME,
    DATASET,
    GROUPS_CSV_PATH
)
from ..solid.utils.pipelines.pipeline_precalculated_sets import (
    PipelineWithPrecalculatedSets,
)
from ..resources import S3FSResource
from ..solid import (
    annotate_pipeline,
    generate_mean_vectors,
    policy_trainer,
    target_set_sampler,
    sample_pipeline_from_models,
)
from utils.s3 import (
    pull_info_json,
    pull_keras_model,
    pull_pipeline_json,
    push_info_json,
    push_keras_model,
    push_pipeline_json,
)


@asset(io_manager_key="s3_io_manager")
def datasets(context: AssetExecutionContext):
    cache = {}
    cache[DATA_NAME] = PipelineWithPrecalculatedSets(
        database_name=DATASET,
        initial_collection_names=[DATA_NAME],
        discrete_categories_count=10,
        min_set_size=10,
        exploration_columns=[
            f"{DATA_NAME}.u",
            f"{DATA_NAME}.g",
            f"{DATA_NAME}.r",
            f"{DATA_NAME}.i",
            f"{DATA_NAME}.z",
            f"{DATA_NAME}.petroRad_r",
            f"{DATA_NAME}.redshift",
        ],
        id_column=f"{DATA_NAME}.objID",
    )
    yield Output(cache)


@asset()
def mean_vectors(datasets: dict):
    pipeline = datasets[DATA_NAME]
    generate_mean_vectors(pipeline)


@asset(io_manager_key="s3_io_manager")
def groups(context: AssetExecutionContext):
    df = pd.read_csv(GROUPS_CSV_PATH)

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)


@asset(io_manager_key="s3_io_manager")
def target_sets(context: AssetExecutionContext, groups: pd.DataFrame):
    df = pd.DataFrame(columns=["id", "item_set", "sampling_method"])

    for sampling_method, sampled_item_sets in target_set_sampler(groups).items():
        sampled_df = pd.DataFrame.from_records(
            [
                {
                    "id": str(uuid.uuid4()),
                    "items": list(sampled_item_set),
                    "sampling_method": sampling_method,
                }
                for sampled_item_set in sampled_item_sets
            ]
        )
        df = pd.concat([df, sampled_df])

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)


@asset(io_manager_key="s3_io_manager")
def training_configs(context: AssetExecutionContext, target_sets: pd.DataFrame):
    df = pd.DataFrame(
        columns=[
            "target_set_id",
            "target_set_items",
            "target_set_sampling_method",
            "policy_mode",
        ]
    )

    for _, target_set in target_sets.iterrows():
        for policy_mode in ["scattered", "concentrated"]:
            df = pd.concat([
                df,
                pd.DataFrame([
                    {
                        "target_set_id": target_set["id"],
                        "target_set_items": target_set["items"],
                        "target_set_sampling_method": target_set["sampling_method"],
                        "policy_mode": policy_mode,
                    },
                ])
            ], ignore_index=True)
            context.log.info(
                f"Populated TargetSet[id={target_set['id']}] with mode={policy_mode}"
            )

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)


@asset(io_manager_key="s3_io_manager")
def policies(
    context: AssetExecutionContext, s3fs: S3FSResource, training_configs: pd.DataFrame
):
    df = pd.DataFrame(columns=["target_set_id", "policy_name", "policy_mode"])
    for _, training_config in training_configs.iterrows():
        policy_config, (operation_actor, set_actor, critic) = policy_trainer(
            training_config["target_set_id"],
            training_config["target_set_items"],
            training_config["policy_mode"],
        )
        policy_name = f"policy_{training_config['target_set_id']}_{training_config['policy_mode']}"

        push_info_json(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            policy_name=policy_name,
            policy_config=policy_config,
        )

        push_keras_model(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            policy_name=policy_name,
            model=operation_actor.model,
            model_name="operation_actor",
        )
        push_keras_model(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            policy_name=policy_name,
            model=set_actor.model,
            model_name="set_actor",
        )
        push_keras_model(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            policy_name=policy_name,
            model=critic.model,
            model_name="critic",
        )

        df = pd.concat([
            df,
            pd.DataFrame([
                {
                    "target_set_id": training_config["target_set_id"],
                    "policy_mode": training_config["policy_mode"],
                    "policy_name": policy_name,
                },
            ])
        ], ignore_index=True)

        context.log.info(
            f"Trained policy with TargetSet[id={training_config['target_set_id']}] and mode={training_config['policy_mode']}]"
        )

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)


@asset(io_manager_key="s3_io_manager")
def pipelines(
    context: AssetExecutionContext,
    s3fs: S3FSResource,
    policies: pd.DataFrame,
    datasets: dict,
    mean_vectors
):
    df = pd.DataFrame(
        columns=["pipeline_name", "policy_name", "policy_mode", "target_set_id"]
    )

    for _, policy in policies.iterrows():
        models = {
            "set": pull_keras_model(
                s3fs=s3fs,
                bucket_name=AWS_S3_BUCKET_NAME,
                policy_name=policy["policy_name"],
                model_name="set_actor",
            ),
            "operation": pull_keras_model(
                s3fs=s3fs,
                bucket_name=AWS_S3_BUCKET_NAME,
                policy_name=policy["policy_name"],
                model_name="operation_actor",
            ),
            "set_op_counters": None,
        }
        info = pull_info_json(
            s3fs=s3fs, bucket_name=AWS_S3_BUCKET_NAME, policy_name=policy["policy_name"]
        )

        raw_pipeline = sample_pipeline_from_models(
            models,
            datasets,
            info,
            logger=context.log,
        )
        raw_pipeline_name = (
            f"raw-pipeline_{policy['target_set_id']}_{policy['policy_mode']}"
        )

        push_pipeline_json(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            pipeline_folder="raw_pipelines",
            pipeline_name=raw_pipeline_name,
            pipeline=raw_pipeline,
        )

        df = pd.concat([
            df,
            pd.DataFrame([
                {
                    "pipeline_name": raw_pipeline_name,
                    "policy_name": policy["policy_name"],
                    "policy_mode": policy["policy_mode"],
                    "target_set_id": policy["target_set_id"],
                },
            ])
        ], ignore_index=True)

        context.log.info(
            f"Generated pipeline using policy trained on TargetSet[id={policy['target_set_id']}] and mode={policy['policy_mode']}"
        )

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)


@asset(io_manager_key="s3_io_manager")
def annotated_pipelines(
    context: AssetExecutionContext,
    s3fs: S3FSResource,
    groups: pd.DataFrame,
    pipelines: pd.DataFrame,
):
    df = pd.DataFrame(
        columns=["pipeline_name", "policy_name", "policy_mode", "target_set_id"]
    )

    for i, pipeline in pipelines.iterrows():
        context.log.info(f"{pipeline['pipeline_name']} annotation has been started")

        raw_pipeline = pull_pipeline_json(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            pipeline_folder="raw_pipelines",
            pipeline_name=pipeline["pipeline_name"],
        )

        annotated_pipeline = annotate_pipeline(groups, raw_pipeline, logger=context.log)
        annotated_pipeline_name = (
            f"annotated-pipeline_{pipeline['target_set_id']}_{pipeline['policy_mode']}"
        )

        push_pipeline_json(
            s3fs=s3fs,
            bucket_name=AWS_S3_BUCKET_NAME,
            pipeline_folder="annotated_pipelines",
            pipeline_name=annotated_pipeline_name,
            pipeline=annotated_pipeline,
        )

        df = pd.concat([
            df,
            pd.DataFrame([
                {
                    "pipeline_name": annotated_pipeline_name,
                    "policy_name": pipeline["policy_name"],
                    "policy_mode": pipeline["policy_mode"],
                    "target_set_id": pipeline["target_set_id"],
                },
            ])
        ], ignore_index=True)

        context.log.info(f"{pipeline['pipeline_name']} annotation has been finished")

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head(n=3).to_markdown()),
        }
    )

    yield Output(df)
