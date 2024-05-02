from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

import pathlib
import s3fs
import typing as t
import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

import constants.aws as aws_secrets
import constants.dataset as dataset_secrets
import constants.dataset as dataset_secrets 
import pipeline.solid.pipeline_sampler.operators as operators

from typings.pipeline import ExplanationRequestData, OperatorRequestData
from pipeline.solid.utils.summary_evaluator import SummaryEvaluator
from pipeline.solid.utils.target_set_generator import TargetSetGenerator
from pipeline.solid.utils.pipelines import PipelineWithPrecalculatedSets
from pipeline.solid.utils.greedy_summarizer import GreedySummarizer
from pipeline.solid.utils.model_manager import ModelManager
from pipeline.solid.utils.pipelines import PipelineWithPrecalculatedSets
from utils.s3 import pull_keras_model
from web.explanation.statistical import explain 


app_dir = pathlib.Path.cwd()
client_dir = app_dir / "web" / "client"

app = FastAPI(
    title="XEDA API",
    description="XEDA API providing access to the CNRS pipelines operators",
    version="1.0.0",
)

fs = s3fs.S3FileSystem(
    key=aws_secrets.AWS_ACCESS_KEY_ID,
    secret=aws_secrets.AWS_SECRET_ACCESS_KEY,
    endpoint_url=aws_secrets.AWS_S3_ENDPOINT_URL,
    use_ssl=aws_secrets.AWS_S3_USE_SSL,
    client_kwargs={
        "region_name": aws_secrets.AWS_S3_REGION_NAME,
    },
)

database_pipeline_cache = {}
database_pipeline_cache[dataset_secrets.DATA_NAME] = PipelineWithPrecalculatedSets(
    database_name=dataset_secrets.DATASET,
    initial_collection_names=[dataset_secrets.DATA_NAME],
    discrete_categories_count=10,
    min_set_size=10,
    exploration_columns=[
        f"{dataset_secrets.DATA_NAME}.u",
        f"{dataset_secrets.DATA_NAME}.g",
        f"{dataset_secrets.DATA_NAME}.r",
        f"{dataset_secrets.DATA_NAME}.i",
        f"{dataset_secrets.DATA_NAME}.z",
        f"{dataset_secrets.DATA_NAME}.petroRad_r",
        f"{dataset_secrets.DATA_NAME}.redshift",
    ],
    id_column=f"{dataset_secrets.DATA_NAME}.objID",
)

model_manager = ModelManager(
    pipeline=database_pipeline_cache[dataset_secrets.DATA_NAME],
    models={
        "set": pull_keras_model(
            s3fs=fs,
            bucket_name=aws_secrets.AWS_S3_BUCKET_NAME,
            policy_name=dataset_secrets.UNIVERSAL_POLICY_NAME,
            model_name="set_actor",
        ),
        "operation": pull_keras_model(
            s3fs=fs,
            bucket_name=aws_secrets.AWS_S3_BUCKET_NAME,
            policy_name=dataset_secrets.UNIVERSAL_POLICY_NAME,
            model_name="operation_actor",
        ),
        "set_op_counters": None,
    }
)

@app.get("/")
async def index():
    return FileResponse(client_dir / "index.html")


@app.get("/dora-summaries")
async def dora_summaries():
    return FileResponse(client_dir / "redirect.html")


def get_items_sets(
    sets,
    pipeline,
    get_scores,
    get_predicted_scores,
    galaxy_class_scores,
    seen_sets=None,
    previous_dataset_ids=None,
    utility_weights=None,
    previous_operations=None,
    decreasing_gamma=False,
):
    results = {"sets": [], "previous_operations": previous_operations}
    evaluator = SummaryEvaluator(pipeline, galaxy_class_scores=galaxy_class_scores)
    evaluator.evaluate_sets(sets)
    results.update(evaluator.get_evaluation_scores())
    results["galaxy_class_scores"] = evaluator.galaxy_class_scores
    for dataset in sets:
        res = {
            "length": len(dataset.data),
            "id": int(dataset.set_id) if dataset.set_id != None else -1,
            "data": [],
            "predicate": [],
        }
        for predicate in dataset.predicate.components:
            res["predicate"].append(
                {"dimension": predicate.attribute, "value": str(predicate.value)}
            )
        if len(dataset.data) > 12:
            data = dataset.data.sample(n=12, random_state=1)
        else:
            data = dataset.data
        for index, galaxy in data[[f"{dataset_secrets.DATA_NAME}.ra", f"{dataset_secrets.DATA_NAME}.dec"]].iterrows():
            res["data"].append(
                {
                    "ra": float(galaxy[f"{dataset_secrets.DATA_NAME}.ra"]),
                    "dec": float(galaxy[f"{dataset_secrets.DATA_NAME}.dec"]),
                }
            )
        results["sets"].append(res)
    if get_scores:
        (
            summary_uniformity_score,
            sets_uniformity_scores,
        ) = pipeline.utility_manager.get_uniformity_scores(sets, pipeline)
        results["distance"] = pipeline.utility_manager.get_min_distance(sets, pipeline)
        results["uniformity"] = summary_uniformity_score

        for index, score in enumerate(sets_uniformity_scores):
            results["sets"][index]["uniformity"] = score
        (
            summary_novelty_score,
            seen_sets,
            new_utility_weights,
        ) = pipeline.utility_manager.get_novelty_scores_and_utility_weights(
            sets,
            seen_sets,
            pipeline,
            utility_weights=utility_weights,
            decreasing_gamma=decreasing_gamma,
        )
        results["novelty"] = summary_novelty_score
        results["utility"] = pipeline.utility_manager.compute_utility(
            utility_weights,
            results["uniformity"],
            results["distance"],
            results["novelty"],
        )
        results["utility_weights"] = utility_weights = new_utility_weights
        results["seen_sets"] = seen_sets
    else:
        results["uniformity"] = None
        results["novelty"] = None

        for dataset in results["sets"]:
            dataset["uniformity"] = None
            dataset["novelty"] = None
        seen_sets = seen_sets | set(map(lambda x: int(x.set_id), sets))
        results["seen_sets"] = seen_sets
    if get_predicted_scores:
        results["predictedScores"] = pipeline.utility_manager.get_future_scores(
            sets,
            pipeline,
            seen_sets,
            previous_dataset_ids,
            utility_weights,
            previous_operations,
        )
    else:
        results["predictedScores"] = {}

    return results


@app.get("/swapSum")
async def swap_sum(
    dataset_to_explore: str,
    min_set_size: int,
    min_uniformity_target: float,
    result_set_count: int,
):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
        dataset_to_explore
    ]
    greedy_summarizer = GreedySummarizer(pipeline)
    result_sets = greedy_summarizer.get_summary(
        min_set_size, min_uniformity_target, result_set_count
    )
    result = get_items_sets(
        result_sets,
        pipeline,
        True,
        False,
        None,
        seen_sets=set(),
        previous_dataset_ids=set(),
        utility_weights=[0.5, 0.5, 0],
        previous_operations=[],
    )

    return result


@app.put("/get_predicted_scores")
async def get_predicted_scores(request_data: OperatorRequestData):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
        request_data.dataset_to_explore
    ]
    datasets = pipeline.get_groups_as_datasets(request_data.dataset_ids)

    result = get_items_sets(
        datasets,
        pipeline,
        True,
        True,
        None,
        seen_sets=set(request_data.seen_sets),
        previous_dataset_ids=set(request_data.dataset_ids),
        utility_weights=request_data.utility_weights,
        previous_operations=request_data.previous_operations,
    )
    return result


@app.put(
    "/explanation", 
    description="", 
    tags=["explanation"]
)
async def explanation(explanation_request_data: ExplanationRequestData):
    partial_pipeline = explanation_request_data.partial_pipeline
    explanation_text, explanation_details = "", []
    if len(partial_pipeline) >= 3:
        try:
            explanation_text, explanation_details = explain(partial_pipeline)
        except Exception as e:
            explanation_text = str(e)
    return {
        "explanation_text": explanation_text,
        "explanation_details": explanation_details
    }


@app.put(
    "/operators/by_facet-g",
    description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
    tags=["operators"],
)
async def by_facet_g(request_data: OperatorRequestData):
    return operators.by_facet(
        database_pipeline_cache,
        model_manager,
        request_data
    )


@app.put(
    "/operators/by_superset-g",
    description="Returns the smallest set completely overget-dataset-informationlapping with the input set",
    tags=["operators"],
)
async def by_superset_g(request_data: OperatorRequestData):
    return operators.by_superset(
        database_pipeline_cache,
        model_manager,
        request_data
    )


@app.put("/operators/by_neighbors-g", description="", tags=["operators"])
async def by_neighbors_g(request_data: OperatorRequestData):
    return operators.by_neighbors(
        database_pipeline_cache,
        model_manager,
        request_data
    )


@app.put("/operators/by_distribution-g", description="", tags=["operators"])
async def by_distribution_g(request_data: OperatorRequestData):
    return operators.by_distribution(
        database_pipeline_cache,
        model_manager,
        request_data
    )


@app.get("/app/get-dataset-information", description="", tags=["info"])
async def get_dataset_information(dataset: str):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[dataset]
    return {
        "dimensions": list(pipeline.ordered_dimensions.keys()),
        "ordered_dimensions": pipeline.ordered_dimensions,
        "length": len(pipeline.initial_collection),
    }


@app.get("/app/get-target-items-and-prediction", description="", tags=["info"])
async def get_target_items_and_prediction(
    target_set: str = None, curiosity_weight: float = None, dataset_ids: t.List[int] = []
):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    target_items = TargetSetGenerator.get_diverse_target_set(
        pipeline.database_name, number_of_samples=100
    )
    items_found_with_ratio = {}
    if len(dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(dataset_ids)
    if curiosity_weight != None:
        prediction_results = model_manager.get_prediction(
            datasets, target_set, curiosity_weight, target_items, items_found_with_ratio
        )
        prediction_results["targetItems"] = list(map(lambda x: str(x), target_items))
    return prediction_results


@app.put("/app/load-model", description="", tags=["info"])
async def load_model(request_data: OperatorRequestData):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if len(request_data.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(request_data.dataset_ids)

    prediction_results = model_manager.get_prediction(
        datasets,
        request_data.target_items,
        request_data.found_items_with_ratio,
        request_data.previous_set_states,
        request_data.previous_operation_states,
    )

    return prediction_results


app.mount(
    "/",
    StaticFiles(
        directory=client_dir, 
        html=True
    ),
    name="client"
)


if __name__ == "__main__":
    uvicorn.run("web.server:app", host="127.0.0.1", port=8090, reload=True)