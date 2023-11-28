from ...utils.summary_evaluator import SummaryEvaluator


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

        if pipeline.database_name == "sdss":
            if len(dataset.data) > 12:
                data = dataset.data.sample(n=12, random_state=1)
            else:
                data = dataset.data
            for index, galaxy in data[["galaxies.ra", "galaxies.dec"]].iterrows():
                res["data"].append(
                    {
                        "ra": float(galaxy["galaxies.ra"]),
                        "dec": float(galaxy["galaxies.dec"]),
                    }
                )
        else:
            data = dataset.data.sort_values(
                "dm-authors.seniority_original", ascending=False
            )
            if len(dataset.data) > 40:
                data = data.iloc[0:40]
            for index, galaxy in data.iterrows():
                res["data"].append({"author_name": galaxy["dm-authors.author_name"]})

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
