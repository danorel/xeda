import heapq
import typing as t
import statistics

from collections import defaultdict, Counter 

from typings.pipeline import Pipeline
from web.explanation.utils import make_explanation_details, results_to_pipelines, results_to_scores, pipeline_to_embedding, vector_store


def payload_from_search_result(pipeline: Pipeline, current_step: int):
    payload = {}

    if current_step < len(pipeline):
        current_node = pipeline[current_step]
    else:
        current_node = pipeline[-1]
    current_annotation = current_node.get("annotation", {})

    payload['total_length'] = current_annotation.get("total_length")
    for dimension_name, dimension_counter in current_annotation.get('remaining_dimensions', {}).items():
        payload[f'remaining_dimensions_{dimension_name}'] = int(dimension_counter) 
    for operator_name, operator_counter in current_annotation.get('remaining_operators', {}).items():
        payload[f'remaining_operators_{operator_name}'] = int(operator_counter) 
    payload['familiarity'] = current_annotation.get("familiarity")
    payload['target_type'] = current_annotation.get("target_type")

    return payload


def number_of_sim_pipelines(payloads):
    return len(payloads)


def length(payloads):
    total_length_values = [p.get('total_length', 0) for p in payloads]
    total_length_counter = Counter(total_length_values)
    total_length_percentages = {key: count / len(total_length_counter) * 100 for key, count in total_length_counter.items()}
    max_length = max(total_length_percentages, key=lambda k: total_length_percentages[k])
    return int(max_length)


def dist_operators(payloads, top_k: int = 2):
    operators = ['remaining_operators_by_neighbors', 'remaining_operators_by_superset', 'remaining_operators_by_distribution', 'remaining_operators_by_facet']
    operator_values = defaultdict(float)

    for operator in operators:
        operator_values[operator] += sum(p.get(operator, 0.0) for p in payloads)

    total_count = sum(operator_values.values()) + 0.000001
    percentages_length = {key: value / total_count * 100 for key, value in operator_values.items()}

    top_k_values = heapq.nlargest(top_k, percentages_length.values())
    top_k_keys = [(key[20:], value) for key, value in percentages_length.items() if value in top_k_values]

    return top_k_keys


def dist_dimensions(payloads, top_k: int = 2):
    dimensions = ['remaining_dimensions_u', 'remaining_dimensions_g', 'remaining_dimensions_r', 'remaining_dimensions_i', 'remaining_dimensions_z', 'remaining_dimensions_petroRad_r', 'remaining_dimensions_redshift']
    dimensions_values = defaultdict(float)

    for dimension in dimensions:
        dimensions_values[dimension] += sum(p.get(dimension, 0.0) for p in payloads)

    total_count = sum(dimensions_values.values()) + 0.000001
    percentages_length = {key: value / total_count * 100 for key, value in dimensions_values.items()}

    top_k_values = heapq.nlargest(top_k, percentages_length.values())
    top_k_keys = [(key[21:], value) for key, value in percentages_length.items() if value in top_k_values]

    return top_k_keys


def familiarity(payloads):
    familiarity_values = [p.get('familiarity', 0.0) for p in payloads]
    median_familiarity = statistics.median(familiarity_values)
    return median_familiarity


def scattered_or_concentrated(payloads, default_type: str = "scattered"):
    type_counts = defaultdict(int)
    for payload in payloads:
        type_value = payload.get('target_type')
        if type_value is None:
            continue
        if type_value in type_counts:
            type_counts[type_value] += 1
        else:
            type_counts[type_value] = 1

    if not len(type_counts):
        return default_type

    try:
        most_common_type = max(type_counts, key=type_counts.get)
        return most_common_type
    except Exception as e:
        print(str(e))

    return default_type


def generate_guidance(payloads, step):
    total_length = length(payloads)
    operator = dist_operators(payloads, top_k=4)
    dimension = dist_dimensions(payloads, top_k=4)
    median_familiarity = familiarity(payloads)
    k = number_of_sim_pipelines(payloads)
    target_type = scattered_or_concentrated(payloads)
    steps = total_length - step

    return f'On average {steps} step/s, you will reach a {target_type} set with an expected final familiarity of {median_familiarity}. ' \
           f'You are more likely to get there by focusing on the {operator[0][0]} and {operator[1][0]} operators and on {dimension[0][0]} and {dimension[1][0]} dimensions. ' \
           f'You will probably finish with total length of {total_length}. ' \
           f'You get this guidance because: in top {k} similar pipelines the following distribution of operator {operator[0][0]} is {round(operator[0][1], 2)}, {operator[1][0]} is {round(operator[1][1], 2)}, {operator[2][0]} is {round(operator[2][1], 2)}, {operator[3][0]} is {round(operator[3][1], 2)}; ' \
           f'the distribution of dimension {dimension[0][0]} is {round(dimension[0][1], 2)}, {dimension[1][0]} is {round(dimension[1][1], 2)}.'


def make_natural_language_explanation(neighbouring_pipelines: t.List[Pipeline], current_step: int):
    payloads = [
        payload_from_search_result(pipeline, current_step)
        for pipeline in neighbouring_pipelines
    ]
    natural_language_explanation = generate_guidance(payloads, current_step)
    return natural_language_explanation


def explain(partial_pipeline: Pipeline, k: int = 5) -> t.Tuple[str, str]:
    partial_pipeline_embedding = pipeline_to_embedding(partial_pipeline)
    neighbouring_results = vector_store.search(partial_pipeline_embedding, k)
    if not len(neighbouring_results):
        raise ValueError("Not found similar pipelines in vector storage")
    neighbouring_pipelines = results_to_pipelines(neighbouring_results)
    neighbouring_scores = results_to_scores(neighbouring_results)
    if not len(neighbouring_pipelines):
        raise ValueError("Not able to provide explanation: lacking similar pipelines")
    current_step = len(partial_pipeline) - 1
    natural_language_explanation, explanation_details = (
        make_natural_language_explanation(neighbouring_pipelines, current_step),
        make_explanation_details(neighbouring_pipelines, neighbouring_scores, current_step)
    )
    return natural_language_explanation, explanation_details