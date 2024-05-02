import heapq
import typing as t
import statistics

from collections import defaultdict, Counter 

from typings.pipeline import Pipeline
from .utils import make_explanation_details, results_to_pipelines, pipeline_to_embedding, vector_store


def payload_from_search_result(pipeline: Pipeline, current_step: int):
    payload = {}

    if current_step >= len(pipeline):
        current_node = pipeline[-1]
    else:
        current_node = pipeline[current_step]
    current_annotation = current_node.get("annotation", {})

    payload['total_length'] = current_annotation.get("total_length")
    for dimension_name, dimension_counter in current_annotation.get('remaining_dimensions', {}).items():
        payload[f'remaining_dimensions_{dimension_name}'] = int(dimension_counter) 
    for operator_name, operator_counter in current_annotation.get('remaining_operators', {}).items():
        payload[f'remaining_operators_{operator_name}'] = int(operator_counter) 
    payload['familiarity'] = current_annotation.get("familiarity")

    return payload


def number_of_sim_pipelines(payloads):
    return len(payloads)


def length(payloads):
    total_length_values = [p.get('total_length', 0) for p in payloads]
    total_length_counter = Counter(total_length_values)
    total_length_percentages = {key: count / len(total_length_counter) * 100 for key, count in total_length_counter.items()}
    max_length = max(total_length_percentages, key=lambda k: total_length_percentages[k])
    return int(max_length)


def dist_operators(payloads):
    operators = ['remaining_operators_by_neighbors', 'remaining_operators_by_superset', 'remaining_operators_by_distribution', 'remaining_operators_by_facet']
    operator_values = defaultdict(float)

    for operator in operators:
        operator_values[operator] += sum(p.get(operator, 0.0) for p in payloads)

    total_count = sum(operator_values.values()) + 0.000001
    percentages_length = {key: value / total_count * 100 for key, value in operator_values.items()}

    top_two_values = heapq.nlargest(4, percentages_length.values())
    top_two_keys = [(key, value) for key, value in percentages_length.items() if value in top_two_values]

    return top_two_keys


def dimensions(payloads):
    dimensions = ['remaining_dimensions_u', 'remaining_dimensions_g', 'remaining_dimensions_r', 'remaining_dimensions_i', 'remaining_dimensions_z', 'remaining_dimensions_petroRad_r', 'remaining_dimensions_redshift']
    dimensions_values = defaultdict(float)

    for dimension in dimensions:
        dimensions_values[dimension] += sum(p.get(dimension, 0.0) for p in payloads)

    total_count = sum(dimensions_values.values()) + 0.000001
    percentages_length = {key: value / total_count * 100 for key, value in dimensions_values.items()}

    top_two_values = heapq.nlargest(2, percentages_length.values())
    top_two_keys = [(key, value) for key, value in percentages_length.items() if value in top_two_values]

    return top_two_keys


def familiarity(payloads):
    familiarity_values = [p.get('familiarity', 0.0) for p in payloads]
    median_familiarity = statistics.median(familiarity_values)
    return median_familiarity


def generate_guidance(payloads, step):
    total_length = length(payloads)
    operator = dist_operators(payloads)
    dimension = dimensions(payloads)
    median_familiarity = familiarity(payloads)
    k = number_of_sim_pipelines(payloads)
    steps = total_length - step

    return f'On average {steps} step/s, you will reach a scattered/concentrated set with an expected final familiarity of {median_familiarity}. ' \
           f'You are more likely to get there by focusing on the {operator[0][0][20:]} and {operator[1][0][20:]} operators and on {dimension[0][0][21:]} and {dimension[1][0][21:]} dimensions. ' \
           f'You will probably finish with total length of {total_length}. ' \
           f'You get this guidance because: in the {k} similar pipelines the following distribution of operator {operator[0][0][20:]} is {round(operator[0][1], 2)}, {operator[1][0][20:]} is {round(operator[1][1], 2)}, {operator[2][0][20:]} is {round(operator[2][1], 2)}, {operator[3][0][20:]} is {round(operator[3][1], 2)}; ' \
           f'the distribution of dimension {dimension[0][0][21:]} is {round(dimension[0][1], 2)}, {dimension[1][0][21:]} is {round(dimension[1][1], 2)}.'


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
    if not len(neighbouring_pipelines):
        raise ValueError("Not able to provide explanation: lacking similar pipelines")
    natural_language_explanation, explanation_details = (
        make_natural_language_explanation(neighbouring_pipelines, current_step=len(partial_pipeline)),
        make_explanation_details(neighbouring_pipelines)
    )
    return natural_language_explanation, explanation_details