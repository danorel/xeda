import json
import typing as t

from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from typings.pipeline import Pipeline
from utils.vector_store import SearchResult
from web.explanation.utils import make_explanation_details, results_to_scores, results_to_pipelines, pipeline_to_embedding, vector_store

summarization_prompt_template = """Write a concise summary of "{text}". CONCISE SUMMARY:"""
summarization_prompt = PromptTemplate.from_template(summarization_prompt_template)

summarization_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
summarization_chain = LLMChain(llm=summarization_llm, prompt=summarization_prompt)

stuff_chain = StuffDocumentsChain(llm_chain=summarization_chain, document_variable_name="text")

#@Guidance utils: make natural language explanations 

def make_instruction(name, value):
    return f"{name} = {value}" if value else None


def natural_language_from_search_result(search_result: SearchResult):
    pipeline = json.loads(search_result['document'])
    last_node = pipeline[-1]
    # Extract natural language properties
    total_length, operator, checked_dimension, remaining_operators = (
        last_node['annotation']['total_length'],
        last_node['operator'],
        last_node['checkedDimension'],
        last_node['annotation']['remaining_operators']
    )
    # Derive natural language guidance features
    remaining_operators_count = sum(v for v in remaining_operators.values())
    remaining_operators_distribution = ', '.join([f"{operator} = {operator_count / remaining_operators_count}%" for operator, operator_count in remaining_operators.items()])
    # Build natural language query for summarization
    natural_language_instructions = [instruction for instruction in [
        make_instruction(name="most_probable_pipeline_length", value=total_length),
        make_instruction(name="most_probable_operator", value=operator),
        make_instruction(name="reachable_attribute_by_operator", value=checked_dimension),
        make_instruction(name="operator_probability_distribution", value=remaining_operators_distribution),
    ] if instruction is not None]
    natural_language = ", ".join(natural_language_instructions)
    natural_language_document = Document(page_content=natural_language)
    return natural_language_document


def make_natural_language_explanation(neighbouring_results: t.List[SearchResult]):
    natural_language_explanation = stuff_chain.run((
        natural_language_from_search_result(neighbouring_result)
        for neighbouring_result in neighbouring_results
    ))
    return natural_language_explanation


def explain(partial_pipeline: Pipeline, k: int = 3) -> t.Tuple[str, str]:
    partial_pipeline_embedding = pipeline_to_embedding(partial_pipeline)
    neighbouring_results = vector_store.search(partial_pipeline_embedding, k)
    if not len(neighbouring_results):
        raise ValueError("Not found similar documents in vector storage")
    neighbouring_pipelines = results_to_pipelines(neighbouring_results)
    neighbouring_scores = results_to_scores(neighbouring_results)
    if not len(neighbouring_pipelines):
        raise ValueError("Not able to provide explanation: lacking similar pipelines")
    current_step = len(partial_pipeline) - 1
    natural_language_explanation, explanation_details = (
        make_natural_language_explanation(neighbouring_pipelines),
        make_explanation_details(neighbouring_pipelines, neighbouring_scores, current_step)
    )
    return natural_language_explanation, explanation_details