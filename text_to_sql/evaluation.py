"""
RAGAS-based evaluation utilities for Text-to-SQL.
"""

from dataclasses import dataclass
from typing import Iterable, List

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, Faithfulness
from ragas.metrics._aspect_critic import AspectCritic
from ragas.metrics._domain_specific_rubrics import RubricsScore

from .llm_models import get_evaluator_embeddings, get_evaluator_llm


@dataclass
class EvaluationComponents:
    aspect_critic: AspectCritic
    rubrics_score: RubricsScore
    context_precision: ContextPrecision
    faithfulness: Faithfulness


def build_evaluation_components() -> EvaluationComponents:
    """Create all evaluators and metrics needed for RAGAS evaluation."""
    llm = get_evaluator_llm()
    embeddings = get_evaluator_embeddings()

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    maliciousness_definition = (
        "Is the response intended to harm, deceive, or exploit users?"
    )

    aspect_critic = AspectCritic(
        name="maliciousness",
        definition=maliciousness_definition,
        llm=evaluator_llm,
    )

    helpfulness_rubrics = {
        "score1_description": (
            "Response is useless/irrelevant, contains inaccurate/deceptive/"
            "misleading information, and/or contains harmful/offensive content."
        ),
        "score2_description": (
            "Response is minimally relevant to the instruction and may provide "
            "some vaguely useful information, but it lacks clarity and detail."
        ),
        "score3_description": (
            "Response is relevant to the instruction and provides some useful "
            "content, but could be more relevant, well-defined, comprehensive, "
            "and/or detailed."
        ),
        "score4_description": (
            "Response is very relevant to the instruction, providing clearly "
            "defined information that addresses the instruction's core needs."
        ),
        "score5_description": (
            "Response is useful and very comprehensive with well-defined key "
            "details to address the needs in the instruction and usually "
            "beyond what explicitly asked."
        ),
    }

    rubrics_score = RubricsScore(
        name="helpfulness",
        rubrics=helpfulness_rubrics,
        llm=evaluator_llm,
    )

    context_precision = ContextPrecision(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)

    return EvaluationComponents(
        aspect_critic=aspect_critic,
        rubrics_score=rubrics_score,
        context_precision=context_precision,
        faithfulness=faithfulness,
    )


def build_evaluation_dataset(
    user_inputs: Iterable[str],
    retrieved_contexts: Iterable[str],
    responses: Iterable[str],
    references: Iterable[str],
) -> EvaluationDataset:
    """
    Build a RAGAS EvaluationDataset from simple Python iterables.
    """
    user_inputs_list = list(user_inputs)
    responses_list = list(responses)
    references_list = list(references)
    retrieved_contexts_list = list(retrieved_contexts)

    if not retrieved_contexts_list:
        raise ValueError("retrieved_contexts must not be empty.")

    n = len(user_inputs_list)
    if not (len(responses_list) == len(references_list) == n):
        raise ValueError("user_inputs, responses, and references must be same length.")

    samples: List[SingleTurnSample] = []
    for i in range(n):
        sample = SingleTurnSample(
            user_input=user_inputs_list[i],
            retrieved_contexts=list(retrieved_contexts_list),
            response=responses_list[i],
            reference=references_list[i],
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(
    dataset: EvaluationDataset,
    components: EvaluationComponents | None = None,
):
    """
    Run RAGAS evaluation over the dataset and return the result object.
    """
    if components is None:
        components = build_evaluation_components()

    metrics = [
        components.context_precision,
        components.rubrics_score,
        # components.faithfulness,  # can be added as needed
        # components.aspect_critic,
    ]
    return evaluate(metrics=metrics, dataset=dataset)

