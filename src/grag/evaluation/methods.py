"""List of evaluation methods"""

from typing import List, Tuple, Union
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    NonLLMContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMStringSimilarity,
    ResponseRelevancy,
    RougeScore,
    ToolCallAccuracy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import EvaluationResult, SingleTurnSample


class StringSimilarity(NonLLMStringSimilarity):
    """Check tool call args string similarity"""

    threshold: float = 0.5

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        assert isinstance(reference, str), "Expecting a string"
        assert isinstance(response, str), "Expecting a string"
        if (
            1
            - self.distance_measure_map[self.distance_measure].normalized_distance(
                reference, response
            )
            > self.threshold
        ):
            return 1.0
        return 0.0


class SimpleContextPrecision(NonLLMContextPrecisionWithReference):
    """Simple context precision metric"""

    name: str = "simple_context_precision"
    threshold: float = 0.5

    def _calculate_average_precision(self, verdict_list: List[int]) -> float:
        """
        Menghitung precision biasa dari daftar biner relevansi.
        """
        if not verdict_list:
            return 0.0  # Handle empty list case

        relevant_count = sum(verdict_list)
        total_count = len(verdict_list)

        # Add a small epsilon to denominator to avoid division by zero,
        # although len(verdict_list) should not be zero if list is not empty
        # return relevant_count / (total_count + 1e-10) # Alternative with epsilon
        return relevant_count / total_count


def evaluate_retriever(
    evaluation_dataset: EvaluationDataset, *, experiment_name: str
) -> EvaluationResult:
    """
    TODO: Docstring
    """
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            SimpleContextPrecision(name="precision"),
            NonLLMContextRecall(name="recall"),
        ],
        experiment_name=experiment_name,
    )


def evaluate_text_generation(
    evaluation_dataset: EvaluationDataset,
    *,
    llm_model: BaseLanguageModel,
    embedding_model: Embeddings,
    experiment_name: str
) -> EvaluationResult:
    """
    TODO: Docstring
    """
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            RougeScore(rouge_type="rouge1", mode="precision"),
            RougeScore(rouge_type="rouge1", mode="recall"),
            RougeScore(rouge_type="rouge1", mode="fmeasure"),
            RougeScore(rouge_type="rougeL", mode="precision"),
            RougeScore(rouge_type="rougeL", mode="recall"),
            RougeScore(rouge_type="rougeL", mode="fmeasure"),
            ResponseRelevancy(),
            Faithfulness(),
        ],
        llm=LangchainLLMWrapper(llm_model),
        embeddings=LangchainEmbeddingsWrapper(embedding_model),
        experiment_name=experiment_name,
    )


def evaluate_end_to_end(
    evaluation_dataset: Tuple[EvaluationDataset, Union[EvaluationDataset, None]],
    *,
    llm_model: BaseLanguageModel,
    embedding_model: Embeddings,
    experiment_name: str
) -> Tuple[EvaluationResult, Union[EvaluationResult, None]]:
    """
    TODO: Docstring
    """
    single_turn_evaluation_dataset, multi_turn_evaluation_dataset = evaluation_dataset

    # General metric evaluation (need single turn evaluation dataset)
    single_turn_evaluation_result = evaluate(
        dataset=single_turn_evaluation_dataset,
        metrics=[
            NonLLMContextPrecisionWithReference(),
            NonLLMContextRecall(),
            RougeScore(rouge_type="rouge1", mode="precision"),
            RougeScore(rouge_type="rouge1", mode="recall"),
            RougeScore(rouge_type="rouge1", mode="fmeasure"),
            RougeScore(rouge_type="rougeL", mode="precision"),
            RougeScore(rouge_type="rougeL", mode="recall"),
            RougeScore(rouge_type="rougeL", mode="fmeasure"),
            ResponseRelevancy(strictness=1),
            Faithfulness(),
        ],
        llm=LangchainLLMWrapper(llm_model),
        embeddings=LangchainEmbeddingsWrapper(embedding_model),
        experiment_name=experiment_name + "_single_turn",
    )

    # ToolCallAccuracy (need multi turn evaluation dataset)
    if multi_turn_evaluation_dataset:
        multi_turn_evaluation_result = evaluate(
            dataset=multi_turn_evaluation_dataset,
            metrics=[ToolCallAccuracy(arg_comparison_metric=StringSimilarity())],
            experiment_name=experiment_name + "_multi_turn",
        )
    else:
        multi_turn_evaluation_result = None

    return (single_turn_evaluation_result, multi_turn_evaluation_result)
