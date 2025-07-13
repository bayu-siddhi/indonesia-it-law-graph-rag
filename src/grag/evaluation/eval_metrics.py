"""List of evaluation metrics"""

import string
from ast import literal_eval
from typing import Dict, List, Set, Tuple, Union
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import (
    AnswerAccuracy,
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


run_config = RunConfig(timeout=None)


def separate_punctuation_with_spaces(text: str) -> str:
    """
    Replaces characters from string.punctuation with spaces.
    Example: 'word,word' into 'word word'.

    Args:
        text (str): The input string.

    Returns:
        result (str): The string with punctuation replaced by spaces.
    """
    return "".join(" " if c in string.punctuation else c for c in text)


def process_data(data: Union[str, Dict, List]) -> Set[str]:
    """
    Processes data to extract strings.

    It handles dictionaries, lists, and other data types.

    Literal evaluation is attempted for non-string types.

    Args:
        data (Union[str, Dict, List]): The data to process.

    Returns:
        result (Set[str]): A set of extracted strings.
    """
    result = []

    def _process_data(data):
        if isinstance(data, dict):
            result.append(str(list(data.values())))
        elif isinstance(data, list):
            for item in data:
                _process_data(item)
        else:
            try:
                data = literal_eval(data)
                _process_data(data)
            except Exception:
                result.append(data)

    _process_data(data)

    return set(result)


class JaccardSimilarity(NonLLMStringSimilarity):
    """
    Jaccard similarity metric for string comparison.
    """

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Calculates the Jaccard similarity between the reference and response
        strings.

        Args:
            sample (SingleTurnSample): The SingleTurnSample object containing
                the data for evaluation.
            callbacks (Callbacks): Callbacks to use during the evaluation.

        Returns:
            score (float): The Jaccard similarity score between 0.0 and 1.0.
        """
        reference = sample.reference or ""
        response = sample.response or ""

        try:
            processed_response_str = separate_punctuation_with_spaces(response.lower())
            processed_reference_str = separate_punctuation_with_spaces(
                reference.lower()
            )
            words_response = set(processed_response_str.split())
            words_reference = set(processed_reference_str.split())
        except Exception:
            return 0.0

        intersection = words_response.intersection(words_reference)
        union = words_response.union(words_reference)
        len_intersection = len(intersection)
        len_union = len(union)

        if len_union == 0:
            return 1.0

        score = float(len_intersection) / len_union

        return score


class NonLLMContextPrecisionMod(NonLLMContextPrecisionWithReference):
    """
    Standard context precision metric.
    """

    name: str = "standard_precision"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Calculates the standard precision score.

        Uses `process_data()` to extract relevant info and applies string similarity.

        Args:
            sample (SingleTurnSample): The SingleTurnSample object containing the data
                for evaluation.
            callbacks (Callbacks): Callbacks to use during the evaluation.

        Returns:
            score (float): The standard context precision score.
        """
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        retrieved_contexts = process_data(retrieved_contexts)
        reference_contexts = process_data(reference_contexts)

        scores = []
        for rc in retrieved_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for ref in reference_contexts
                    ]
                )
            )
        scores = [1 if score >= self.threshold else 0 for score in scores]
        return self._calculate_standard_precision(scores)

    def _calculate_standard_precision(self, verdict_list: List[int]) -> float:
        """
        Calculate standard precision from list of binary relevancy.

        Args:
            verdict_list (List[int]): A list of binary relevancy (0 or 1).

        Returns:
            score (float): The calculated standard precision.
        """
        if not verdict_list:
            return 0.0

        relevant_count = sum(verdict_list)
        total_count = len(verdict_list)

        return relevant_count / total_count


class NonLLMContextRecallMod(NonLLMContextRecall):
    """
    Standard context recall metric.
    """

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Calculates the standard recall score.

        Uses `process_data()` to extract relevant info and applies string similarity.

        Args:
            sample (SingleTurnSample): The SingleTurnSample object containing the data
                for evaluation.
            callbacks (Callbacks): Callbacks to use during the evaluation.

        Returns:
            score (float): The standard context recall score.
        """
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        retrieved_contexts = process_data(retrieved_contexts)
        reference_contexts = process_data(reference_contexts)

        scores = []
        for ref in reference_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for rc in retrieved_contexts
                    ]
                )
            )
        return self._compute_score(scores)


def evaluate_retriever(
    evaluation_dataset: EvaluationDataset, *, experiment_name: str
) -> EvaluationResult:
    """
    Evaluates the RAG retriever performance.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset object
            containing the data for evaluation.
        experiment_name (str): The name of the experiment.

    Returns:
        result (EvaluationResult): The EvaluationResult object containing the
            evaluation results.
    """
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            NonLLMContextPrecisionMod(
                name="precision", distance_measure=JaccardSimilarity(), threshold=0.8
            ),
            NonLLMContextRecallMod(
                name="recall", _distance_measure=JaccardSimilarity(), threshold=0.8
            ),
        ],
        experiment_name=experiment_name,
        run_config=run_config,
    )


def evaluate_text_generation(
    evaluation_dataset: EvaluationDataset,
    *,
    llm_model: BaseLanguageModel,
    embedding_model: Embeddings,
    experiment_name: str
) -> EvaluationResult:
    """
    Evaluates the RAG text generation performance.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset object
            containing the data for evaluation.
        llm_model (BaseLanguageModel): The BaseLanguageModel object.
        embedding_model (Embeddigns): The Embeddings object.
        experiment_name (str): The name of the experiment.

    Returns:
        result (EvaluationResult): The EvaluationResult object containing the
            evaluation results.
    """
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            RougeScore(rouge_type="rougeL", mode="fmeasure", name="rougeL_fmeasure"),
            ResponseRelevancy(),
            AnswerAccuracy(),
            Faithfulness(),
        ],
        llm=LangchainLLMWrapper(llm_model, run_config=run_config),
        embeddings=LangchainEmbeddingsWrapper(embedding_model, run_config=run_config),
        experiment_name=experiment_name,
        run_config=run_config,
    )


def evaluate_tools_selection(
    evaluation_dataset: EvaluationDataset, *, experiment_name: str
) -> EvaluationResult:
    """
    Evaluates the accuracy of retrieval tool selection.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset object
            containing the data for evaluation.
        experiment_name (str): The name of the experiment.

    Returns:
        result (EvaluationResult): The EvaluationResult object containing the
            tool selection accuracy metric.
    """
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[ToolCallAccuracy()],
        experiment_name=experiment_name,
        run_config=run_config,
    )


def evaluate_end_to_end_graph_rag(
    evaluation_dataset: Tuple[EvaluationDataset, Union[EvaluationDataset, None]],
    *,
    llm_model: BaseLanguageModel,
    embedding_model: Embeddings,
    experiment_name: str
) -> Tuple[EvaluationResult, Union[EvaluationResult, None]]:
    """
    Evaluates the end-to-end RAG performance, incorporating both retriever and
    generator metrics.

    Args:
        evaluation_dataset (Tuple[EvaluationDataset, Union[EvaluationDataset, None]]):
            The first is single-turn dataset, the second is multi-turn evaluation dataset.
            The second dataset can be None.
        llm_model (BaseLanguageModel): The BaseLanguageModel object.
        embedding_model (Embeddigns): The Embeddings object.
        experiment_name (str): The name of the experiment.

    Returns:
        result (Tuple[EvaluationResult, Union[EvaluationResult, None]]): A tuple containing
            the EvaluationResult objects for the single-turn and multi-turn evaluations.
            multi_turn_evaluation_result will be None if the multi_turn_evaluation_dataset
            is None
    """
    single_turn_evaluation_dataset, multi_turn_evaluation_dataset = evaluation_dataset

    # ToolCallAccuracy (need multi turn evaluation dataset)
    if multi_turn_evaluation_dataset:
        multi_turn_evaluation_result = evaluate(
            dataset=multi_turn_evaluation_dataset,
            metrics=[ToolCallAccuracy()],
            experiment_name=experiment_name + "_multi_turn",
            run_config=run_config,
        )
    else:
        multi_turn_evaluation_result = None

    # General metric evaluation (need single turn evaluation dataset)
    single_turn_evaluation_result = evaluate(
        dataset=single_turn_evaluation_dataset,
        metrics=[
            NonLLMContextPrecisionMod(
                name="precision", distance_measure=JaccardSimilarity(), threshold=0.8
            ),
            NonLLMContextRecallMod(
                name="recall", _distance_measure=JaccardSimilarity(), threshold=0.8
            ),
            RougeScore(rouge_type="rougeL", mode="fmeasure", name="rougeL_fmeasure"),
            ResponseRelevancy(),
            AnswerAccuracy(),
            Faithfulness(),
        ],
        llm=LangchainLLMWrapper(llm_model, run_config=run_config),
        embeddings=LangchainEmbeddingsWrapper(embedding_model, run_config=run_config),
        experiment_name=experiment_name + "_single_turn",
        run_config=run_config,
    )

    return (single_turn_evaluation_result, multi_turn_evaluation_result)
