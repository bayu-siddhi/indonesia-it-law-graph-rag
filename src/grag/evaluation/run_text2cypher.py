"""Text2Cypher retriever workflow"""

import copy
import uuid
from typing import List, Optional, Tuple
from tqdm import tqdm
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset
from ..retrievers import create_text2cypher_retriever_tool, extract_cypher


def run_text2cypher_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseChatModel,
    embedder_model: Optional[Embeddings] = None,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, List[str]]:
    """
    Runs a text-to-cypher retrieval workflow to enriches the evaluation 
    dataset with retrieved contexts and generated Cypher queries.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset object 
            containing questions for evaluation.
        experiment_name (str): The name of the experiment.
        neo4j_graph (Neo4jGraph): The Neo4j graph instance.
        cypher_llm (BaseChatModel): The language model used for generating 
            Cypher queries.
        embedder_model (Optional[Embeddings], optional): The embedding model 
            used for semantic similarity search (few-shot examples). Defaults 
            to None.
        verbose (bool, optional): Whether to display progress information. 
            Defaults to True.

    Returns:
        results (Tuple[EvaluationDataset, List[str]]): A tuple containing 
            the modified EvaluationDataset object with retrieved contexts 
            and a list of generated Cypher queries.
    """
    generated_cypher_results = []
    evaluation_dataset = copy.deepcopy(evaluation_dataset)

    text2cypher_retriever = create_text2cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        cypher_llm=cypher_llm,
        embedder_model=embedder_model,
        add_context_to_artifact=True,
    )

    for data in tqdm(
        iterable=evaluation_dataset,
        desc=f"Running text2cypher_retriever: `{experiment_name}`",
        disable=not verbose,
    ):
        tool_result = text2cypher_retriever.invoke(
            ToolCall(
                name=text2cypher_retriever.model_dump()["name"],
                args={"query": data.user_input},
                id=f"run-{uuid.uuid4()}-0",
                type="tool_call",
            )
        )

        generated_cypher_results.append(extract_cypher(tool_result.content))

        if tool_result.artifact["is_context_fetched"]:
            retrieved_contexts = []
            for context in tool_result.artifact["context"]:
                retrieved_contexts.append(str(context))
        else:
            retrieved_contexts = [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]

        data.retrieved_contexts = retrieved_contexts


    return evaluation_dataset, generated_cypher_results
