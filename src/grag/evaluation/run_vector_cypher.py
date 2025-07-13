"""Vector-Cypher retriever workflow"""

import copy
import uuid
from tqdm import tqdm
from neo4j import RoutingControl
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset
from ..retrievers import create_vector_cypher_retriever_tool
from ...prep.encodings import REGULATION_CODES


def run_vector_cypher_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    neo4j_graph: Neo4jGraph,
    embedder_model: Embeddings,
    top_k_initial_article: int,
    max_k_expanded_article: int = 100,
    total_definition_limit: int = 5,
    verbose: bool = True,
) -> EvaluationDataset:
    """
    Runs a vector-cypher retrieval workflow to enriches the evaluation 
    dataset with retrieved contexts.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset object 
            containing questions for evaluation.
        experiment_name (str): The name of the experiment.
        neo4j_graph (Neo4jGraph): The Neo4j graph instance.
        embedder_model (Embeddings): The embedding model used for vector search.
        top_k_initial_article (int): The number of initial articles to retrieve 
            using vector search.
        max_k_expanded_article (int, optional): The maximum number of articles 
            to expand to using cypher. Defaults to 100.
        total_definition_limit (int, optional): The maximum number of definitions 
            to retrieve. Defaults to 5.
        verbose (bool, optional): Whether to display progress information. Defaults 
            to True.

    Returns:
        result (EvaluationDataset): The modified EvaluationDataset object with 
            retrieved contexts added to each data point.
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)

    vector_cypher_retriever = create_vector_cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        embedder_model=embedder_model,
        top_k_initial_article=top_k_initial_article,
        max_k_expanded_article=max_k_expanded_article,
        total_definition_limit=total_definition_limit,
    )

    for data in tqdm(
        iterable=evaluation_dataset,
        desc=f"Running vector_cypher_retriever: `{experiment_name}`",
        disable=not verbose,
    ):
        tool_result = vector_cypher_retriever.invoke(
            ToolCall(
                name=vector_cypher_retriever.model_dump()["name"],
                args={"query": data.user_input},
                id=f"run-{uuid.uuid4()}-0",
                type="tool_call",
            )
        )
        article_node_ids = tool_result.artifact["node_ids"][:-total_definition_limit]
        definition_node_ids = tool_result.artifact["node_ids"][-total_definition_limit:]
        art_definition_node_ids = []

        for def_node_id in definition_node_ids:
            new_definition_id = int(
                str(def_node_id)[:-6] + REGULATION_CODES["section"]["article"] + "00100"
            )
            art_definition_node_ids.append(new_definition_id)

        current_article_node_ids = list(
            dict.fromkeys(article_node_ids + art_definition_node_ids)
        )

        query_result = neo4j_graph._driver.execute_query(
            query_="""
                UNWIND $node_ids AS node_id
                MATCH (n)
                WHERE n.id = node_id
                RETURN n.text AS text
            """,
            parameters_={"node_ids": current_article_node_ids},
            routing_=RoutingControl.READ,
            database_=neo4j_graph._database,
        )

        retrieved_contexts = []
        for record in query_result.records:
            retrieved_contexts.append(record["text"])
        data.retrieved_contexts = retrieved_contexts

    return evaluation_dataset
