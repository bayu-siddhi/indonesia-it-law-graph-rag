"""Vector Cypher retriever evaluation workflow"""

import copy
import uuid
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from neo4j import RoutingControl
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset
from ..retrievers import create_vector_cypher_retriever_tool
from ...prep.encodings import REGULATION_CODES


def vector_cypher_eval_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    embedder_model: Embeddings,
    neo4j_graph: Neo4jGraph,
    neo4j_config: Dict[str, str],
    total_definition_limit: int = 5,
    top_k_initial_article: int = 5,
    max_k_expanded_article: int = -1,
    total_article_limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, List[List[int]]]:
    """
    TODO: Docstring
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)
    vector_cypher_retriever = create_vector_cypher_retriever_tool(
        embedder_model=embedder_model,
        neo4j_driver=neo4j_graph._driver,
        neo4j_config=neo4j_config,
        total_definition_limit=total_definition_limit,
        top_k_initial_article=top_k_initial_article,
        max_k_expanded_article=max_k_expanded_article,
        total_article_limit=total_article_limit,
    )

    all_article_node_ids = []

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

        all_article_node_ids.append(current_article_node_ids)

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

    return evaluation_dataset, all_article_node_ids
