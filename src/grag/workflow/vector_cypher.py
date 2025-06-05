import uuid
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from ragas import EvaluationDataset
from neo4j import RoutingControl
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from ..retrievers import create_vector_cypher_retriever_tool


def vector_cypher_workflow(
    evaluation_dataset: EvaluationDataset,
    *,
    embedder_model: Embeddings,
    neo4j_graph: Neo4jGraph,
    neo4j_config: Dict[str, str],
    total_definition_limit: int = 5,
    top_k_initial_article: int = 5,
    max_k_expanded_article: int = -1,
    total_article_limit: Optional[int] = None,
) -> Tuple[EvaluationDataset, List[List[int]]]:
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

    for idx, data in enumerate(evaluation_dataset):
        tool_result = vector_cypher_retriever.invoke(
            ToolCall(
                name=vector_cypher_retriever.model_dump()["name"],
                args={"query": data.user_input},
                id=f"run-{uuid.uuid4()}-0",  # required
                type="tool_call"             # required
            )
        )
        
        article_node_ids = tool_result.artifact["node_ids"][:-total_definition_limit]
        definition_node_ids = tool_result.artifact["node_ids"][-total_definition_limit:]
        art_definition_node_ids = []

        for _idx in range(len(definition_node_ids)):
            new_definition_id = int(str(definition_node_ids[_idx])[:-6] + "500100")
            art_definition_node_ids.append(new_definition_id)

        all_article_node_ids.append(
            article_node_ids + art_definition_node_ids
        )

        query_result = neo4j_graph._driver.execute_query(
            query_="""
                UNWIND $node_ids AS node_id
                MATCH (n)
                WHERE n.id = node_id
                RETURN n.text AS text
            """,
            parameters_={
                "node_ids": article_node_ids + definition_node_ids
            },
            routing_=RoutingControl.READ,
            database_=neo4j_graph._database
        )

        retrieved_contexts = []
        for record in query_result.records:
            retrieved_contexts.append(record["text"])

        evaluation_dataset[idx].retrieved_contexts = retrieved_contexts
    
    return evaluation_dataset, all_article_node_ids
