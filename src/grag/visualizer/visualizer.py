"""Graph visualizer tool"""

import time
import uuid
from typing import Callable, Dict, List, Set, Tuple, Union
from neo4j import Result, RoutingControl
from neo4j.graph import Graph
from neo4j_viz import VisualizationGraph
from neo4j_viz.neo4j import from_neo4j
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import BasePromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall, ToolMessage
from pydantic_extra_types.color import Color
from .prompts import CYPHER_FIX_PROMPT, CYPHER_GENERATION_PROMPT
from ..retrievers import create_text2cypher_retriever_tool, extract_cypher


CAPTION_MAPPING = {
    "Regulation": {"caption": "download_name", "color": Color("#F79767")},
    "Subject": {"caption": "title", "color": Color("#ffdf81"), "size": 40},
    "Consideration": {"caption": "old_caption", "color": Color("#569480")},
    "Observation": {"caption": "old_caption", "color": Color("#D9C8AE")},
    "Definition": {"caption": "name", "color": Color("#DA7194")},
    "Article:Effective": {"caption": "number", "color": Color("#4C8EDA")},
    "Effective:Article": {"caption": "number", "color": Color("#4C8EDA")},
    "Article:Ineffective": {"caption": "number", "color": Color("#F16667")},
    "Ineffective:Article": {"caption": "number", "color": Color("#F16667")},
    # "size": 50  # default
}


def _get_unique_node_ids(result: Graph) -> List[str]:
    """
    Extracts unique node IDs from a Neo4j Graph result object.

    Args:
        result (Graph): A Neo4j query result in graph format.

    Returns:
        List[str]: A list of unique node IDs found in the graph
            result.
    """
    node_ids: Set[str] = set()
    for node in result.nodes:
        node_ids.add(node["id"])

    return list(node_ids)


def _autocomplete_relationship(neo4j_graph: Neo4jGraph, node_ids: List[str]) -> Graph:
    """
    Retrieves related relationships for the specified node IDs from the
    Neo4j database.

    Args:
        neo4j_graph (Neo4jGraph): The Neo4j graph instance used to run
            the query.
        node_ids (List[str]): A list of node IDs to find relationships
            for.

    Returns:
        Graph: A graph result including nodes and relationships.
    """
    result = neo4j_graph._driver.execute_query(
        query_="""
            MATCH (n)
            WHERE n.id IN $node_ids
            OPTIONAL MATCH (re:Regulation)-[rel1]->(n)
            OPTIONAL MATCH (re)-[rel2]->(su:Subject)
            OPTIONAL MATCH (n)-[rel3]->(m)
            WHERE m.id IN $node_ids
            RETURN *
        """,
        parameters_={"node_ids": node_ids},
        routing_=RoutingControl.READ,
        database_=neo4j_graph._database,
        result_transformer_=Result.graph,
    )

    return result


def _remove_attribute_from_node(vg: VisualizationGraph, attribute: str) -> None:
    """
    Removes a given attribute from every node in a VisualizationGraph.

    Args:
        vg (VisualizationGraph): The visualization graph containing
            nodes.
        attribute (str): The attribute name (or substring of it) to
            be removed.
    """
    for node in vg.nodes:
        attributes_to_remove = []
        for data in node:
            # data[0] is key, [1] is value
            if attribute in data[0]:
                attributes_to_remove.append(data[0])

        # Hapus atribut di loop terpisah (wajib)
        for attribute_name in attributes_to_remove:
            delattr(node, attribute_name)


def _modify_nodes_caption_and_relationship(
    vg: VisualizationGraph, caption_mapping: Dict[str, str]
) -> None:
    """
    Modifies node captions and colors based on a caption mapping
    dictionary. Also increases caption size for relationships.

    Args:
        vg (VisualizationGraph): The visualization graph containing
            nodes and relationships.
        caption_mapping (Dict[str, str]): A mapping between original
            captions and their replacement attribute and color.
    """
    for node in vg.nodes:
        if node.caption in caption_mapping:
            node.old_caption = node.caption
            node.caption = eval(f"node.{caption_mapping[node.old_caption]['caption']}")
            # node.caption = f"node.{caption_mapping[node.old_caption]['caption']}"
            node.color = caption_mapping[node.old_caption]["color"]
            node.size = caption_mapping[node.old_caption].get("size", 50)
            node.caption_size = 1

    for rel in vg.relationships:
        rel.caption_size = 2


def create_graph_visualizer_tool(
    llm: BaseChatModel,
    neo4j_graph: Neo4jGraph,
    *,
    cypher_generation_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
    cypher_fix_prompt: BasePromptTemplate = CYPHER_FIX_PROMPT,
    caption_mapping: Dict[str, Tuple[str, Color]] = CAPTION_MAPPING,
    autocomplete_relationship: bool = False,
    verbose: bool = False,
) -> Callable[[ToolMessage], Dict[str, Union[bool, VisualizationGraph, float, Dict]]]:
    """
    Creates a graph visualization tool that supports visualizing Neo4j
    graphs from either Cypher queries or lists of node IDs. Automatically
    formats nodes and relationships with custom styling.

    Args:
        llm (BaseChatModel): Language model used to generate Cypher
            queries.
        neo4j_graph (Neo4jGraph): Neo4j graph instance to run queries on.
        cypher_generation_prompt (BasePromptTemplate): Prompt template for
            Cypher generation.
        cypher_fix_prompt (BasePromptTemplate): Prompt template for Cypher
            fixing.
        caption_mapping (Dict[str, Tuple[str, Color]]): Mapping of node
            types to caption keys and colors.
        autocomplete_relationship (bool): Whether to fetch extra
            relationships for the selected nodes.
        verbose (bool): If True, enables verbose output.

    Returns:
        graph_visualizer (Callable): A callable function `graph_visualizer` that
            can be invoked with either a Cypher query or a list of node IDs.
    """
    # Create a Cypher generator tool that returns artifacts and Cypher code
    cypher_viz_generator = create_text2cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        cypher_llm=llm,
        qa_llm=llm,
        cypher_generation_prompt=cypher_generation_prompt,
        cypher_fix_prompt=cypher_fix_prompt,
        skip_qa_llm=True,
        verbose=verbose,
    )

    def graph_visualizer(
        tool_message: ToolMessage,
    ) -> Dict[str, Union[VisualizationGraph, bool, float]]:
        """
        Visualizes a subgraph from Neo4j using either a Cypher query or
        node IDs from ToolMessage.

        Args:
            tool_message (ToolMessage): ToolMessage contain either Cypher
                query in `.content`, or key "node_ids" in `.artifact`.

        Returns:
            result (Dict[str, Union[VisualizationGraph, bool, float]]):
                A dictionary containing:
                - "viz": the VisualizationGraph instance (or False if failed),
                - "run_time": execution time,
                - "artifact": additional metadata and model usage info.
        """
        start_time = time.time()
        artifact = {}

        # Validate input
        if tool_message.name == "text2cypher_retriever":
            cypher_query = extract_cypher(tool_message.content)
            node_ids = None
        elif tool_message.name == "vector_cypher_retriever":
            node_ids = tool_message.artifact["node_ids"]
            cypher_query = None
        else:
            return {
                "viz": False,
                "run_time": time.time() - start_time,
                "artifact": artifact,
            }

        if cypher_query:
            # Modify the Cypher query to return all data.
            new_cypher_query = cypher_viz_generator.invoke(
                ToolCall(
                    name="cypher_viz_generator",
                    args={"query": cypher_query},
                    id=f"run-{uuid.uuid4()}-0",  # required
                    type="tool_call",  # required
                )
            )

            # If context was successfully fetched
            if new_cypher_query.artifact["is_context_fetched"]:
                artifact = new_cypher_query.artifact
                new_cypher_query = extract_cypher(new_cypher_query.content)
                result = neo4j_graph._driver.execute_query(
                    query_=new_cypher_query,
                    routing_=RoutingControl.READ,
                    database_=neo4j_graph._database,
                    result_transformer_=Result.graph,
                )

                # Optionally autocomplete missing relationships
                if autocomplete_relationship:
                    node_ids = _get_unique_node_ids(result)
                    result = _autocomplete_relationship(neo4j_graph, node_ids)

                # Convert raw result to visualization object
                vg = from_neo4j(result)
                _remove_attribute_from_node(vg, "date")
                _modify_nodes_caption_and_relationship(vg, caption_mapping)

            return {
                "viz": vg if artifact else False,
                "run_time": time.time() - start_time,
                "artifact": artifact,
            }

        # When node_ids are given directly
        result = _autocomplete_relationship(neo4j_graph, node_ids)

        vg = from_neo4j(result)
        _remove_attribute_from_node(vg, "date")
        _modify_nodes_caption_and_relationship(vg, caption_mapping)

        return {
            "viz": vg,
            "run_time": time.time() - start_time,
            "artifact": artifact,
        }

    return graph_visualizer
