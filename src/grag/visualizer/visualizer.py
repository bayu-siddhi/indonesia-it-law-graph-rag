import time
import uuid
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union
)
from neo4j import (
    Result,
    RoutingControl
)
from neo4j.graph import Graph
from IPython.display import HTML
from neo4j_viz import VisualizationGraph
from neo4j_viz.neo4j import from_neo4j
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from pydantic_extra_types.color import Color
from .prompts import (
    CYPHER_FIX_PROMPT,
    CYPHER_GENERATION_PROMPT
)
from ..retrievers import (
    create_text2cypher_retriever_tool,
    extract_cypher
)


CAPTION_MAPPING = {
    "Regulation": ("download_name", Color("#F79767")),
    "Consideration": ("old_caption", Color("#569480")),
    "Observation": ("old_caption", Color("#D9C8AE")),
    "Definition": ("name", Color("#DA7194")),
    "Article:Effective": ("number", Color("#4C8EDA")),
    "Effective:Article": ("number", Color("#4C8EDA")),
    "Article:Ineffective": ("number", Color("#F16667")),
    "Ineffective:Article": ("number", Color("#F16667")),
}


def _get_unique_node_ids(
    result: Graph
) -> List[str]:
    node_ids: Set[str] = set()
    for node in result.nodes:
        node_ids.add(node.element_id)
    
    return list(node_ids)


def _autocomplete_relationship(
    neo4j_graph: Neo4jGraph,
    node_ids: List[str]
) -> Graph:
    result = neo4j_graph._driver.execute_query(
        query_="""
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            OPTIONAL MATCH (re:Regulation)-[rel1]->(n)
            OPTIONAL MATCH (n)-[rel2]->(m)
            WHERE elementId(m) IN $node_ids
            RETURN *
        """,
        parameters_={"node_ids": node_ids},
        routing_=RoutingControl.READ,
        database_=neo4j_graph._database,
        result_transformer_=Result.graph,
    )

    return result


def _remove_attribute_from_node(
    vg: VisualizationGraph,
    attribute: str
) -> None:
    for node in vg.nodes:
        attributes_to_remove = []
        for property in node:
            # property[0] is key, [1] is value
            if attribute in property[0]:
                attributes_to_remove.append(property[0])

        # Hapus atribut di loop terpisah (wajib)
        for property_name in attributes_to_remove:
            delattr(node, property_name)


def _modify_nodes_caption_and_relationship(
    vg: VisualizationGraph,
    caption_mapping: Dict[str, str] 
) -> None:
    for node in vg.nodes:
        if node.caption in caption_mapping.keys():
            node.old_caption = node.caption
            node.caption = eval(
                f"node.{caption_mapping[node.old_caption][0]}"
            )
            node.color = caption_mapping[node.old_caption][1]
            node.caption_size = 1
            node.size = 50
    
    for rel in vg.relationships:
        rel.caption_size = 2


def create_graph_visualizer_tool(
    llm: BaseLanguageModel,
    neo4j_graph: Neo4jGraph,
    cypher_generation_prompt: Optional[BasePromptTemplate] = None,
    cypher_fix_prompt: Optional[BasePromptTemplate] = None,
    caption_mapping: Optional[Dict[str, Tuple[str, Color]]] = None,
    autocomplete_relationship: bool = False,
    verbose: bool = False
) -> Callable[[str, List[str]], Dict[str, Union[HTML, Dict, bool, float]]]:
    
    if cypher_generation_prompt is None:
        cypher_generation_prompt = CYPHER_GENERATION_PROMPT
    if cypher_fix_prompt is None:
        cypher_fix_prompt = CYPHER_FIX_PROMPT
    if caption_mapping is None:
        caption_mapping = CAPTION_MAPPING
    
    cypher_viz_generator = create_text2cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        cypher_llm=llm,
        qa_llm=llm,
        cypher_generation_prompt=cypher_generation_prompt,
        cypher_fix_prompt=cypher_fix_prompt,
        skip_qa_llm=True,
        verbose=verbose
    )

    def graph_visualizer(
        cypher_query: Optional[str] = None,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[HTML, bool, float]]:
        
        start_time = time.time()
        artifact = {}

        if (
            cypher_query
            and node_ids
        ) or (
            cypher_query is None
            and node_ids is None
        ):
            raise ValueError((
                "Exactly one of `cypher_query` or `node_ids` "
                "must be provided.  Do not provide both or neither."
            ))

        if cypher_query:
            # ToolCalling agar return AIMessage berisi artifact 
            new_cypher_query = cypher_viz_generator.invoke(
                ToolCall(
                    name="cypher_viz_generator",
                    args={"query": cypher_query},
                    id=f"run-{uuid.uuid4()}-0",  # required
                    type="tool_call"             # required
                )
            )

            if new_cypher_query.artifact["is_context_fetched"]:
                artifact = new_cypher_query.artifact
                new_cypher_query = extract_cypher(new_cypher_query.content)
                result = neo4j_graph._driver.execute_query(
                    query_=new_cypher_query,
                    routing_=RoutingControl.READ,
                    database_=neo4j_graph._database,
                    result_transformer_=Result.graph,
                )

                if autocomplete_relationship:
                    node_ids = _get_unique_node_ids(result)
                    result = _autocomplete_relationship(
                        neo4j_graph, node_ids
                    )

                vg = from_neo4j(result)
                _remove_attribute_from_node(vg, "date")
                _modify_nodes_caption_and_relationship(
                    vg, caption_mapping
                )
            
            return {
                "viz": vg.render() if artifact else False,
                "run_time": time.time() - start_time,
                "artifact": artifact
            }
        
        else:
            result = _autocomplete_relationship(
                neo4j_graph, node_ids
            )

            vg = from_neo4j(result)
            _remove_attribute_from_node(vg, "date")
            _modify_nodes_caption_and_relationship(
                vg, caption_mapping
            )

            return {
                "viz": vg.render(),
                "run_time": time.time() - start_time,
                "artifact": artifact
            }

    return graph_visualizer
