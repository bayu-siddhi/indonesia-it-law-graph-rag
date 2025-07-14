"""Initializes the system and configures the Graph-RAG components"""

import os
from typing import Any, Callable, Dict, Tuple, Union
from langchain_core.messages import ToolMessage
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.grag import (
    create_graph_rag_workflow,
    create_graph_visualizer_tool,
    create_vector_cypher_retriever_tool,
    create_text2cypher_retriever_tool,
    FallbackToolCalling,
)


def initialize_system() -> Tuple[Neo4jGraph, Embeddings]:
    """
    Initializes the Neo4j graph database and the embedding model.

    Retrieves Neo4j connection details from environment variables and 
    initializes a Neo4jGraph object. Also initializes the Embeddings model.

    Returns:
        results (Tuple[Neo4jGraph, Embeddings]): A tuple containing the 
            initialized Neo4jGraph and Embeddings objects.
    """
    URI = os.environ["NEO4J_HOST"]
    DATABASE = os.environ["NEO4J_DATABASE"]
    USERNAME = os.environ["NEO4J_USERNAME"]
    PASSWORD = os.environ["NEO4J_PASSWORD"]

    neo4j_graph = Neo4jGraph(
        url=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        enhanced_schema=True,
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.environ["EMBEDDING_MODEL"]
    )

    return neo4j_graph, embedding_model


def configure_graph_rag(
    llm_name: str, neo4j_graph: Neo4jGraph, embedder_model: Embeddings
) -> Tuple[CompiledStateGraph, Callable[[ToolMessage], Dict[str, Any]]]:
    """
    Configures the Graph-RAG workflow with the specified language model, 
    Neo4j graph, and embedding model.

    This function initializes the language model based on the provided name, 
    creates the vector-cypher and text2cypher retrievers, configures the graph 
    visualizer tool, creates a memory-based checkpointer and creates the state 
    graph, then combines them into a compiled state graph.

    Args:
        llm_name (str): The name of the language model to use.
        neo4j_graph (Neo4jGraph): The Neo4j graph instance.
        embedder_model (Embeddings): The embedding model instance.

    Returns:
        results (Tuple[CompiledStateGraph, Callable[[ToolMessage], Dict[str, Any]]]): 
            A tuple containing the compiled state graph and the graph visualizer tool.
    
    Raises:
        exception (ValueError): If the specified LLM model is not supported.
    """
    if llm_name.startswith("claude"):
        llm = ChatAnthropic(
            model_name=llm_name,
            max_tokens_to_sample=4096,
            temperature=0.0,
            timeout=None,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    elif llm_name == "llama3.1:8b-instruct-q4_K_M":
        llm = ChatOllama(
            model=llm_name,
            num_ctx=32768,
            num_predict=4096,
            temperature=0.0,
        )
    else:
        raise ValueError(f"LLM model `{llm_name}` is not supported")

    vector_cypher_retriever: Callable[[str], Union[str, ToolMessage]] = (
        create_vector_cypher_retriever_tool(
            neo4j_graph=neo4j_graph,
            embedder_model=embedder_model,
            top_k_initial_article=7,
        )
    )

    text2cypher_retriever: Callable[[str], Union[str, ToolMessage]] = (
        create_text2cypher_retriever_tool(
            neo4j_graph=neo4j_graph, embedder_model=embedder_model, cypher_llm=llm
        )
    )

    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]] = (
        create_graph_visualizer_tool(
            llm=llm, neo4j_graph=neo4j_graph, autocomplete_relationship=True
        )
    )

    checkpointer = MemorySaver()

    graph_rag: CompiledStateGraph = create_graph_rag_workflow(
        model=llm,
        tools=[text2cypher_retriever, vector_cypher_retriever],
        checkpointer=checkpointer,
        fallback_tool_calling_cls=FallbackToolCalling,
    )

    return graph_rag, graph_visualizer_tool
