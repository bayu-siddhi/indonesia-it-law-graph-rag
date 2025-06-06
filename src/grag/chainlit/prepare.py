import os
from typing import (
    Any,
    Callable,
    Dict,
    Tuple
)
from langchain_core.messages import (
    ToolMessage
)
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.grag import (
    create_grag_workflow,
    create_graph_visualizer_tool,
    create_vector_cypher_retriever_tool,
    create_text2cypher_retriever_tool,
    Text2CypherFallback,
)


def prepare_graph_rag(
    llm_name: str
) -> Tuple[CompiledStateGraph, Callable[[ToolMessage], Dict[str, Any]]]:
    
    URI = os.environ["DATABASE_HOST"]
    DATABASE = os.environ["DATABASE_SMALL"]
    USERNAME = os.environ["DATABASE_USERNAME"]
    PASSWORD = os.environ["DATABASE_PASSWORD"]

    neo4j_config = {
        "DATABASE_NAME": DATABASE,
        "ARTICLE_VECTOR_INDEX_NAME": os.environ["ARTICLE_VECTOR_INDEX_NAME"],
        "ARTICLE_FULLTEXT_INDEX_NAME": os.environ["ARTICLE_FULLTEXT_INDEX_NAME"],
        "DEFINITION_VECTOR_INDEX_NAME": os.environ["DEFINITION_VECTOR_INDEX_NAME"],
        "DEFINITION_FULLTEXT_INDEX_NAME": os.environ["DEFINITION_FULLTEXT_INDEX_NAME"],
    }

    neo4j_graph = Neo4jGraph(
        url=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        enhanced_schema=True
    )

    neo4j_driver = neo4j_graph._driver

    embedder_model = HuggingFaceEmbeddings(
        model_name=os.environ["EMBEDDING_MODEL"]
    )

    if llm_name == "gemini-2.0-flash":
        # TODO: Cari konfigurasi paling bagus
        llm = ChatGoogleGenerativeAI(
            model=llm_name,
            temperature=0.0,
            api_key=os.environ["GOOGLE_API_KEY"],
            streaming=True
        )
    else:
        print(llm_name)
        # TODO: Cari konfigurasi paling bagus
        llm = ChatOllama(
            model=llm_name,
            # num_ctx=16000,
            # num_predict=2048,
            temperature=0.0
        )

    vector_cypher_retriever: Callable[[str], ToolMessage] = \
        create_vector_cypher_retriever_tool(
            embedder_model=embedder_model,
            neo4j_driver=neo4j_driver,
            neo4j_config=neo4j_config,
            total_definition_limit=5,
            top_k_initial_article=5,
            max_k_expanded_article=-1,
            total_article_limit=None
        )

    text2cypher_retriever: Callable[[str], ToolMessage] = \
        create_text2cypher_retriever_tool(
            neo4j_graph=neo4j_graph,
            embedder_model=embedder_model,
            cypher_llm=llm,
            qa_llm=llm,
            skip_qa_llm=True,
            verbose=False
        )

    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]] = \
        create_graph_visualizer_tool(
            llm=llm,
            neo4j_graph=neo4j_graph,
            autocomplete_relationship=True,
            verbose=False
        )

    checkpointer = MemorySaver()

    workflow: CompiledStateGraph = create_grag_workflow(
        model=llm,
        tools=[vector_cypher_retriever],
        # tools=[],
        checkpointer=checkpointer,
        fallback_tool_calling_cls=Text2CypherFallback
    )

    return workflow, graph_visualizer_tool
