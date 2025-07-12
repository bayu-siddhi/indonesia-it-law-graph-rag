"""Vector Cypher retriever tool"""

import time
from typing import Callable, List, Tuple, Union
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from ..models import SimpleQueryInput
from .retrieval_query import (
    ARTICLE_RETRIEVAL_QUERY,
    DEFINITION_RETRIEVAL_QUERY,
)


def _tool_result_formatter(
    articles: List[Tuple[Document, float]], definitions: List[Tuple[Document, float]]
) -> Tuple[str, List[int]]:
    """
    Formats the retriever results into a tool output string and a list
    of node IDs.

    Args:
        articles (List[Tuple[Document, float]]): List of article results and
            score.
        definitions (List[Tuple[Document, float]]): List of definition results
            and score.

    Returns:
        (content, node_ids) (Tuple[str, List[int]]): Formatted output string
            and list of node IDs.
    """
    node_ids = []

    article_result = (
        "## **Daftar Pasal Peraturan Perundang-Undangan yang "
        "(Mungkin) Relevan untuk Menjawab Kueri:**\n"
    )
    article_result += ("-" * 80) + "\n\n"
    for article in articles:
        article_result += article[0].page_content + "\n\n"
        node_ids.append(article[0].metadata["id"])

    definition_result = (
        "## **Daftar Definisi Konsep Menurut Peraturan Perundang-"
        "Undangan yang (Mungkin) Relevan untuk Menjawab Kueri:**\n"
    )
    definition_result += ("-" * 80) + "\n\n"
    for definition in definitions:
        definition_result += "- " + definition[0].page_content + "\n\n"
        node_ids.append(definition[0].metadata["id"])

    content = (article_result + "\n" + definition_result).strip()

    return content, node_ids


def create_vector_cypher_retriever_tool(
    neo4j_graph: Neo4jGraph,
    embedder_model: Embeddings,
    top_k_initial_article: int = 7,
    max_k_expanded_article: int = 100,
    total_definition_limit: int = 5,
) -> Callable[[str], Union[str, ToolMessage]]:
    """
    Create a Vector Cypher retriever tool for retrieving legal articles
    and definitions.

    This tool uses both full-text and vector-based search over a Neo4j
    graph to retrieve nodes representing legal articles and concept
    definitions, and formats the results into structured text.

    Args:
        neo4j_graph (Neo4jGraph): Neo4j graph interface.
        embedder_model (Embeddings): The embedding model for vector search.
        top_k_initial_article (int): Number of top articles to initially
            retrieve. Default to 7.
        max_k_expanded_article (int): Maximum number of related articles to
            expand to from the initial set. Default to 100.
        total_definition_limit (int): Maximum number of definition nodes to
            retrieve. Default to 5.

    Returns:
        vector_cypher_retriever (Callable[[str], ToolMessage]):
            A LangChain-compatible tool callable.
    """
    if (
        top_k_initial_article <= 0
        or max_k_expanded_article <= 0
        or total_definition_limit <= 0
    ):
        raise ValueError(
            "`top_k_initial_article`, `max_k_expanded_article`, and "
            "`total_definition_limit` must be greater than zero (0)"
        )

    article_retriever = Neo4jVector.from_existing_graph(
        embedding=embedder_model,
        node_label="Effective",
        embedding_node_property="embedding",
        text_node_properties=["text"],
        index_name="effective_vector_index",
        retrieval_query=ARTICLE_RETRIEVAL_QUERY,
        graph=neo4j_graph,
    )

    definition_retriever = Neo4jVector.from_existing_graph(
        embedding=embedder_model,
        node_label="Definition",
        embedding_node_property="embedding",
        text_node_properties=["text"],
        index_name="definition_vector_index",
        retrieval_query=DEFINITION_RETRIEVAL_QUERY,
        graph=neo4j_graph,
    )

    @tool(args_schema=SimpleQueryInput, response_format="content_and_artifact")
    def vector_cypher_retriever(query: str) -> Union[str, ToolMessage]:
        """
        Gunakan alat ini untuk mendapatkan data/informasi hukum dari kueri
        yang kurang spesifik. Misal ketika pertanyaan/kueri pengguna menanyakan:

        *   Suatu hal yang bersifat lebih umum, luas, dan tidak secara langsung
            menanyakan struktur atau hubungan spesifik dalam data.
        *   Membutuhkan pencarian kata kunci atau frasa *dalam teks* isi peraturan
            atau pasal.
        *   Membutuhkan sintesis informasi dari berbagai bagian graf yang tidak mudah
            diambil dengan kueri terstruktur sederhana.
        *   Pertanyaan yang jelas tidak masuk kategori `text2cypher_retriever`.

        **Catatan Penting**: Gunakan alat ini jika Anda ragu harus menggunakan alat
        apa untuk mengambil data/informasi hukum untuk menjawab kueri pengguna.
        """

        start_time = time.time()
        query = query.lower()

        # Article search
        articles = article_retriever.similarity_search_with_score(
            query=query,
            k=top_k_initial_article,
            params={"limit": max_k_expanded_article},
        )

        # Definition search
        definitions = definition_retriever.similarity_search_with_score(
            query=query, k=total_definition_limit
        )

        # Format output and record node IDs
        content, node_ids = _tool_result_formatter(articles, definitions)

        run_time = time.time() - start_time
        artifact = {
            "run_time": run_time,
            "is_context_fetched": True,
            "node_ids": node_ids,
        }

        return content, artifact

    return vector_cypher_retriever
