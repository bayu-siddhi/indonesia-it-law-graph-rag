"""Vector Cypher retriever tool"""

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
from neo4j import Driver, Record
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.embeddings import Embeddings
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.retrievers import VectorCypherRetriever
from ..models import SimpleQueryInput
from .retrieval_query import (
    ARTICLE_RETRIEVAL_QUERY_1,
    ARTICLE_RETRIEVAL_QUERY_2,
    DEFINITION_RETRIEVAL_QUERY,
)


def _input_validation(
    total_definition_limit: int,
    top_k_initial_article: int,
    max_k_expanded_article: int,
    total_article_limit: Optional[int],
) -> int:
    """
    Validates and adjusts input parameters for retrieval limits.

    Args:
        total_definition_limit (int): Maximum number of definition nodes.
        top_k_initial_article (int): Number of initial articles to retrieve.
        max_k_expanded_article (int): Maximum number of related articles to
            expand to (-1 for no limit).
        total_article_limit (Optional[int]): Total article retrieval limit
            (None or -1 for no limit).

    Returns:
        total_article_limit (int): The adjusted total_article_limit (-1 if no
            limit is applied).
    """
    # 1. Handle None for total_article_limit
    if total_article_limit is None:
        total_article_limit = -1

    # 2. Validate basic positive limits
    if total_definition_limit <= 0:
        raise ValueError("`total_definition_limit` must be greater than zero (0)")
    if top_k_initial_article <= 0:
        raise ValueError("`top_k_initial_article` must be greater than zero (0)")
    # total_article_limit can be -1 or > 0
    if total_article_limit == 0 or total_article_limit < -1:
        raise ValueError(
            "`total_article_limit` must be None, -1, or greater than zero (0)"
        )
    # max_k_expanded_article can be -1 or > 0
    if max_k_expanded_article <= 0 and max_k_expanded_article != -1:
        raise ValueError(
            "`max_k_expanded_article` must be (-1) or greater than zero (0)"
        )

    # 3. Handle conflicting limits and infinite expansion scenarios
    should_warn_and_reset_total = False
    warning_message = ""

    # Scenario A: Infinite expansion conflicts with a positive total limit
    if max_k_expanded_article == -1 and total_article_limit > 0:
        should_warn_and_reset_total = True
        warning_message = (
            "Setting `max_k_expanded_article=-1` means `total_article_limit` "
            "will be ignored. This is equivalent to setting "
            "`total_article_limit=None`"
        )
    # Scenario B: Sum of initial and expanded exceeds a positive total limit
    elif (
        max_k_expanded_article > 0
        and total_article_limit > 0
        and (top_k_initial_article + max_k_expanded_article) > total_article_limit
    ):
        should_warn_and_reset_total = True
        warning_message = (
            "Setting (`top_k_initial_article` + `max_k_expanded_article`) "
            "greater than `total_article_limit` means `total_article_limit` "
            "will be ignored. This is equivalent to setting "
            "`total_article_limit=None`"
        )

    # Apply warning and reset if needed
    if should_warn_and_reset_total:
        total_article_limit = -1
        warnings.warn(warning_message)

    return total_article_limit

    # TODO: Hapus jika kode buatan Gemini di atas (skenario A dan B) sudah benar
    # if max_k_expanded_article != -1:
    #     if top_k_initial_article + max_k_expanded_article > total_article_limit:
    #         if total_article_limit != -1:
    #             total_article_limit = -1
    #             warnings.warn(
    #                 "Setting (`top_k_initial_article` + `max_k_expanded_article`) "
    #                 "greater than `total_article_limit` means `total_article_limit` "
    #                 "will be ignored. This is equivalent to setting "
    #                 "`total_article_limit=None`"
    #             )
    # else:
    #     if total_article_limit > 0:
    #         total_article_limit = -1
    #         warnings.warn(
    #             "Setting `max_k_expanded_article=-1` means `total_article_limit` "
    #             "will be ignored. This is equivalent to setting "
    #             "`total_article_limit=None`"
    #         )
    #
    # return total_article_limit


def _retriever_result_formatter(record: Record) -> RetrieverResultItem:
    """
    Formats a Neo4j record into a RetrieverResultItem.

    Args:
        record (Record): A Neo4j record containing 'text' and 'metadata'
            fields.

    Returns:
        retriever_result_item (RetrieverResultItem): A formatted retriever
            result item with content and metadata.
    """
    return RetrieverResultItem(
        content=record["text"],
        metadata={k: v for k, v in record["metadata"].items() if v is not None},
    )


def _tool_result_formatter(
    articles: List[RetrieverResultItem], definitions: List[RetrieverResultItem]
) -> Tuple[str, List[int]]:
    """
    Formats the retriever results into a tool output string and a list
    of node IDs.

    Args:
        articles (List[RetrieverResultItem]): List of article results.
        definitions (List[RetrieverResultItem]): List of definition results.

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
        article_result += article.content + "\n\n"
        node_ids.append(article.metadata["id"])

    definition_result = (
        "## **Daftar Definisi Konsep Menurut Peraturan Perundang-"
        "Undangan yang (Mungkin) Relevan untuk Menjawab Kueri:**\n"
    )
    definition_result += ("-" * 80) + "\n\n"
    for definition in definitions:
        definition_result += "- " + definition.content + "\n\n"
        node_ids.append(definition.metadata["id"])

    content = (article_result + "\n" + definition_result).strip()

    return content, node_ids


def create_vector_cypher_retriever_tool(
    embedder_model: Embeddings,
    neo4j_driver: Driver,
    neo4j_config: Dict[str, str],
    total_definition_limit: int = 5,
    top_k_initial_article: int = 5,
    max_k_expanded_article: int = -1,
    total_article_limit: Optional[int] = None,
) -> Callable[[str], ToolMessage]:
    """
    Create a vector Cypher retriever tool for retrieving legal articles
    and definitions.

    This tool uses vector-based search over a Neo4j graph to retrieve 
    nodes representing legal articles and concept definitions, and 
    formats the results into structured text.

    Args:
        embedder_model (Embeddings): The embedding model for vector search.
        neo4j_driver (Driver): Neo4j driver instance.
        neo4j_config (Dict[str, str]): Configuration dict for Neo4j database
            and index names.
        total_definition_limit (int): Maximum number of definition nodes to
            retrieve.
        top_k_initial_article (int): Number of top articles to initially
            retrieve.
        max_k_expanded_article (int): Maximum number of related articles to
            expand to from the initial set. Use -1 for no limit.
        total_article_limit (Optional[int]): Total article retrieval limit
            including fallback mechanisms. If None, no total limit is enforced.

    Returns:
        vector_cypher_retriever (Callable[[str], ToolMessage]):
            A LangChain-compatible tool callable.
    """
    # Safety checks for input configuration
    total_article_limit = _input_validation(
        total_definition_limit=total_definition_limit,
        top_k_initial_article=top_k_initial_article,
        max_k_expanded_article=max_k_expanded_article,
        total_article_limit=total_article_limit,
    )
    
    # Initialize vector retrievers for articles and definitions
    article_retriever_1 = VectorCypherRetriever(
        driver=neo4j_driver,
        index_name=neo4j_config["ARTICLE_VECTOR_INDEX_NAME"],
        retrieval_query=ARTICLE_RETRIEVAL_QUERY_1,
        embedder=embedder_model,
        result_formatter=_retriever_result_formatter,
        neo4j_database=neo4j_config["DATABASE_NAME"],
    )

    article_retriever_2 = VectorCypherRetriever(
        driver=neo4j_driver,
        index_name=neo4j_config["ARTICLE_VECTOR_INDEX_NAME"],
        retrieval_query=ARTICLE_RETRIEVAL_QUERY_2,
        embedder=embedder_model,
        result_formatter=_retriever_result_formatter,
        neo4j_database=neo4j_config["DATABASE_NAME"],
    )

    definition_retriever = VectorCypherRetriever(
        driver=neo4j_driver,
        index_name=neo4j_config["DEFINITION_VECTOR_INDEX_NAME"],
        retrieval_query=DEFINITION_RETRIEVAL_QUERY,
        embedder=embedder_model,
        result_formatter=_retriever_result_formatter,
        neo4j_database=neo4j_config["DATABASE_NAME"],
    )

    @tool(args_schema=SimpleQueryInput, response_format="content_and_artifact")
    def vector_cypher_retriever(query: str) -> ToolMessage:
        """
        **`vector_cypher_retriever`**:
        *   **Fungsi:** Mengambil informasi dari database graf menggunakan pendekatan yang lebih fleksibel, mungkin melibatkan pencarian teks (full-text dan semantic search) dalam konten node atau properti, atau kueri yang lebih kompleks yang tidak secara langsung memetakan ke kueri Cypher sederhana untuk data terstruktur.
        *   **Kapan Digunakan:** Gunakan alat ini ketika pertanyaan pengguna:
            *   Bersifat lebih umum dan tidak secara langsung menanyakan struktur atau hubungan spesifik.
            *   Membutuhkan pencarian kata kunci atau frasa *dalam teks* isi peraturan atau pasal.
            *   Membutuhkan sintesis informasi dari berbagai bagian graf yang tidak mudah diambil dengan kueri terstruktur sederhana.
            *   Pertanyaan yang tidak jelas masuk kategori `text2cypher_retriever`.
        *   **Singkatnya:** Gunakan untuk pertanyaan yang lebih *luas*, membutuhkan *pencarian konten*, atau tidak dapat diatasi dengan kueri terstruktur sederhana.
        """

        start_time = time.time()
        query = query.lower()

        # Initial article search
        articles = article_retriever_1.search(
            query_text=query,
            top_k=top_k_initial_article,
            query_params={
                "limit": (
                    max_k_expanded_article if max_k_expanded_article != -1 else 100
                )
            },
        )

        # If needed, perform additional article search to fill remaining quota
        if len(articles.items) < total_article_limit:
            additional_articles = article_retriever_2.search(
                query_text=query,
                top_k=total_article_limit,
                query_params={
                    "excluded_ids": [item.metadata["id"] for item in articles.items],
                    "limit": total_article_limit - len(articles.items),
                },
            )

            for item in additional_articles.items:
                articles.items.append(item)

            # Re-rank merged results
            articles.items = sorted(
                articles.items, key=lambda item: item.metadata["score"], reverse=True
            )

        # Definition search
        definitions = definition_retriever.search(
            query_text=query,
            top_k=total_definition_limit,
        )

        # Format output and record node IDs
        content, node_ids = _tool_result_formatter(articles.items, definitions.items)

        run_time = time.time() - start_time
        artifact = {
            "run_time": run_time,
            "is_context_fetched": True,
            "node_ids": node_ids,
        }

        return content, artifact

    return vector_cypher_retriever
