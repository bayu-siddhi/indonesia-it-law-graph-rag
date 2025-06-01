import time
import warnings
from typing import (
    Callable,
    Dict,
    List,
    Tuple
)
from neo4j import (
    Driver,
    Record
)
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.embeddings import Embeddings
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.retrievers import VectorCypherRetriever
from ..models import SimpleQuery
from .retrieval_query import (
    ARTICLE_RETRIEVAL_QUERY_1,
    ARTICLE_RETRIEVAL_QUERY_2,
    DEFINITION_RETRIEVAL_QUERY
)


def _retriever_result_formatter(
    record: Record
) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=record["text"],
        metadata={
            k: v for k, v in record["metadata"].items() if v is not None
        }
    )


def _tool_result_formatter(
    articles: List[RetrieverResultItem],
    definitions: List[RetrieverResultItem]
) -> Tuple[str, List[int]]:
    node_ids = []

    article_result = (
        "## **Daftar Pasal Peraturan Perundang-Undangan yang "
        "(Mungkin) Relevan untuk Menjawab Kueri:**\n"
    )
    article_result += (("-" * 80) + "\n\n") 
    for article in articles:
        article_result += (article.content + "\n\n")
        node_ids.append(article.metadata["id"])

    definition_result = (
        "## **Daftar Definisi Konsep Menurut Peraturan Perundang-"
        "Undangan yang (Mungkin) Relevan untuk Menjawab Kueri:**\n"
    )
    definition_result += (("-" * 80) + "\n\n") 
    for definition in definitions:
        definition_result += ("- " + definition.content + "\n\n")
        node_ids.append(definition.metadata["id"])

    content = (article_result + "\n" + definition_result).strip()

    return content, node_ids


def create_vector_cypher_retriever_tool(
    embedder_model: Embeddings,
    neo4j_driver: Driver,
    neo4j_config: Dict[str, str],
    top_k_initial_article: int = 5,
    total_article_limit: int = 10,
    total_definition_limit: int = 5,
) -> Callable[[str], ToolMessage]:
    if top_k_initial_article == total_article_limit:
        warnings.warn(
            "Setting top_k_initial_article equal to total_article_limit "
            "means all article nodes will be retrieved via vector search "
            "only. No additional nodes will be expanded through Cypher "
            "query traversal."
        )
    elif top_k_initial_article > total_article_limit:
        raise ValueError(
            "top_k_initial_article must be less than or equal to total_"
            "article_limit"
        )

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

    @tool(
        args_schema=SimpleQuery,
        response_format="content_and_artifact"
    )
    def vector_cypher_retriever(
        query: str
    ) -> ToolMessage:
        """Vector Cypher Retriever Tool"""
        # TODO: PERBAIKI DESKRIPSI TOOL
        
        start_time = time.time()
        query = query.lower()

        articles = article_retriever_1.search(
            query_text=query,
            top_k=top_k_initial_article,
            query_params={
                "limit": total_article_limit
            }
        )

        if len(articles.items) < total_article_limit:
            additional_articles = article_retriever_2.search(
                query_text=query,
                top_k=top_k_initial_article + total_article_limit,
                query_params={
                    "excluded_ids": [
                        item.metadata["id"] for item in articles.items
                    ],
                    "limit": total_article_limit - len(articles.items)
                }
            )

            for item in additional_articles.items:
                articles.items.append(item)
        
            articles.items = sorted(
                articles.items,
                key=lambda item: item.metadata["score"],
                reverse=True
            )
        
        definitions = definition_retriever.search(
            query_text=query,
            top_k=total_definition_limit,
            query_params={
                "limit": total_definition_limit
            }
        )

        content, node_ids = _tool_result_formatter(
            articles.items, definitions.items
        )
        run_time = time.time() - start_time
        
        artifact = {
            "run_time": run_time,
            "is_context_fetched": True,
            "node_ids": node_ids
        }
        
        return content, artifact
    
    return vector_cypher_retriever
    