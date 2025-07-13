"""A class to build a knowledge graph of regulations in a Neo4j database."""

import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import neo4j
import pyvis
from tqdm import tqdm
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer


class RegulationGraphBuilder:
    """
    A class to build a knowledge graph of regulations in a Neo4j database.
    """

    def __init__(
        self,
        uri: str,
        auth: Tuple[str, str],
        database: str,
        embedding_model: str,
    ) -> None:
        """
        Initializes the RegulationGraphBuilder with connection details and 
        the embedding model.

        Args:
            uri (str): The Neo4j connection URI.
            auth (Tuple[str, str]): The Neo4j authentication credentials 
                (username, password).
            database (str): The Neo4j database name.
            embedding_model (str): The name of the SentenceTransformer 
                embedding model to use.
        
        Returns:
            None
        """
        self.URI = uri
        self.AUTH = auth
        self.DATABASE = database
        self.embedding_model = embedding_model

    # https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/managing-indexes/
    def _create_index_id(self, tx: neo4j.Session) -> None:
        """
        Creates indexes on the `id` property for various node labels 
        in the Neo4j database.
        
        Args:
            tx (neo4j.Session): The Neo4j transaction object.

        Returns:
            None
        """
        range_indexes = {
            "Regulation": "regulation_id_index",
            "Consideration": "consideration_id_index",
            "Observation": "observation_id_index",
            "Definition": "definition_id_index",
            "Article": "article_id_index",
            "Effective": "effective_id_index",
            "Ineffective": "ineffective_id_index",
        }

        for label, index_name in range_indexes.items():
            tx.run(
                query="""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.id)
                """.format(
                    index_name=index_name, label=label
                )
            )

    # https://neo4j.com/docs/python-manual/current/data-types/#_date
    def _string_to_neo4j_date(self, date: str) -> Optional[neo4j.time.Date]:
        """
        Parses a string representing a date in the format YYYY-MM-DD and 
        converts it to a Neo4j Date object. 
        
        If the string does not match the expected format, returns None.

        Args:
            date (str): A string representing a date in YYYY-MM-DD format.

        Returns:
            neo4j_date (Optional[neo4j.time.Date]): A Neo4j Date object 
                representing the parsed date, or None if parsing fails.
        """
        date_components = re.search(r"(\d{4})-(\d{2})-(\d{2})", date)
        neo4j_date = (
            neo4j.time.Date(
                year=int(date_components[1]),
                month=int(date_components[2]),
                day=int(date_components[3]),
            )
            if date_components
            else None
        )
        return neo4j_date

    def _create_regulation_and_subject_node(
        self, tx: neo4j.Session, regulation: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Creates a Regulation node and Subject nodes, and connects them with a 
        HAS_SUBJECT relationship.

        Merges a Regulation node based on its ID. If it exists, it updates its 
        properties; otherwise, it creates a new node. It also creates Subject 
        nodes for each subject listed in the regulation and links them to the 
        Regulation node with HAS_SUBJECT relationships.

        Args:
            tx (neo4j.Session): The Neo4j transaction object.
            regulation (Dict[str, Any]): A dictionary containing the regulation 
                data

        Returns:
            result (Dict[str, int]): A dictionary containing the ID of the 
                Regulation node, count of the Regulation nodes (should be 1), 
                and count of the created HAS_SUBJECT relationships.
        """
        regulation_result = tx.run(
            query="""
            MERGE (r:Regulation {id: $id})
            SET r.title = $title,
                r.type = $type,
                r.number = $number,
                r.year = $year,
                r.is_amendment = $is_amendment,
                r.amendment_number = $amendment_number,
                r.institution = $institution,
                r.issue_place = $issue_place,
                r.issue_date = $issue_date,
                r.effective_date = $effective_date,
                r.reference_url = $reference_url,
                r.download_url = $download_url,
                r.download_name = $download_name 
            RETURN r.id AS ID, COUNT(r) AS Regulation
            """,
            parameters={
                "id": int(regulation["id"]),
                "title": regulation["title"],
                "type": regulation["short_type"],
                "number": int(regulation["number"]),
                "year": int(regulation["year"]),
                "is_amendment": bool(int(regulation["amendment"])),
                "amendment_number": int(regulation["amendment"]),
                "institution": regulation["institution"],
                "issue_place": regulation["issue_place"],
                "issue_date": self._string_to_neo4j_date(regulation["issue_date"]),
                "effective_date": self._string_to_neo4j_date(
                    regulation["effective_date"]
                ),
                "reference_url": regulation["url"],
                "download_url": regulation["download_link"],
                "download_name": regulation["download_name"],
            },
        )

        edge_result = tx.run(
            query="""
            UNWIND $subjects AS subject
            MATCH (r:Regulation {id: $regulation_id})
            MERGE (s:Subject {title: subject})
            MERGE (r)-[rel:HAS_SUBJECT]->(s) 
            RETURN COUNT(rel) AS HAS_SUBJECT
            """,
            parameters={
                "regulation_id": int(regulation["id"]),
                "subjects": regulation["subjects"],
            },
        )

        regulation_result = regulation_result.single()
        edge_result = edge_result.single()

        result = {}
        result["ID"] = regulation_result["ID"]
        result["Regulation"] = regulation_result["Regulation"]
        result["HAS_SUBJECT"] = edge_result["HAS_SUBJECT"]

        return result

    def _create_reg_amendment_rel(
        self, tx: neo4j.Session, regulation: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Creates AMENDED_BY relationships between regulations that are amended.

        For each regulation in the `amend` list of the provided regulation's status, 
        it creates an AMENDED_BY relationship from the amended regulation to the 
        current regulation, only if the amended regulation is not from bpk.go.id.

        Args:
            tx (neo4j.Session): The Neo4j transaction object.
            regulation (Dict[str, Any]): A dictionary containing the regulation data, 
                including the "status" key with an "amend" list of regulation IDs.

        Returns:
            result (Dict[str, int]): A dictionary containing the count of created 
                AMENDED_BY relationships.
        """
        result = {"AMENDED_BY": 0}
        for amended_regulation in regulation["status"]["amend"]:
            if (
                re.search(
                    r"peraturan\.bpk\.go\.id", amended_regulation, re.IGNORECASE
                ) is None
            ):
                query_result = tx.run(
                    query="""
                    MATCH (current_regulation:Regulation {id: $current_regulation})
                    MATCH (amended_regulation:Regulation {id: $amended_regulation})
                    MERGE (amended_regulation)-[rel:AMENDED_BY {
                        amendment_number: current_regulation.amendment_number
                    }]->(current_regulation)
                    RETURN COUNT(rel) AS num_edges
                    """,
                    parameters={
                        "current_regulation": int(regulation["id"]),
                        "amended_regulation": int(amended_regulation),
                    },
                )

                result["AMENDED_BY"] += query_result.single()["num_edges"]

        return result

    def _create_regulation_content(
        self, tx: neo4j.Session, regulation: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Creates content nodes (Consideration, Observation, Definition, Article) 
        and their relationships to the Regulation node.

        Iterates through the "content" dictionary of the regulation, creating nodes 
        for considerations, observations, definitions, and articles. It also creates 
        relationships between these content nodes and the Regulation node, as well 
        as relationships between articles (NEXT_ARTICLE, PREVIOUS_ARTICLE, REFER_TO, 
        AMENDED_BY).

        Args:
            tx (neo4j.Session): The Neo4j transaction object.
            regulation (Dict[str, Any]): A dictionary containing the regulation data, 
                including the "content" key, which is a dictionary containing 
                considerations, observations, definitions, and articles.

        Returns:
            result (Dict[str, int]): A dictionary containing the counts of created 
                nodes and relationships.
        """
        result = {
            "Consideration": 0,
            "HAS_CONSIDERATION": 0,
            "Observation": 0,
            "HAS_OBSERVATION": 0,
            "Definition": 0,
            "HAS_DEFINITION": 0,
            "Article": 0,
            "HAS_ARTICLE": 0,
            "NEXT_ARTICLE": 0,
            "PREVIOUS_ARTICLE": 0,
            "REFER_TO": 0,
            "Art_AMENDED_BY": 0,
        }

        for key, content in regulation["content"].items():
            if key in ["consideration", "observation"]:
                modified_text = (
                    "Daftar "
                    f"{'Latar Belakang' if key == 'consideration' else 'Dasar Hukum'} "
                    f"dari {regulation['title']}:\n"
                    f"{content['text']}".strip()
                )

                query_result = tx.run(
                    query="""
                    MERGE (n {id: $id})
                    SET n.text = $modified_text,
                        n.real_text = $real_text,
                        n.title = "Daftar " + $object + " dari "
                            + $regulation_short_title
                    WITH n
                    CALL apoc.create.addLabels(n, $labels)
                    YIELD node
                    MATCH (reg:Regulation {id: $regulation_id})
                    CALL apoc.create.relationship(reg, $relationship_type, {}, node)
                    YIELD rel
                    RETURN COUNT(node) AS num_nodes, COUNT(rel) AS num_edges
                    """,
                    parameters={
                        "id": int(content["id"]),
                        "modified_text": modified_text,
                        "real_text": content["text"].lower(),
                        "object": (
                            "Latar Belakang"
                            if key == "consideration"
                            else "Dasar Hukum"
                        ),
                        "regulation_short_title": regulation["short_title"],
                        "labels": [key.title()],
                        "regulation_id": int(regulation["id"]),
                        "relationship_type": (
                            "HAS_CONSIDERATION"
                            if key == "consideration"
                            else "HAS_OBSERVATION"
                        ),
                    },
                )

                query_result = query_result.single()
                if key == "consideration":
                    result["Consideration"] += query_result["num_nodes"]
                    result["HAS_CONSIDERATION"] += query_result["num_edges"]
                else:
                    result["Observation"] += query_result["num_nodes"]
                    result["HAS_OBSERVATION"] += query_result["num_edges"]

            elif key == "definitions":
                for definition in content:
                    modified_text = (
                        f"Definisi \"{definition['name']}\" Menurut Pasal 1 "
                        f"{regulation['title']}:\n"
                        f"{definition['definition']}".strip()
                    )

                    query_result = tx.run(
                        query="""
                        MERGE (n:Definition {id: $id})
                        SET n.name = $name,
                            n.text = $modified_text,
                            n.real_text = $real_text,
                            n.title = "Definisi '" + $name
                                + "' Menurut Pasal 1 " + $regulation_short_title
                        WITH n
                        MATCH (reg:Regulation {id: $regulation_id})
                        MERGE (reg)-[rel:HAS_DEFINITION]->(n)
                        RETURN COUNT(n) AS num_nodes, COUNT(rel) AS num_edges
                        """,
                        parameters={
                            "id": int(definition["id"]),
                            "name": definition["name"],
                            "regulation_short_title": regulation["short_title"],
                            "modified_text": modified_text,
                            "real_text": definition["definition"].lower(),
                            "regulation_id": int(regulation["id"]),
                        },
                    )

                    query_result = query_result.single()
                    result["Definition"] += query_result["num_nodes"]
                    result["HAS_DEFINITION"] += query_result["num_edges"]

            else:
                for article in content.values():
                    parts = [
                        regulation['title'],
                        article['chapter_about'],
                        article['part_about'],
                        article['paragraph_about'],
                        f"Pasal {article['article_number']}:\n{article['text']}".strip()
                    ]

                    modified_text = ", ".join(filter(None, parts))

                    query_result = tx.run(
                        query="""
                        MERGE (n:Article:Effective {id: $id})
                        SET n.number = $number,
                            n.chapter = $chapter,
                            n.part = $part,
                            n.paragraph = $paragraph,
                            n.title = "Pasal " + $number + " " + $regulation_short_title,
                            n.text = $modified_text,
                            n.real_text = $real_text,
                            n.next_article = $next_article_id,
                            n.status = "Berlaku"
                        WITH n
                        MATCH (reg:Regulation {id: $regulation_id})
                        MERGE (reg)-[rel:HAS_ARTICLE]->(n)
                        RETURN COUNT(n) AS num_nodes, COUNT(rel) AS num_edges
                        """,
                        parameters={
                            "id": int(article["id"]),
                            "number": article["article_number"],
                            "chapter": article["chapter_number"],
                            "part": article["part_number"],
                            "paragraph": article["paragraph_number"],
                            "regulation_short_title": regulation["short_title"],
                            "modified_text": modified_text,
                            "real_text": article["text"].lower(),
                            "next_article_id": (
                                int(article["next_article"])
                                if article["next_article"]
                                else None
                            ),
                            "regulation_id": int(regulation["id"]),
                        },
                    )

                    query_result = query_result.single()
                    result["Article"] += query_result["num_nodes"]
                    result["HAS_ARTICLE"] += query_result["num_edges"]

                    if article["previous_article"] != "":
                        query_result = tx.run(
                            query="""
                            MATCH (regulation:Regulation)-[:HAS_ARTICLE]->(article:Article {
                                id: $article_id
                            })
                            MATCH (prev_article:Article {id: $prev_article_article_id})
                            MERGE (prev_article)-[next_rel:NEXT_ARTICLE {
                                    amendment_number: regulation.amendment_number,
                                    is_effective: true
                                }]->(article)
                            MERGE (article)-[previous_rel:PREVIOUS_ARTICLE {
                                    amendment_number: regulation.amendment_number,
                                    is_effective: true
                                }]->(prev_article)
                            RETURN COUNT(next_rel) AS NEXT_ARTICLE, 
                                COUNT(previous_rel) AS PREVIOUS_ARTICLE
                            """,
                            parameters={
                                "article_id": int(article["id"]),
                                "prev_article_article_id": int(
                                    article["previous_article"]
                                ),
                            },
                        )

                        query_result = query_result.single()
                        result["NEXT_ARTICLE"] += query_result["NEXT_ARTICLE"]
                        result["PREVIOUS_ARTICLE"] += query_result["PREVIOUS_ARTICLE"]

                    if article["references"]:
                        for reference_article_id in article["references"]:
                            query_result = tx.run(
                                query="""
                                MATCH (article:Article {id: $article_id})
                                MATCH (reference_article:Article {id: $reference_article_id})
                                MERGE (article)-[rel:REFER_TO]->(reference_article)
                                RETURN COUNT(rel) AS num_edges
                                """,
                                parameters={
                                    "article_id": int(article["id"]),
                                    "reference_article_id": int(reference_article_id),
                                },
                            )

                            query_result = query_result.single()
                            result["REFER_TO"] += query_result["num_edges"]

                    if article["amend"]:
                        for amended_article_id in article["amend"]:
                            query_result = tx.run(
                                query="""
                                MATCH (article:Article {id: $article_id})
                                MATCH (amended_article:Article {id: $amended_article_id})
                                MERGE (amended_article)-[rel:AMENDED_BY {
                                    amendment_number: $amendment_number
                                }]->(article)
                                RETURN COUNT(rel) AS num_edges
                                """,
                                parameters={
                                    "article_id": int(article["id"]),
                                    "amended_article_id": int(amended_article_id),
                                    "amendment_number": int(regulation["amendment"]),
                                },
                            )

                            query_result = query_result.single()
                            result["Art_AMENDED_BY"] += query_result["num_edges"]

        return result

    def _complete_article_sequence_rel(
        self, tx: neo4j.Session, regulation: Dict[str, Any]
    ) -> neo4j.Record:
        """
        Completes the `NEXT_ARTICLE` relationships between articles within a 
        regulation.

        This function ensures that every article with a `next_article` property 
        has an explicit `NEXT_ARTICLE` relationship with the subsequent article, 
        even in cases where articles have been amended. It addresses situations 
        where the relationship is not automatically formed due to amendments, 
        ensuring the article sequence is complete.

        Args:
            tx (neo4j.Session): The Neo4j session object for executing queries.
            regulation (Dict[str, Any]): The regulation data containing the 
                regulation ID.

        Returns:
            result (neo4j.Record): A query result containing the number of 
                `NEXT_ARTICLE` and `PREVIOUS_ARTICLE` relationships that were 
                created. The values are accessed via keys "NEXT_ARTICLE" and 
                "PREVIOUS_ARTICLE" respectively.
        """
        query_result = tx.run(
            query="""
            MATCH (current_regulation:Regulation)-[:HAS_ARTICLE]->(current: Article)
            WHERE current_regulation.id = $regulation_id 
                AND current.next_article IS NOT NULL
            MATCH (next:Article {id: current.next_article})
            WHERE NOT (current)-[:NEXT_ARTICLE]->(next)
            MERGE (current)-[next_rel:NEXT_ARTICLE {
                    amendment_number: current_regulation.amendment_number,
                    is_effective: true
                }]->(next)
            MERGE (next)-[previous_rel:PREVIOUS_ARTICLE {
                    amendment_number: current_regulation.amendment_number,
                    is_effective: true
                }]->(current)
            RETURN COUNT(next_rel) AS NEXT_ARTICLE, 
                COUNT(previous_rel) AS PREVIOUS_ARTICLE
            """,
            parameters={"regulation_id": int(regulation["id"])},
        )

        return query_result.single()

    def _set_ineffective_node_and_edge(self, tx: neo4j.Session) -> bool:
        """
        Marks amended articles and outdated `NEXT_ARTICLE`/`PREVIOUS_ARTICLE` 
        relationships as ineffective.

        This function performs three main operations:
        1. Marks amended `Article` nodes as `Ineffective`, removes the `Effective` 
            label, and sets their status to "Tidak Berlaku".  It also sets the 
            `is_effective` property of incoming and outgoing `NEXT_ARTICLE` 
            relationships to `False`.
        2. Identifies `Article` nodes with multiple `NEXT_ARTICLE` relationships 
            and marks all but the one with the highest `amendment_number` as 
            ineffective.
        3. Identifies `Article` nodes with multiple `PREVIOUS_ARTICLE` relationships 
            and marks all but the one with the highest `amendment_number` as 
            ineffective.

        Args:
            tx (neo4j.Session): The Neo4j transaction object.

        Returns:
            result (bool): True if any nodes or relationships were modified, False 
                otherwise.
        """
        query_result_1 = tx.run(
            query="""
            MATCH (amended:Article)-[rel:AMENDED_BY]->(:Article)
            REMOVE amended:Effective
            WITH amended
            SET amended.status = "Tidak Berlaku"
            WITH amended
            // OPTIONAL because Article 1 definitely doesn't have one
            OPTIONAL MATCH ()-[rel_in:NEXT_ARTICLE]->(amended)
            // OPTIONAL because the final Article definitely doesn't have one
            OPTIONAL MATCH (amended)-[rel_out:NEXT_ARTICLE]->()
            SET amended:Ineffective,
                rel_in.is_effective = False,
                rel_out.is_effective = False
            RETURN COUNT(amended) AS num_nodes
            """
        )

        query_result_2 = tx.run(
            query="""
            MATCH (a:Article)-[rel:NEXT_ARTICLE]->(next:Article)
            WITH a, 
                COLLECT(next) AS next_articles, 
                COLLECT(rel.amendment_number) AS orders
            WHERE SIZE(next_articles) > 1
            // Only those who have > 1 Next Article

            // Find the largest amendment_number from the NEXT_ARTICLE 
            // relationship
            WITH a, next_articles, orders, 
                REDUCE(
                    maxOrder = 0, o IN orders | 
                    CASE 
                        WHEN o > maxOrder THEN o ELSE maxOrder 
                    END
                ) AS max_amendment_number

            // Fetch only the NEXT ARTICLE with the largest amendment number
            UNWIND next_articles AS candidate
            WITH a, candidate, max_amendment_number
            MATCH (a)-[rel:NEXT_ARTICLE]->(candidate)
            WHERE rel.amendment_number <> max_amendment_number
            SET rel.is_effective = False

            RETURN COUNT(rel) AS num_edges
            """
        )

        query_result_3 = tx.run(
            query="""
            MATCH (previous:Article)<-[rel:PREVIOUS_ARTICLE]-(a:Article)
            WITH a, 
                COLLECT(previous) AS previous_article,
                COLLECT(rel.amendment_number) AS orders
            WHERE SIZE(previous_article) > 1
            // Only those who have > 1 Previous Article

            // Find the largest amendment_number from the PREVIOUS_ARTICLE 
            // relationship
            WITH a, previous_article, orders, 
                REDUCE(
                    maxOrder = 0, o IN orders | 
                    CASE 
                        WHEN o > maxOrder THEN o ELSE maxOrder 
                    END
                ) AS max_amendment_number

            // Fetch only PREVIOUS_ARTICLE with the largest amendment_number
            UNWIND previous_article AS candidate
            WITH a, candidate, max_amendment_number
            MATCH (a)-[rel:PREVIOUS_ARTICLE]->(candidate)
            WHERE rel.amendment_number <> max_amendment_number
            SET rel.effective = False

            RETURN COUNT(rel) AS num_edges
            """
        )

        query_result_1 = query_result_1.single()
        query_result_2 = query_result_2.single()
        query_result_3 = query_result_3.single()

        return bool(
            query_result_1["num_nodes"]
            + query_result_2["num_edges"]
            + query_result_3["num_edges"]
        )

    def _import_batch(
        self,
        tx: neo4j.Session,
        nodes_with_embeddings: List[Dict[str, Union[int, List[float]]]],
    ) -> None:
        """
        Imports a batch of node embeddings into Neo4j.

        This function takes a list of dictionaries, each containing a node ID and 
        its corresponding embedding vector, and updates the specified nodes in the 
        Neo4j database with the embedding. Applies to Article, Definition, 
        Consideration, and Observation nodes.

        Args:
            tx (neo4j.Session): The Neo4j transaction object.
            nodes_with_embeddings (List[Dict[str, Union[int, List[float]]]]): A 
                list of dictionaries, where each dictionary contains the node ID 
                ("id") and its embedding vector ("embedding").
                
        Returns:
            None
        """
        # Add embeddings to Consideration, Observation, Definition, and Article nodes
        tx.run(
            query="""
            UNWIND $nodes as node
            MATCH (n:Article|Definition|Consideration|Observation {id: node.id})
            CALL db.create.setNodeVectorProperty(n, "embedding", node.embedding)
            """,
            parameters={"nodes": nodes_with_embeddings},
        )

    # https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/embeddings/sentence-transformers/
    def _create_vector_embedding(
        self, driver: neo4j.Driver, batch_size: int, verbose: bool = True
    ) -> neo4j.Record:
        """
        Creates vector embeddings for Article, Definition, Consideration, and 
        Observation nodes and stores them in Neo4j.

        This function fetches text content from nodes of specified labels, 
        generates embeddings using a SentenceTransformer model, and writes these 
        embeddings back to the nodes in Neo4j. 
        
        It also creates vector indexes to enable efficient similarity searches.

        Args:
            driver (neo4j.Driver): The Neo4j driver instance for database 
                interaction.
            batch_size (int): The number of nodes to process in each batch.
            verbose (bool, optional): Whether to display a progress bar. 
                Defaults to True.

        Returns:
            records (neo4j.Record): A record containing the count of nodes with 
                embeddings and the embedding size. Returns None if 
                records["count_nodes_with_embeddings"] is not > 0.
        """
        nodes_with_embeddings = []
        model = SentenceTransformer(self.embedding_model)

        with driver.session(database=self.DATABASE) as session:
            result = session.execute_read(
                lambda tx: list(
                    tx.run(
                        query="""
                        MATCH (n:Article|Definition|Consideration|Observation)
                        RETURN n.id AS id, n.real_text AS text
                        """
                    )
                )
            )

            num_result = len(result)
            total_batch = num_result / batch_size

            # Create batching numbering
            for batch_n in tqdm(
                iterable=range(
                    1,
                    int(total_batch + float(total_batch % 1 > 0) + 1.0)
                ),
                desc="Create vector embeddings  ",
                disable=not verbose,
            ):

                # Process per batch (e.g., batch_size = 32, batch 1 will be result[0:32])
                for record in result[(batch_n - 1) * batch_size : batch_n * batch_size]:
                    node_id = record.get("id")
                    node_text = record.get("text")

                    # Create embedding for text node
                    if node_id is not None and node_text is not None:
                        nodes_with_embeddings.append(
                            {"id": node_id, "embedding": model.encode(node_text)}
                        )

                    # Import when a batch of movies has embeddings ready; flush buffer
                    if len(nodes_with_embeddings) == batch_size:
                        session.execute_write(self._import_batch, nodes_with_embeddings)
                        nodes_with_embeddings = []

            # Flush last batch
            session.execute_write(self._import_batch, nodes_with_embeddings)

            # Import complete, show counters
            records = session.execute_read(
                lambda tx: tx.run(
                    query="""
                    MATCH (n:Article|Definition|Consideration|Observation)
                    WHERE n.embedding IS NOT NULL
                    RETURN count(*) AS count_nodes_with_embeddings,
                        size(n.embedding) AS embedding_size
                    """
                ).single()
            )

            if records["count_nodes_with_embeddings"] > 0:
                with driver.session(database=self.DATABASE) as session:
                    vector_index_dict = {
                        "effective_vector_index": "Effective",
                        "definition_vector_index": "Definition",
                    }

                    # Create vector index for the embeedings
                    for index_name, label in vector_index_dict.items():
                        session.execute_write(
                            lambda tx: tx.run(
                                query=f"""
                                CREATE VECTOR INDEX {index_name}
                                IF NOT EXISTS FOR (n:{label})
                                ON n.embedding
                                OPTIONS {{ indexConfig: {{
                                    `vector.dimensions`: $vector_dimensions,
                                    `vector.similarity_function`: "cosine"
                                }} }}
                                """,
                                parameters={
                                    "vector_dimensions": records["embedding_size"]
                                },
                            )
                        )
            else:
                print("Failed to create node embeddings")

        return records

    def _create_related_to_relationship(
        self, driver: neo4j.Driver
    ) -> List[neo4j.Record]:
        """
        Creates `RELATED_TO` relationships between `Effective` articles based 
        on semantic similarity.

        This function performs a vector similarity search to find semantically 
        similar articles across different regulations. It creates `RELATED_TO` 
        relationships between articles that meet the specified similarity 
        threshold (0.96) and are not definition or effective date articles.

        Args:
            driver (neo4j.Driver): The Neo4j driver instance for database 
                interaction.

        Returns:
            query_results (List[neo4j.Record]): A list of records containing 
                the regulation name and the count of `RELATED_TO` relationships 
                created for that regulation. Ordered by regulation name.
        """
        # Create text index for text node
        with driver.session(database=self.DATABASE) as session:
            session.execute_write(
                lambda tx: tx.run(
                    query="""
                    CREATE TEXT INDEX effective_text_index
                    IF NOT EXISTS FOR (n:Effective) ON (n.text)
                    """
                )
            )

        # Create REALTED_TO relationship between effective article, that:
        # 1. Are in different regulations
        # 2. Have similarity level > 0.96
        # 3. Is not a definition article and not an effective date article
        with driver.session(database=self.DATABASE) as session:
            query_results = session.execute_write(
                lambda tx: list(
                    tx.run(
                        query="""
                        MATCH (regulation:Regulation)-[:HAS_ARTICLE]->(article:Effective)

                        WITH regulation.download_name AS regulation_name,
                            article.id AS source_id,
                            article.embedding AS embedding,
                            toInteger(substring(toString(article.id), 0, 9)) AS base

                        WITH regulation_name, source_id, embedding,
                            base * 1000000 AS lower_bound_id,
                            (base + 1) * 1000000 AS upper_bound_id

                        CALL db.index.vector.queryNodes(
                            "effective_vector_index", 100, embedding
                        )
                        YIELD node, score
                        WHERE (node.id < lower_bound_id OR node.id > upper_bound_id)
                            AND score > 0.96

                        WITH regulation_name, source_id, node.id AS target_id, score

                        MATCH (source_node:Effective {id: source_id})
                        MATCH (target_node:Effective {id: target_id})
                        WHERE NOT source_node.text CONTAINS "yang dimaksud dengan"
                            AND NOT source_node.text CONTAINS "berlaku pada tanggal"
                            AND NOT target_node.text CONTAINS "yang dimaksud dengan"
                            AND NOT target_node.text CONTAINS "berlaku pada tanggal"
                        MERGE (source_node)-[rel:RELATED_TO {score: score}]->(target_node)

                        RETURN regulation_name, COUNT(DISTINCT rel) AS RELATED_TO
                        ORDER BY regulation_name ASC
                        """
                    )
                )
            )

        return query_results

    def _delete_unused_properties(self, tx: neo4j.Session) -> bool:
        """
        Deletes unused properties from nodes to optimize database storage.

        This function removes the `next_article` and `real_text` properties 
        from nodes with the labels `Regulation`, `Consideration`, `Observation`, 
        `Definition`, and `Article`.

        Args:
            tx (neo4j.Session): The Neo4j transaction object.

        Returns:
            result (bool): True if any nodes were modified (properties deleted), 
                False otherwise.
        """
        query_result = tx.run(
            query="""
            MATCH (n:Regulation|Consideration|Observation|Definition|Article)
            REMOVE n.next_article, n.real_text
            RETURN COUNT(n) AS num_nodes
            """
        )

        query_result = query_result.single()
        return bool(query_result["num_nodes"])

    def detach_delete_all(self) -> bool:
        """
        Deletes all nodes and indexes from the Neo4j database.

        This function performs a complete cleanup of the Neo4j database by 
        deleting all nodes and dropping all indexes. It first deletes all nodes 
        and relationships, then retrieves a list of existing indexes, and finally 
        drops each index whose name ends with "_index".

        Returns:
            result (bool): True if the database is empty after the deletion and 
                index removal, False otherwise.
        """
        result = False
        with neo4j.GraphDatabase.driver(uri=self.URI, auth=self.AUTH) as driver:
            with driver.session(database=self.DATABASE) as session:
                session.execute_write(
                    lambda tx: tx.run(query="MATCH (n) DETACH DELETE n")
                )
                indexes = session.execute_write(
                    lambda tx: list(tx.run(query="SHOW INDEX"))
                )

                for index in indexes:
                    index_name: str = index["name"]
                    if index_name.endswith("_index"):
                        session.execute_write(
                            lambda tx: tx.run(query=f"DROP INDEX {index_name}")
                        )

                result = session.execute_read(
                    lambda tx: tx.run(
                        query="MATCH (n) RETURN COUNT(n) AS num_nodes"
                    ).single()
                )

                result = not bool(result["num_nodes"])

        return result

    def build_graph(
        self, json_input: str, batch_size: int = 64, verbose=True
    ) -> Dict[str, List[Union[str, int]]]:
        """
        Builds a knowledge graph in Neo4j from a JSON file containing regulation 
        data.

        This function reads regulation data from a JSON file, creates nodes and 
        relationships in Neo4j representing the regulations, their content 
        (considerations, observations, definitions, articles), and relationships 
        between articles (NEXT_ARTICLE, PREVIOUS_ARTICLE, REFER_TO, AMENDED_BY). 
        
        It also generates vector embeddings for text content and creates `RELATED_TO` 
        relationships based on semantic similarity.

        Args:
            json_input (str): The path to the JSON file containing the regulation 
                data.
            batch_size (int, optional): The number of nodes to process in each batch 
                when creating vector embeddings. Defaults to 64.
            verbose (bool, optional): Whether to display progress information. 
                Defaults to True.

        Returns:
            summary (Dict[str, List[Union[str, int]]]): A dictionary summarizing the 
                number of nodes and relationships created for each entity type.

        """
        if not json_input.endswith(".json"):
            json_input = json_input + ".json"

        with open(json_input, encoding="utf-8") as file:
            json_data = json.load(file)

        summary = {
            "Name": [],
            "ID": [],
            "Regulation": [],
            "Consideration": [],
            "Observation": [],
            "Definition": [],
            "Article": [],
            "HAS_SUBJECT": [],
            "Reg_AMENDED_BY": [],
            "HAS_CONSIDERATION": [],
            "HAS_OBSERVATION": [],
            "HAS_DEFINITION": [],
            "HAS_ARTICLE": [],
            "NEXT_ARTICLE": [],
            "PREVIOUS_ARTICLE": [],
            "REFER_TO": [],
            "Art_AMENDED_BY": [],
            "RELATED_TO": [],
        }

        with neo4j.GraphDatabase.driver(uri=self.URI, auth=self.AUTH) as driver:
            start_time = time.time()

            with driver.session(database=self.DATABASE) as session:
                # Create node ID index
                session.execute_write(self._create_index_id)

                # Iterate for all regulation data
                for regulation in tqdm(
                    iterable=json_data,
                    desc="Building regulation graph ",
                    disable=not verbose,
                ):
                    summary["Name"].append(regulation["download_name"])
                    summary["RELATED_TO"].append(0)

                    # Create Regulation, Subject, HAS_SUBJECT
                    results = session.execute_write(
                        self._create_regulation_and_subject_node, regulation
                    )
                    summary["ID"].append(results["ID"])
                    summary["Regulation"].append(results["Regulation"])
                    summary["HAS_SUBJECT"].append(results["HAS_SUBJECT"])

                    # Create edge: AMENDED_BY (Reg.)
                    results = session.execute_write(
                        self._create_reg_amendment_rel, regulation
                    )
                    summary["Reg_AMENDED_BY"].append(results["AMENDED_BY"])

                    # Create all (except RELATED_TO)
                    results = session.execute_write(
                        self._create_regulation_content, regulation
                    )
                    for key, value in results.items():
                        summary[key].append(value)

                    # Complete NEXT_ARTICLE, PREVIOUS_ARTICLE
                    results = session.execute_write(
                        self._complete_article_sequence_rel, regulation
                    )
                    for key, value in results.items():
                        summary[key][-1] += value

                # Set ineffective nodes and edges
                session.execute_write(self._set_ineffective_node_and_edge)

            # Create embedding for all text nodes
            self._create_vector_embedding(
                driver=driver, batch_size=batch_size, verbose=verbose
            )

            # Create RELATED_TO based on articles similarity
            results = self._create_related_to_relationship(driver=driver)
            for result in results:
                index = summary["Name"].index(result["regulation_name"])
                summary["RELATED_TO"][index] = result["RELATED_TO"]

            # Delete unused properties
            with driver.session(database=self.DATABASE) as session:
                session.execute_write(self._delete_unused_properties)

        if verbose:
            print(
                "Finished building regulation graph in "
                f"{round(time.time() - start_time, 2)} seconds"
            )
            self.print_summary(summary=summary)

        return summary

    def print_summary(self, summary: Dict[str, List[Union[str, int]]]) -> None:
        """
        Prints a summary of the graph building process using PrettyTable.

        This function takes a summary dictionary generated by the `build_graph` 
        method and prints it in a formatted table using the PrettyTable library. 
        The table includes the counts for each entity type and a total row at the 
        end.

        Args:
            summary (Dict[str, List[Union[str, int]]]): A dictionary containing 
                the summary data, as generated by the `build_graph` method.

        Returns:
            None
        """
        table = PrettyTable()

        for key, value in summary.items():
            table.add_column(key, value)

        table.add_divider()
        table.add_row(
            ["TOTAL", ""] + [sum(value) for value in list(summary.values())[2:]]
        )

        for key in list(summary.keys())[2:]:
            table.align[key] = "r"

        print(table)

    def visualize_graph(self, output_html: str) -> None:
        """
        Visualizes the Neo4j graph and saves the visualization to an HTML file.

        This function retrieves the entire graph structure from Neo4j, uses pyvis 
        to create an interactive visualization, and saves the visualization to an 
        HTML file. Node labels are determined based on node type and a predefined 
        mapping of node types to text properties.

        Args:
            output_html (str): The path to the output HTML file where the 
                visualization will be saved.

        Returns:
            None
        """
        if not output_html.endswith(".html"):
            output_html = output_html + ".html"

        graph_result = None

        with neo4j.GraphDatabase.driver(uri=self.URI, auth=self.AUTH) as driver:
            with driver.session(database=self.DATABASE) as session:
                # Query to get a graphy result
                graph_result = session.execute_read(
                    lambda tx: tx.run(
                        query="MATCH (n)-[r]-(m) RETURN n, r, m"
                    ).graph()
                )

        # What property to use as text for each node
        nodes_text_properties = {
            "Subject": "title",
            "Regulation": "download_name",
            "Consideration": "title",
            "Observation": "title",
            "Definition": "name",
            "Effective": "number",
            "Ineffective": "number",
        }

        # Draw graph
        visual_graph = pyvis.network.Network(height="1080px")

        # Draw node
        for node in graph_result.nodes:
            node_label = list(set(node.labels) & set(nodes_text_properties))[0]
            node_text = node[nodes_text_properties[node_label]]
            visual_graph.add_node(
                n_id=node.element_id, label=node_text, group=node_label
            )

        # Draw relationship
        for relationship in graph_result.relationships:
            visual_graph.add_edge(
                source=relationship.start_node.element_id,
                to=relationship.end_node.element_id,
                title=relationship.type,
            )

        visual_graph.show(output_html, notebook=False)
