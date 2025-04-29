import re
import json
import time
import neo4j
import pyvis
import tqdm
import prettytable
import sentence_transformers


class RegulationGraphBuilder:

    def __init__(self, uri: str, auth: tuple[str], database: str, embedding_model: str) -> None:
        self.URI = uri
        self.AUTH = auth
        self.DATABASE = database
        self.embedding_model = embedding_model
    

    def detach_delete_all(self) -> bool:
        result = False
        with neo4j.GraphDatabase.driver(uri=self.URI, auth=self.AUTH) as driver:
            with driver.session(database=self.DATABASE) as session:

                session.execute_write(lambda tx: tx.run(query="MATCH (n) DETACH DELETE n"))
                result = session.execute_read(lambda tx: tx.run(query="MATCH (n) RETURN COUNT(n) AS num_nodes").single())
                result = not bool(result["num_nodes"])
        
        return result
    

    def build_regulation_graph(self, json_input: str, batch_size: int = 32, verbose=True) -> dict[list]:
        if not json_input.endswith(".json"):
            json_input = json_input + ".json"

        # https://stackoverflow.com/questions/20199126/reading-json-from-a-file
        with open(json_input, encoding="utf-8") as file:
            json_data = json.load(file)

        summary = {
            'Name': [],
            'ID': [],
            'Regulation': [],
            'Consideration': [],
            'Observation': [],
            'Definition': [],
            'Article': [],
            'Reg_AMENDED_BY': [],
            'HAS_CONSIDERATION': [],
            'HAS_OBSERVATION': [],
            'HAS_DEFINITION': [],
            'HAS_ARTICLE': [],
            'NEXT_ARTICLE': [],
            'PREVIOUS_ARTICLE': [],
            'REFER_TO': [],
            'Art_AMENDED_BY': [],
            'RELATED_TO': []
        }

        with neo4j.GraphDatabase.driver(uri=self.URI, auth=self.AUTH) as driver:
            start_time = time.time()

            with driver.session(database=self.DATABASE) as session:
                # Create node ID index
                session.execute_write(self.__create_id_index)

                for regulation in tqdm.tqdm(iterable=json_data, desc="Building regulation graph ", disable=not verbose):
                    summary["Name"].append(regulation["download_name"])
                    summary["RELATED_TO"].append(0)

                    results = session.execute_write(self.__create_regulation_node, regulation)
                    summary["ID"].append(results["ID"])
                    summary["Regulation"].append(results["Regulation"])

                    results = session.execute_write(self.__create_regulation_amendment_relationship, regulation)
                    summary["Reg_AMENDED_BY"].append(results["AMENDED_BY"])

                    results = session.execute_write(self.__create_regulation_content, regulation)
                    for key, value in results.items():
                        summary[key].append(value) 

                    results = session.execute_write(self.__complete_article_sequence_relationship, regulation)
                    for key, value in results.items():
                        summary[key][-1] += value

                # Set ineffective node and edge
                session.execute_write(self.__set_ineffective_node_and_edge)

            # Create embedding for all text node
            self.__create_vector_embedding(driver=driver, batch_size=batch_size, verbose=verbose)

            # Create RELATED_TO relationship based on articles similarity
            results = self.__create_related_to_relationship(driver=driver)
            for result in results:
                index = summary["Name"].index(result["regulation_name"])
                summary["RELATED_TO"][index] = result["RELATED_TO"]
            
            # Delete unused properties
            with driver.session(database=self.DATABASE) as session:
                session.execute_write(self.__delete_unused_properties)
        
        if verbose:
            print(f"Finished building regulation graph in {round(time.time() - start_time, 2)} seconds")
            self.print_summary(summary=summary)
        
        return summary
    

    def print_summary(self, summary: dict[list]) -> None:
        table = prettytable.PrettyTable()
        
        for key, value in summary.items():
            table.add_column(key, value)
        
        table.add_divider()
        table.add_row(["TOTAL", ""] + [sum(value) for value in list(summary.values())[2:]])
        
        for key in list(summary.keys())[2:]:
            table.align[key] = "r"
        
        print(table)
    

    def visualize_graph(self, output_html: str) -> None:
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
            "Regulation": "download_name",
            "Consideration": "id",
            "Observation": "id",
            "Definition": "name",
            "Article": "number",
        }

        # Draw graph
        visual_graph = pyvis.network.Network()

        # Draw node
        for node in graph_result.nodes:
            # print(list(node.labels))
            node_label = list(node.labels)[0] if list(node.labels)[0] in nodes_text_properties.keys() else "Article"
            # if node_label not in ["Effective", "Ineffective"]:
            node_text = node[nodes_text_properties[node_label]]
            visual_graph.add_node(n_id=node.element_id, label=node_text, group=node_label)

        # Draw relationship
        for relationship in graph_result.relationships:
            visual_graph.add_edge(
                source=relationship.start_node.element_id,
                to=relationship.end_node.element_id,
                title=relationship.type
            )
        
        visual_graph.show(output_html, notebook=False)


    # https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/managing-indexes/
    def __create_id_index(self, tx: neo4j.Session) -> None:
        indexes = {
            "Regulation": "regulation_id_index",
            "Consideration": "consideration_id_index",
            "Observation": "observation_id_index",
            "Definition": "definition_id_index",
            "Article": "article_id_index",
            "Effective": "effective_id_index",
            "Ineffective": "ineffective_id_index"
        }

        for label, index_name in indexes.items():
            tx.run(
                query="""
                CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.id)
                """.format(
                    index_name=index_name,
                    label=label
                )
            )
        
        tx.run(
            query="""
            CREATE FULLTEXT INDEX effective_fulltext_index IF NOT EXISTS
            FOR (n:Effective) ON EACH [n.text]
            """
        )
        

    
    # https://neo4j.com/docs/python-manual/current/data-types/#_date
    def __string_to_neo4j_date(self, date: str) -> neo4j.time.Date | None:
        date = re.search(r"(\d{4})-(\d{2})-(\d{2})", date)
        date = neo4j.time.Date(year=int(date[1]), month=int(date[2]), day=int(date[3])) if date else None
        return date


    def __create_regulation_node(self, tx, regulation) -> neo4j.Record:
        query_result = tx.run(
            query="""
            MERGE (r:Regulation {id: $id})
            SET r.title = $title,
                r.type = $type,
                r.number = $number,
                r.year = $year,
                r.is_amendment = $is_amendment,
                r.order_of_amendment = $order_of_amendment,
                r.institution = $institution,
                r.issue_place = $issue_place,
                r.issue_date = $issue_date,
                r.effective_date = $effective_date,
                r.subjects = $subjects,
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
                "order_of_amendment": int(regulation["amendment"]),
                "institution": regulation["institution"],
                "issue_place": regulation["issue_place"],
                "issue_date": self.__string_to_neo4j_date(regulation["issue_date"]),
                "effective_date": self.__string_to_neo4j_date(regulation["effective_date"]),
                "subjects": regulation["subjects"],
                "reference_url": regulation["url"],
                "download_url": regulation["download_link"],
                "download_name": regulation["download_name"]
            }
        )

        return query_result.single()


    def __create_regulation_amendment_relationship(self, tx, regulation) -> dict:
        result = {"AMENDED_BY": 0}
        for amended_regulation in regulation["status"]["amend"]:
            if re.search(r"peraturan\.bpk\.go\.id", amended_regulation, re.IGNORECASE) is None:
                query_result = tx.run(
                    query="""
                    MATCH (current_regulation:Regulation {id: $current_regulation})
                    MATCH (amended_regulation:Regulation {id: $amended_regulation})
                    MERGE (amended_regulation)-[rel:AMENDED_BY {order_of_amendment: current_regulation.order_of_amendment}]->(current_regulation)
                    RETURN COUNT(rel) AS num_edges
                    """,
                    parameters={
                        "current_regulation": int(regulation["id"]),
                        "amended_regulation": int(amended_regulation)
                    }
                )

                result["AMENDED_BY"] += query_result.single()["num_edges"]
            
        return result


    def __create_regulation_content(self, tx, regulation) -> dict:
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
                    f"{regulation['title']}, "
                    f"Bagian {'Menimbang' if key == 'consideration' else 'Mengingat'}:\n"
                    f"{content['text']}".strip()
                )

                query_result = tx.run(
                    query="""
                    MERGE (n {id: $id})
                    SET n.text = $modified_text,
                        n.real_text = $real_text
                    WITH n
                    CALL apoc.create.addLabels(n, $labels)
                    YIELD node
                    MATCH (regulation:Regulation {id: $regulation_id})
                    CALL apoc.create.relationship(regulation, $relationship_type, {}, node)
                    YIELD rel
                    RETURN COUNT(node) AS num_nodes, COUNT(rel) AS num_edges
                    """,
                    parameters={
                        "id": int(content["id"]),
                        "modified_text": modified_text,
                        "real_text": content["text"].lower(),
                        "labels": [key.title()],
                        "regulation_id": int(regulation["id"]),
                        "relationship_type": "HAS_CONSIDERATION" if key == "consideration" else "HAS_OBSERVATION"
                    }
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
                        f"{regulation['title']}, "
                        f"Definisi {definition['name']}:\n"
                        f"{definition['definition']}".strip()
                    )

                    query_result = tx.run(
                        query="""
                        MERGE (n:Definition {id: $id})
                        SET n.name = $name,
                            n.text = $modified_text,
                            n.real_text = $real_text
                        WITH n
                        MATCH (regulation:Regulation {id: $regulation_id})
                        MERGE (regulation)-[rel:HAS_DEFINITION]->(n)
                        RETURN COUNT(n) AS num_nodes, COUNT(rel) AS num_edges
                        """,
                        parameters={
                            "id": int(definition["id"]),
                            "name": definition["name"],
                            "modified_text": modified_text,
                            "real_text": definition['definition'].lower(),
                            "regulation_id": int(regulation["id"]),
                            "relationship_type": "HAS_DEFINITION"
                        }
                    )

                    query_result = query_result.single()
                    result["Definition"] += query_result["num_nodes"]
                    result["HAS_DEFINITION"] += query_result["num_edges"]

            else:
                for article in content.values():
                    
                    modified_text = (
                        f"{regulation['title']}, "
                        f"{(article['chapter_about'] or '') + ', ' if article['chapter_about'] else ''}"
                        f"{(article['part_about'] or '') + ', ' if article['part_about'] else ''}"
                        f"{(article['paragraph_about'] or '') + ', ' if article['paragraph_about'] else ''}"
                        f"Pasal {article['article_number']}:\n"
                        f"{article['text']}".strip()
                    )

                    query_result = tx.run(
                        query="""
                        MERGE (n:Article:Effective {id: $id})
                        SET n.number = $number,
                            n.chapter = $chapter,
                            n.part = $part,
                            n.paragraph = $paragraph,
                            n.text = $modified_text,
                            n.real_text = $real_text,
                            n.next_article = $next_article_id
                        WITH n
                        MATCH (reg:Regulation {id: $regulation_id})
                        MERGE (reg)-[rel:HAS_ARTICLE]->(n)
                        SET n.name = reg.type + " No. " + reg.number + " Tahun " + reg.year + " Pasal " + n.number
                        RETURN COUNT(n) AS num_nodes, COUNT(rel) AS num_edges
                        """,
                        parameters={
                            "id": int(article["id"]),
                            "number": article["article_number"],
                            "chapter": article["chapter_number"],
                            "part": article["part_number"],
                            "paragraph": article["paragraph_number"],
                            "modified_text": modified_text,
                            "real_text": article["text"].lower(),
                            "next_article_id": int(article["next_article"]) if article["next_article"] else None,
                            "regulation_id": int(regulation["id"])
                        }
                    )

                    query_result = query_result.single()
                    result["Article"] += query_result["num_nodes"]
                    result["HAS_ARTICLE"] += query_result["num_edges"]

                    if article["previous_article"] != "":
                        query_result = tx.run(
                            query="""
                            MATCH (regulation:Regulation)-[:HAS_ARTICLE]->(article:Article {id: $article_id})
                            MATCH (prev_article:Article {id: $prev_article_article_id})
                            MERGE (prev_article)-[next_rel:NEXT_ARTICLE {
                                    order_of_amendment: regulation.order_of_amendment,
                                    effective: true
                                }]->(article)
                            MERGE (article)-[previous_rel:PREVIOUS_ARTICLE {
                                    order_of_amendment: regulation.order_of_amendment,
                                    effective: true
                                }]->(prev_article)
                            RETURN COUNT(next_rel) AS NEXT_ARTICLE, COUNT(previous_rel) AS PREVIOUS_ARTICLE
                            """,
                            parameters={
                                "article_id": int(article["id"]),
                                "prev_article_article_id": int(article["previous_article"])
                            }
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
                                    "reference_article_id": int(reference_article_id)
                                }
                            )

                            query_result = query_result.single()
                            result["REFER_TO"] += query_result["num_edges"]

                    if article["amend"]:
                        for amended_article_id in article["amend"]:
                            query_result = tx.run(
                                query="""
                                MATCH (article:Article {id: $article_id})
                                MATCH (amended_article:Article {id: $amended_article_id})
                                MERGE (amended_article)-[rel:AMENDED_BY {order_of_amendment: $order_of_amendment}]->(article)
                                RETURN COUNT(rel) AS num_edges
                                """,
                                parameters={
                                    "article_id": int(article["id"]),
                                    "amended_article_id": int(amended_article_id),
                                    "order_of_amendment": int(regulation["amendment"])
                                }
                            )

                            query_result = query_result.single()
                            result["Art_AMENDED_BY"] += query_result["num_edges"]

        return result


    def __complete_article_sequence_relationship(self, tx: neo4j.Session, regulation: dict) -> neo4j.Record:
        """
        Melengkapi relasi `NEXT_ARTICLE` antar pasal dalam suatu peraturan.

        Fungsi ini diperlukan karena relasi `NEXT_ARTICLE` dibentuk berdasarkan properti 
        `next_article` dari setiap pasal. Dalam kondisi normal, relasi ini tidak bermasalah 
        karena setiap pasal memiliki urutan yang jelas. Namun, masalah muncul ketika ada 
        pasal yang mengalami amandemen. 

        Pasal amandemen sering kali merupakan hasil percabangan dari pasal sebelumnya, bukan 
        kelanjutan langsung. Sebagai contoh, jika Pasal 10 diamandemen menjadi Pasal 10*, maka 
        `previous_article` dari Pasal 10* tetap mengarah ke Pasal 9. Seharusnya, setelah Pasal 10*, 
        urutan kembali ke Pasal 11. Namun, karena relasi `NEXT_ARTICLE` hanya dibentuk berdasarkan 
        `previous_article`, hubungan dari Pasal 10* ke Pasal 11 tidak akan terbentuk secara otomatis. 

        Fungsi ini memastikan bahwa setiap pasal yang memiliki nilai `next_article` akan memiliki 
        hubungan eksplisit dengan pasal berikutnya, termasuk dalam kasus pasal yang telah diamandemen.

        Args:
            tx (neo4j.Session): Objek sesi Neo4j untuk menjalankan query.
            regulation (dict): Data peraturan yang berisi ID peraturan.

        Returns:
            neo4j.Record: Hasil query yang mengembalikan jumlah relasi NEXT_ARTICLE yang berhasil dibuat.
        """
        query_result = tx.run(
            query="""
            MATCH (current_regulation:Regulation {id: $regulation_id})-[:HAS_ARTICLE]->(current: Article)
            WHERE current.next_article IS NOT NULL
            MATCH (next:Article {id: current.next_article})
            WHERE NOT (current)-[:NEXT_ARTICLE]->(next)
            MERGE (current)-[next_rel:NEXT_ARTICLE {
                    order_of_amendment: current_regulation.order_of_amendment,
                    effective: true
                }]->(next)
            MERGE (next)-[previous_rel:PREVIOUS_ARTICLE {
                    order_of_amendment: current_regulation.order_of_amendment,
                    effective: true
                }]->(current)
            RETURN COUNT(next_rel) AS NEXT_ARTICLE, COUNT(previous_rel) AS PREVIOUS_ARTICLE
            """,
            parameters={
                "regulation_id": int(regulation["id"])
            }
        )
        
        return query_result.single()
    

    def __set_ineffective_node_and_edge(self, tx: neo4j.Session) -> bool:
        query_result_1 = tx.run(
            query="""
            MATCH (amended:Article)-[rel:AMENDED_BY]->(:Article)
            REMOVE amended:Effective
            WITH amended
            OPTIONAL MATCH ()-[rel_in:NEXT_ARTICLE]->(amended)   // OPTIONAL karena Pasal 1 pasti tidak punya
            OPTIONAL MATCH (amended)-[rel_out:NEXT_ARTICLE]->()  // OPTIONAL karena Pasal akhir pasti tidak punya
            SET amended:Ineffective,
                rel_in.effective = False,
                rel_out.effective = False
            RETURN COUNT(amended) AS num_nodes
            """
        )

        query_result_2 = tx.run(
            query="""
            MATCH (a:Article)-[rel:NEXT_ARTICLE]->(next:Article)
            WITH a, COLLECT(next) AS next_articles, COLLECT(rel.order_of_amendment) AS orders
            WHERE SIZE(next_articles) > 1  // Hanya memiliki > 1 Next Article

            // Cari order_of_amendment terbesar dari relationship NEXT_ARTICLE
            WITH a, next_articles, orders, 
                REDUCE(maxOrder = 0, o IN orders | CASE WHEN o > maxOrder THEN o ELSE maxOrder END) AS max_order_of_amendment

            // Ambil hanya NEXT_ARTICLE dengan order_of_amendment terbesar
            UNWIND next_articles AS candidate
            WITH a, candidate, max_order_of_amendment
            MATCH (a)-[rel:NEXT_ARTICLE]->(candidate)
            WHERE rel.order_of_amendment <> max_order_of_amendment
            SET rel.effective = False
            RETURN COUNT(rel) AS num_edges
            """
        )

        query_result_3 = tx.run(
            query="""
            MATCH (previous:Article)<-[rel:PREVIOUS_ARTICLE]-(a:Article)
            WITH a, COLLECT(previous) AS previous_article, COLLECT(rel.order_of_amendment) AS orders
            WHERE SIZE(previous_article) > 1  // Hanya memiliki > 1 Previous Article

            // Cari order_of_amendment terbesar dari relationship PREVIOUS_ARTICLE
            WITH a, previous_article, orders, 
                REDUCE(maxOrder = 0, o IN orders | CASE WHEN o > maxOrder THEN o ELSE maxOrder END) AS max_order_of_amendment

            // Ambil hanya PREVIOUS_ARTICLE dengan order_of_amendment terbesar
            UNWIND previous_article AS candidate
            WITH a, candidate, max_order_of_amendment
            MATCH (a)-[rel:PREVIOUS_ARTICLE]->(candidate)
            WHERE rel.order_of_amendment <> max_order_of_amendment
            SET rel.effective = False
            RETURN COUNT(rel) AS num_edges
            """
        )

        query_result_1 = query_result_1.single()
        query_result_2 = query_result_2.single()
        query_result_3 = query_result_3.single()
        
        return bool(query_result_1["num_nodes"] + query_result_2["num_edges"] + query_result_3["num_edges"])
    

    # https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/embeddings/sentence-transformers/
    def __create_vector_embedding(
            self,
            driver: neo4j.Driver,
            batch_size: int,
            verbose: bool = True
    ) -> neo4j.Record:
        
        nodes_with_embeddings = []
        model = sentence_transformers.SentenceTransformer(self.embedding_model)

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
            for batch_n in tqdm.tqdm(iterable=range(1, int(total_batch + (total_batch % 1 > 0) + 1)),
                                     desc="Create vector embeddings  ", disable=not verbose):
                
                # Process per batch (e.g., batch_size = 32, batch 1 will be result[0:32])
                for record in result[(batch_n - 1) * batch_size: batch_n * batch_size]:
                    node_id = record.get("id")
                    node_text = record.get("text")

                    # Create embedding for text node
                    if node_id is not None and node_text is not None:
                        nodes_with_embeddings.append({
                            "id": node_id,
                            "embedding": model.encode(node_text)
                        })
                    
                    # Import when a batch of movies has embeddings ready; flush buffer
                    if len(nodes_with_embeddings) == batch_size:
                        session.execute_write(self.__import_batch, nodes_with_embeddings)
                        nodes_with_embeddings = []
            
            # Flush last batch
            session.execute_write(self.__import_batch, nodes_with_embeddings)
        
            # Import complete, show counters
            records = session.execute_read(
                lambda tx: tx.run(
                    query="""
                    MATCH (n:Article|Definition|Consideration|Observation WHERE n.embedding IS NOT NULL)
                    RETURN count(*) AS count_nodes_with_embeddings, size(n.embedding) AS embedding_size
                    """
                ).single()
            )

            if records["count_nodes_with_embeddings"] > 0:

                # print("=" * 100)
                # print(f"{'Total nodes with embeddings':<28}: {records['count_nodes_with_embeddings']}\n"
                #       f"{'Embedding model':<28}: {self.embedding_model}\n"
                #       f"{'Embedding size':<28}: {records['embedding_size']}")

                with driver.session(database=self.DATABASE) as session:
                    vector_index_dict = {
                        "effective_vector_index": "Effective", 
                        "definition_vector_index": "Definition" 
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
                                parameters={"vector_dimensions": records["embedding_size"]}
                            )
                        )

                # print("=" * 100)
            
            else:
                print("Failed to create node embeddings")

        return records
    

    def __import_batch(self, tx: neo4j.Session, nodes_with_embeddings: list[dict]) -> None:
        # Add embeddings to Consideration, Observation, Definition, and Article nodes
        tx.run(
            query="""
            UNWIND $nodes as node
            MATCH (n:Article|Definition|Consideration|Observation {id: node.id})
            CALL db.create.setNodeVectorProperty(n, "embedding", node.embedding)
            """,
            parameters={"nodes": nodes_with_embeddings}
        )

    
    def __create_related_to_relationship(self, driver: neo4j.Driver) -> list:
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
        # 2. Have similarity level > 0.95
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

                        CALL db.index.vector.queryNodes("effective_vector_index", 100, embedding)
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
    

    def __delete_unused_properties(self, tx: neo4j.Session) -> bool:
        query_result = tx.run(
            query="""
            MATCH (n:Regulation|Consideration|Observation|Definition|Article)
            REMOVE n.order_of_amendment, n.next_article, n.real_text
            RETURN COUNT(n) AS num_nodes
            """
        )

        query_result = query_result.single()

        return bool(query_result["num_nodes"])
    