"""Text2Cypher retriever few-shot examples"""

from langchain_core.prompts import PromptTemplate


TEXT2CYPHER_EXAMPLES_PROMPT = PromptTemplate.from_template(
    "Tujuan: {intent}\n"
    "Contoh: {query}\n"
    "Neo4j Cypher: {cypher}"
)

TEXT2CYPHER_EXAMPLES = [
    {
        "intent": (
            "mendapatkan isi konten pasal tertentu dari suatu peraturan spesifik"
        ),
        "query": (
            "apa isi pasal 100 undang undang / uu nomor 90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN a.text AS text
```
""",
    },
    {
        "intent": (
            "mendapatkan nomor pasal tertentu dari suatu peraturan spesifik"
        ),
        "query": (
            "apa nomor pasal dari pasal 100 undang undang / uu nomor 90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN a.number AS number
```
""",
    },
    {
        "intent": (
            "mendapatkan judul / nama pasal selanjutnya dari suatu peraturan spesifik"
        ),
        "query": (
            "apa pasal selanjutnya setelah pasal 100 undang-undang / uu nomor 90 tahun "
            "2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:NEXT_ARTICLE]->(next_article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN next_article.title AS title
```
""",
    },
    {
        "intent": (
            "mendapatkan judul / nama pasal yang dirujuk / refered oleh suatu pasal "
            "spesifik"
        ),
        "query": (
            "apa saja pasal yang dirujuk oleh pasal 100 undang-undang / uu nomor 90 "
            "tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:REFER_TO]->(refered_article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN refered_article.title AS title
```
""",
    },
    {
        "intent": (
            "mendapatkan judul / nama pasal yang terkait / related dengan suatu pasal"
        ),
        "query": (
            "apa saja pasal yang terkait dengan pasal 100 undang-undang / uu nomor 90 "
            "tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:RELATED_TO]->(refered_article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN refered_article.title AS title
```
""",
    },
    {
        "intent": (
            "mendapatkan status berlaku atau tidak berlaku / dicabut dari suatu pasal"
        ),
        "query": (
            "apakah pasal 100 undang-undang / uu nomor 90 tahun 2020 masih berlaku?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN a.status AS status
```
""",
    },
    {
        "intent": (
            "mendapatkan judul / nama pasal amandemen / perubahan"
        ),
        "query": (
            "apa pasal amandemen / perubahan dari pasal 100 undang-undang / uu nomor "
            "90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:AMENDED_BY]->(amend_article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN amend_article.title AS title
```
""",
    },
    {
        "intent": (
            "mendapatkan jenis hubungan / relasi antara dua entitas"
        ),
        "query": (
            "Apa hubungan pasal 100 undang-undang / uu nomor 90 tahun 2020, "
            "dengan pasal 101 undang-undang / uu nomor 91 tahun 2021?"
        ),
        "cypher": """```cypher
MATCH (r1:Regulation)-[:HAS_ARTICLE]->(a1:Article)
MATCH (r2:Regulation)-[:HAS_ARTICLE]->(a2:Article)
MATCH (a1)-[rel]-(a2)
WHERE 
    r1.type = 'UU' AND r1.number = 90 AND r1.year = 2020 AND a1.number = '100'
    AND
    r2.type = 'UU' AND r2.number = 91 AND r2.year = 2021 AND a2.number = '101'
RETURN type(rel) AS relationship
```
""",
    },
    {
        "intent": "mendapatkan semua hubungan dari suatu entitas",
        "query": (
            "apa saja semua pasal yang berhubungan dengan pasal 100 dari "
            "undang-undang / uu nomor 90 tahun 2020, dan apa jenis hubungannya?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[rel]->(other_article)
WHERE r.type = 'UU' AND r.number = 11 AND r.year = 2008 AND a.number = '28'
RETURN a.title AS source, type(rel) AS rel, other_article.title AS target
UNION
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)<-[rel]-(other_article)
WHERE r.type = 'UU' AND r.number = 11 AND r.year = 2008 AND a.number = '28'
RETURN other_article.title AS source, type(rel) AS rel, a.title AS target
```
""",
    },
    {
        "intent": (
            "mendapatkan peraturan yang judulnya membahas topik / kata kunci tertentu"
        ),
        "query": (
            "dapatkan lima judul / nama peraturan yang membahas mengenai informasi "
            "elektronik!"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)
WHERE lower(r.title) CONTAINS 'informasi elektronik'
RETURN r.title AS title
LIMIT 5
```
""",
    },
    {
        "intent": (
            "mendapatkan judul / nama peraturan amandemen / perubahan"
        ),
        "query": (
            "apa saja peraturan amandemen / perubahan dari undang-undang / uu nomor 90 "
            "tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[rel:AMENDED_BY]->(amendment_regulation)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020
RETURN amendment_regulation.title AS title
```
""",
    },
    {
        "intent": (
            "mendapatkan bagian pertimbangan / latar belakang  dari terbentuknya suatu "
            "peraturan"
        ),
        "query": (
            "apa pertimbangan / latar belakang dari pembuatan uu nomor 90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_CONSIDERATION]->(consideration)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020
RETURN consideration.text AS text
```
""",
    },
    {
        "intent": (
            "mendapatkan bagian mengingat / daftar dasar hukum dari terbentuknya suatu "
            "peraturan"
        ),
        "query": (
            "apa dasar hukum dari pembuatan uu nomor 90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_OBSERVATION]->(observation)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020
RETURN observation.text AS text
```
""",
    },
    {
        "intent": (
            "mendapatkan definisi suatu konsep di dalam suatu peraturan"
        ),
        "query": (
            "apa definisi 'sistem elektronik' menurut uu nomor 90 tahun 2020?"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_DEFINITION]->(d:Definition)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020
    AND lower(d.name) CONTAINS 'sistem eletronik'
RETURN d.text AS definition
```
""",
    },
    {
        "intent": (
            "mendapatkan daftar subjek yang dimiliki suatu peraturan"
        ),
        "query": (
            "apa saja daftar subjek yang dimiliki oleh uu nomor 90 tahun 2020??"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_SUBJECT]->(s:Subject)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020
RETURN s.title AS title
```
""",
    },
    {
        "intent": (
            "(Analisis Graf) mendapatkan semua entitas paling berpengaruh (pagerank "
            "centrality)"
        ),
        "query": (
            "apa saja sepuluh ** pasal ** yang paling berpengaruh?"
        ),
        "cypher": """```cypher
CALL gds.pageRank.stream('graph')
YIELD nodeId, score
WITH nodeId, score
MATCH (node)
WHERE id(node) = nodeId AND 'Article' IN labels(node) AND 'Effective' IN labels(node)
RETURN node.title AS name, score AS centrality_score
ORDER BY centrality_score DESC
LIMIT 10
```
""",
    },
    {
        "intent": (
            "(Analisis Graf) mendapatkan semua entitas penjembatan paling berpengaruh "
            "(betweenness centrality)"
        ),
        "query": (
            "apa saja lima ** pasal ** penjembatan paling berpengaruh?"
        ),
        "cypher": """```cypher
CALL gds.betweenness.stream('graph')
YIELD nodeId, score
WITH nodeId, score
MATCH (node)
WHERE id(node) = nodeId AND 'Article' IN labels(node) AND 'Effective' IN labels(node)
RETURN node.title AS name, score AS centrality_score
ORDER BY centrality_score DESC
LIMIT 5
```
""",
    },
    {
        "intent": (
            "(Analisis Graf) mendapatkan rekomendasi entitas (personalized pagerank "
            "centrality)"
        ),
        "query": (
            "berikan lima pasal yang Anda rekomendasikan kepada saya jika saya tertarik "
            "dengan pasal 100 undang-undang / uu nomor 90 tahun 2020!"
        ),
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(article:Article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND source.number = '100'
CALL gds.pageRank.stream('graph', {{sourceNodes: [article]}})
// Catatan: Kurung kurawalnya 1 pasang saja, tidak 2 pasang
YIELD nodeId, score
WITH nodeId, score
MATCH (node)
WHERE id(node) = nodeId AND 'Article' IN labels(node) AND 'Effective' IN labels(node)
RETURN node.title AS name, score AS centrality_score
ORDER BY centrality_score DESC
LIMIT 5
```
""",
    },
    {
        "intent": (
            "(Analisis graf) mendapatkan komunitas entitas (community detection: "
            "louvain modularity)"
        ),
        "query": ("apa saja komunitas yang terbentuk dari semua peraturan yang ada?"),
        "cypher": """```cypher
CALL gds.louvain.stream('graph')
YIELD nodeId, communityId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, communityId
MATCH (r:Regulation)-[:HAS_ARTICLE]->(article1)-[:RELATED_TO|REFER_TO]-(article2)
RETURN
    r.title AS title,
    communityId,
    count(*) AS count
ORDER BY communityId, count DESC;
```
""",
    },
    {
        "intent": (
            "(Analisis graf) mendapatkan jalur atau hubungan terdekat (shortest path) "
            "di antara dua entitas"
        ),
        "query": (
            "Apa jalur hubungan terdekat antara pasal pasal 100 undang-undang / uu "
            "nomor 90 tahun 2020, dengan pasal 101 undang-undang / uu nomor 91 tahun "
            "2021?"
        ),
        "cypher": """```cypher
MATCH (r1:Regulation)-[:HAS_ARTICLE]->(source:Article)
MATCH (r2:Regulation)-[:HAS_ARTICLE]->(target:Article)
WHERE r1.type = 'UU' AND r1.number = 90 AND r1.year = 2020 AND source.number = '100'
    AND
    r2.type = 'UU' AND r2.number = 91 AND r2.year = 2021 AND target.number = '101'
CALL gds.shortestPath.dijkstra.stream('graph', {{sourceNode: source, targetNodes: target}})
// Catatan: Kurung kurawalnya 1 pasang saja, tidak 2 pasang
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
WITH [nodeId IN nodeIds | gds.util.asNode(nodeId).title] AS nodePathNames
UNWIND nodePathNames AS node
RETURN node
```
""",
    },
]
