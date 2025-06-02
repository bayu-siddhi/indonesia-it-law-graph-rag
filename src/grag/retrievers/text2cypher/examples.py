from langchain_core.prompts import PromptTemplate


# TODO: Ubah semua ke bahasa Indonesia
text2cypher_example_prompt = PromptTemplate.from_template("Question: {query}\nNeo4j Cypher: {cypher}")

text2cypher_example = [
    {
        "query": "isi pasal 100 UU undang undang nomor 90 tahun 2020",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
RETURN a.text AS text
```
"""
    },
    {
        "query": "apa isi konten dari pasal 50 pp (peraturan pemerintah) no. 70 tahun 2015",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'PP' AND r.number = 70 AND r.year = 2015 AND a.number = '50'
RETURN a.text AS text
```
"""
    },
    {
        "query": "dapatkan isi pasal 90 permenkominfo (peraturan menteri kominfo) no 120 tahun 2021",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
WHERE r.type = 'PERMENKOMINFO' AND r.number = 120 AND r.year = 2021 AND a.number = '90'
RETURN a.text AS text
```
"""
    },
    {
        "query": "apa pasal selanjutnya dari pasal 9 undang-undang / uu nomor 10 tahun 2010",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:NEXT_ARTICLE]->(next_article)
WHERE r.type = 'UU' AND r.number = 10 AND r.year = 2010 AND a.number = '9'
RETURN next_article.text AS text
```
"""
    },
    {
        "query": "carikan pasal setelah pasal 99 pp / peraturan pemerintah no 5 tahun 2000",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:NEXT_ARTICLE]->(next_article)
WHERE r.type = 'PP' AND r.number = 5 AND r.year = 2000 AND a.number = '99'
RETURN next_article.text AS text
```
"""
    },
    {
        "query": "apa isi pasal habis pasal 15 peraturan menteri kominfo (permenkominfo) no. 10 tahun 2008",
        "cypher": """```cypher
MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:NEXT_ARTICLE]->(next_article)
WHERE r.type = 'PERMENKOMINFO' AND r.number = 10 AND r.year = 2008 AND a.number = '15'
RETURN next_article.text AS text
```
"""
    },
]
