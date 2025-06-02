from langchain_core.prompts import PromptTemplate


# CYPHER_GENERATION_TEMPLATE = """
# ## Task:
# - Generate Neo4j Cypher statement to query a graph database.

# ## Instructions:
# - Make a cypher code for user query or user questions.
# - Use only the provided relationship types and properties in the schema.
# - Do not use any other relationship types or properties that are not provided.

# ## User Question or Query:
# {question}

# ## Schema:
# {schema}

# ## Note:
# - Do not include any explanations or apologies in your responses.
# - Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
# - Do not include any text except the generated Cypher statement.

# ## Examples:
# Following are some examples that you can use as a reference to create Cypher code according to user questions.

# [Example 1]
# User query   : "Apa isi pasal 100 UU nomor 90 tahun 2020?"
# Cypher query : ```MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)
# WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
# RETURN a.text AS text```

# [Example 2]
# User query   : "Apa isi pasal selanjutnya dari pasal 100 undang-undang / UU nomor 90 tahun 2020?"
# Cypher query : ```MATCH (r:Regulation)-[:HAS_ARTICLE]->(a:Article)-[:NEXT_ARTICLE]->(next_article)
# WHERE r.type = 'UU' AND r.number = 90 AND r.year = 2020 AND a.number = '100'
# RETURN next_article.text AS text```

# ## Your Generated Cypher Statement

# """


# TODO
# Mainkan prefix nya,
# tambahkan judul ## Example dan beberapa keterangan tambahan sebelum melihat example,
FEW_SHOT_PREFIX_TEMPLATE = """## Examples:
Following are some examples that you can use as a reference to create Cypher code according to user questions.

"""


# TODO
# Tambahkan beberapa keterangan penting sebelum membuat Cypher Query,
# contoh nya jelaskan bahwa ada 3 jenis peraturan yang tersedia di database,
# yaitu Peraturan Menteri Komunikasi dan Informatika disingkat PERMENKOMINFO,
# Undang-Undang adalah UU, dan Peraturan Pemerintah adalah PP
CYPHER_GENERATION_TEMPLATE = """
## Task:
- Generate Neo4j Cypher statement to query a graph database.

## Instructions:
- Make a cypher code for user query or user questions.
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.

## Schema:

{schema}

## Note:
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

{example}

## Current User Question:

{question}
"""

# {examples}
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "example", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_FIX_TEMPLATE = """## Task:
- Address the Neo4j Cypher query error message.
- You are a Neo4j Cypher query expert responsible for correcting the provided `Generated Cypher Query` based on the provided `Cypher Error`.
- The `Cypher Error` explains why the `Generated Cypher Query` could not be executed in the database.

Generated Cypher Query:
{cypher_query}

Cypher Error:
{cypher_error}

## Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.

Neo4j Schema:
{schema}

Note:
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

You will output the `Corrected Cypher Query` wrapped in 3 backticks (```).
Do not include any text except the `Corrected Cypher Query`.

Remember to think step by step.

Corrected Cypher Query:
"""

CYPHER_FIX_PROMPT = PromptTemplate(
    input_variables=["schema", "cypher_query", "cypher_error"], template=CYPHER_FIX_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.

Follow this instruction when generating answers.
- Don't answer the user question if it not in legal/law scope, like science, math, social, ect.
- If the provided information is empty, say "Tidak dapat menemukan data yang sesuai dengan permintaan query".

Question:
{question}

Provided Information:
{context}

Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)
