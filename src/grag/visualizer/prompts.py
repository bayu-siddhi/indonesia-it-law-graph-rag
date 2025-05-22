from langchain_core.prompts import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """
## Instruksi:
- Ubah kode Cypher Neo4j di bawah ini.
- Ubah sehingga output nya adalah mengembalikan (return) semua node dan relationship yang terdapat di dalam MATCH clause.
- Jika kode Cypher Neo4j tersebut telah mengembalikan semua node dan relationship yang terdapat di dalam MATCH clause, maka kembalikan kode Cypher Neo4j tersebut apa adanya tanpa perubahan.

## Catatan:
- Anda hanya perlu mengembalikan kode Cypher Neo4j, jangan mengembalikan penjelasan apa pun.
- Anda sangat tidak boleh memodifikasi arah relationship / edge yang terdapat di dalam MATCH clause sama sekali

## Format Hasil Akhir:
- Semua kode Cypher Neo4j yang Anda hasilkan harus berada di antara tanda \"```cypher\" dan \"```\"
- Buat hasil return value nya dalam bentuk: "RETURN *"
- Contoh format:

    ```cypher
    <hasil_modifikasi>
    RETURN *
    ```

## Kode Cypher Neo4j:
Berikut ini kode Cypher Neo4j harus Anda ubah:

    ```cypher
    {question}
    ```
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_FIX_TEMPLATE = """
## Task:
- Address the Neo4j Cypher query error message.
- You are a Neo4j Cypher query expert responsible for correcting the provided `Generated Cypher Query` based on the provided `Cypher Error`.
- The `Cypher Error` explains why the `Generated Cypher Query` could not be executed in the database.

Generated Cypher Query:
{cypher_query}

Cypher Error:
{cypher_error}

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
    input_variables=["cypher_query", "cypher_error"], template=CYPHER_FIX_TEMPLATE
)