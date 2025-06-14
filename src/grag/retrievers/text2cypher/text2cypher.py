"""Text2Cypher retriever tool"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import BasePromptTemplate, FewShotPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_neo4j import Neo4jGraph
from ..models import SimpleQueryInput
from .prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_FIX_PROMPT,
    CYPHER_QA_PROMPT,
    FEW_SHOT_PREFIX_TEMPLATE,
)
from .examples import text2cypher_example, text2cypher_example_prompt
from .cypher_mod import GraphCypherQAChainMod


def _exclude_keys_from_data(data: Any, excluded_keys: List[str]) -> Any:
    """
    Recursively excludes keys from dictionaries and lists.

    Args:
        data (Any): Any nested Python structure (dict, list, or scalar).
        excluded_keys (List[str]): Keys to remove from dictionaries.

    Returns:
        data (Any): Cleaned data with excluded keys removed.
    """
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key not in excluded_keys:
                new_data[key] = _exclude_keys_from_data(value, excluded_keys)
        return new_data
    if isinstance(data, list):
        new_data = []
        for item in data:
            new_data.append(_exclude_keys_from_data(item, excluded_keys))
        return new_data
    return data


def _tool_result_formatter(
    result: Dict[str, Any], add_context_to_artifact: bool, skip_qa_llm: bool
) -> Tuple[str, Dict[str, Any]]:
    """
    Formats the result from the text2cypher chain into a tool output.

    Args:
        result (Dict[str, Any]): The result dictionary from the text2cypher chain,
            including Cypher generation, execution, and optional QA result.
        add_context_to_artifact (bool): Wheter to add context result to artifact
            or not.
        skip_qa_llm (bool): Flag to skip the QA LLM (query explanation phase)
            if True.

    Returns:
        (response, artifact) (Tuple[str, Dict[str, Any]]): A formatted response
            and a metadata artifact dictionary.
    """
    default_content = "Tidak dapat menemukan data yang sesuai dengan permintaan query"
    artifact = {
        "is_context_fetched": False,
        "context": [],
        # "cypher_gen_usage_metadata": {},
        # "qa_usage_metadata": {}
    }

    # Handle cases where Cypher generation fails
    if not result["cypher"][-1]:
        print("cypher None")
        result["cypher"][-1] = AIMessage(
            content="Tidak dapat membuat kode Cypher Neo4j berdasarkan permintaan query"
        )

    # Format response differently based on whether the QA LLM was used
    if skip_qa_llm:
        artifact["is_context_fetched"] = bool(result["result"])
        if result["result"]:
            if add_context_to_artifact:
                artifact["context"] = result["result"]

            temp_result = result["result"]
            result["result"] = ""
            for context in temp_result:
                result["result"] += str(context) + "\n"
        else:
            result["result"] = default_content

        cypher = result["cypher"][-1].content.replace("`", r"\`")

        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{cypher}\n\n"
            "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
            f"{result['result']}"
        )

    else:
        if not result["result"] or result["result"].content == "":
            result["result"] = AIMessage(content=default_content)

        if bool(result["context"]):
            artifact["is_context_fetched"] = bool(result["context"])
            if add_context_to_artifact:
                artifact["context"] = result["context"]

        # if result["result"].usage_metadata:
        #     for key, value in result["result"].usage_metadata.items():
        #         artifact["qa_usage_metadata"][key] = value

        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{result['cypher'][-1].content}\n\n"
            "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
            f"{result['result'].content}"
        )

    # # Aggregate usage metadata from all Cypher generation steps
    # for cypher_gen_ai_message in result["cypher"]:
    #     if cypher_gen_ai_message.usage_metadata:
    #         # TERNYATA GARA2 usage_metadata DEFAULTNYA ADALAH NONE
    #         # MAKANYA ERROR PAS DIKASIH FUNCITON .items()
    #         for key, value in cypher_gen_ai_message.usage_metadata.items():
    #             if isinstance(value, (int, float)):
    #                 artifact["cypher_gen_usage_metadata"][key] = \
    #                     artifact["cypher_gen_usage_metadata"].get(key, 0) + value
    #             else:
    #                 artifact["cypher_gen_usage_metadata"][key] = value

    return response, artifact


def create_text2cypher_retriever_tool(
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseChatModel,
    qa_llm: BaseChatModel,
    embedder_model: Optional[Embeddings] = None,
    *,
    qa_prompt: Optional[BasePromptTemplate] = None,
    cypher_generation_prompt: Optional[BasePromptTemplate] = None,
    cypher_fix_prompt: Optional[BasePromptTemplate] = None,
    few_shot_prefix_template: Optional[str] = None,
    few_shot_num_examples: Optional[int] = None,
    add_context_to_artifact: bool = False,
    skip_qa_llm: bool = False,
    verbose: bool = False,
) -> Callable[[str], ToolMessage]:
    """
    Create a text-to-Cypher retriever tool using Neo4j and LLMs.

    This tool converts natural language queries into Cypher code using LLM
    prompting, executes the generated Cypher on a Neo4j database, and
    optionally explains the result using a second QA LLM.

    The tool supports few-shot prompting with semantic similarity if an
    embedding model is provided.

    Args:
        neo4j_graph (Neo4jGraph): Neo4j graph interface.
        cypher_llm (BaseChatModel): LLM for Cypher generation and error
            correction.
        qa_llm (BaseChatModel): LLM for formulating the final answer from
            query results.
        embedder_model (Optional[Embeddings]): Embedding model for semantic
            few-shot example selection (required if examples used).
        qa_prompt (Optional[BasePromptTemplate]): Prompt template for the
            QA LLM.
        cypher_generation_prompt (Optional[BasePromptTemplate]): Prompt
            template for initial Cypher generation.
        cypher_fix_prompt (Optional[BasePromptTemplate]): Prompt template
            for Cypher error correction.
        few_shot_prefix_template (Optional[str]): Template string prefix for
            few-shot examples (required if examples used).
        few_shot_num_examples (int): Number of few-shot examples to include (if enabled).
        skip_qa_llm (bool): If True, skips the QA stage and returns raw data.
        verbose (bool): If True, enables verbose logging.

    Returns:
        Callable[[str], ToolMessage]: A LangChain tool callable that takes a query
            string and returns a `ToolMessage` with results or error.

    Returns:
        text2cypher_retriever (Callable[[str], ToolMessage]): A LangChain
            tool that receives a user query and returns result.
    """
    # Validated input data
    if qa_prompt is None:
        qa_prompt = CYPHER_QA_PROMPT
    if cypher_generation_prompt is None:
        cypher_generation_prompt = CYPHER_GENERATION_PROMPT
    if cypher_fix_prompt is None:
        cypher_fix_prompt = CYPHER_FIX_PROMPT
    if few_shot_prefix_template is None:
        few_shot_prefix_template = FEW_SHOT_PREFIX_TEMPLATE
    if few_shot_num_examples is None:
        few_shot_num_examples = 5

    # Create the text2cypher chain
    text2cypher = GraphCypherQAChainMod.from_llm(
        graph=neo4j_graph,
        qa_prompt=qa_prompt,
        cypher_generation_prompt=cypher_generation_prompt,
        cypher_fix_prompt=cypher_fix_prompt,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        exclude_types=["embedding"],
        return_intermediate_steps=not skip_qa_llm,
        return_direct=skip_qa_llm,
        allow_dangerous_requests=True,
        verbose=verbose,
    )

    # Add few-shot prompting with semantic similarity example selection
    # if an embedder model is provided
    if embedder_model is not None:
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=text2cypher_example,
            embeddings=embedder_model,
            vectorstore_cls=InMemoryVectorStore,
            k=few_shot_num_examples,
        )

        few_shot_prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=text2cypher_example_prompt,
            prefix=few_shot_prefix_template,
            suffix="",
            input_variables=["question"],
        )

        few_shot_selection = RunnableLambda(
            lambda x: {
                "query": x,
                "example": few_shot_prompt_template.invoke(
                    {"question": x.lower()}
                ).to_string(),
            }
        )

        text2cypher_chain = few_shot_selection | text2cypher
    else:
        text2cypher_chain = text2cypher

    @tool(args_schema=SimpleQueryInput, response_format="content_and_artifact")
    def text2cypher_retriever(
        query: str,
        # example: str = ""
    ) -> ToolMessage:
        """
        **`text2cypher_retriever`**:
        *   **Fungsi:** Menerjemahkan pertanyaan bahasa alami menjadi kueri Cypher untuk mengambil informasi spesifik dan terstruktur langsung dari database graf hukum Neo4j.
        *   **Kapan Digunakan:** Gunakan alat ini ketika pertanyaan pengguna secara spesifik menanyakan:
            *   Isi dari pasal, bagian, atau properti tertentu dari suatu peraturan.
            *   Hubungan antar entitas hukum (misalnya, peraturan mana yang mencabut peraturan lain, pasal mana yang saling merujuk, dan jenis hubungan lainnya).
            *   Struktur hierarki peraturan atau komponennya.
            *   Properti atau fakta spesifik tentang entitas yang diketahui dalam graf (misalnya, tanggal berlaku, nomor, tahun).
            *   **Informasi yang membutuhkan analisis graf**, seperti mencari pasal paling berpengaruh, entitas yang berperan sebagai penjembatan, rekomendasi pasal, deteksi komunitas tertentu, jalur terpendek antar entitas, dan analisis graf lainnya.
        *   **Singkatnya:** Gunakan untuk kueri yang tepat mengenai struktur graf, hubungan, fakta spesifik, atau analisis graf, di mana jawabannya dapat ditemukan dengan mengueri skema graf yang telah ditentukan.
        """

        start_time = time.time()

        result = text2cypher_chain.invoke(query)
        result = _exclude_keys_from_data(result, excluded_keys=["embedding"])
        response, artifact = _tool_result_formatter(
            result, add_context_to_artifact, skip_qa_llm
        )

        # artifact["cypher_gen_usage_metadata"]["model"] = cypher_llm.model
        # if not skip_qa_llm: artifact["qa_usage_metadata"]["model"] = qa_llm.model
        artifact["run_time"] = time.time() - start_time

        return response, artifact
        # return result, artifact

    return text2cypher_retriever
