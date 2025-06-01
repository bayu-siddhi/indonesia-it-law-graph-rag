import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple
)
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    BasePromptTemplate,
    FewShotPromptTemplate
)
from langchain_core.messages import (
    AIMessage,
    ToolMessage
)
from langchain_neo4j import Neo4jGraph
# from ..models import SimpleQuery
from .models import Text2CypherInput
from .prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_FIX_PROMPT,
    CYPHER_QA_PROMPT,
)
from .examples import (
    text2cypher_example,
    text2cypher_example_prompt
)
from .cypher_mod import GraphCypherQAChainMod


def _tool_result_formatter(
    result: Dict[str, Any],
    skip_qa_llm: bool
) -> Tuple[str, Dict[str, Any]]:
    artifact = {
        "is_context_fetched": False,
        "cypher_gen_usage_metadata": {},
        "qa_usage_metadata": {}
    }

    if not result["cypher"][-1]:
        print("cypher None")
        result["cypher"][-1] = AIMessage(
            content="Tidak dapat membuat kode Cypher Neo4j berdasarkan permintaan query"
        )

    if skip_qa_llm:
        artifact["is_context_fetched"] = bool(result["result"])
        if not result["result"]:
            result["result"] = [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]
        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{result['cypher'][-1].content}\n\n"
            "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
            f"{result['result']}"
        )
    
    else:
        if not result["result"] or result["result"].content == "":
            result["result"] = AIMessage(
                content="Tidak dapat menemukan data yang sesuai dengan permintaan query"
            )

        artifact["is_context_fetched"] = bool(result["context"])
        if result["result"].usage_metadata:
            for key, value in result["result"].usage_metadata.items():
                artifact["qa_usage_metadata"][key] = value

        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{result['cypher'][-1].content}\n\n"
            "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
            f"{result['result'].content}"
        )
    
    for cypher_gen_ai_message in result["cypher"]:
        for key, value in cypher_gen_ai_message.usage_metadata.items():
            if isinstance(value, (int, float)):
                artifact["cypher_gen_usage_metadata"][key] = \
                    artifact["cypher_gen_usage_metadata"].get(key, 0) + value
            else:
                artifact["cypher_gen_usage_metadata"][key] = value
    
    return response, artifact


def _exclude_keys_from_data(
    data: Any,
    excluded_keys: List[str]
) -> Any:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key not in excluded_keys:
                new_data[key] = _exclude_keys_from_data(value, excluded_keys)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for item in data:
            new_data.append(_exclude_keys_from_data(item, excluded_keys))
        return new_data
    else:
        return data


def create_text2cypher_retriever_tool(
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseLanguageModel,
    qa_llm: BaseLanguageModel,
    embedder_model: Optional[Embeddings] = None,
    qa_prompt: Optional[BasePromptTemplate] = None,
    cypher_generation_prompt: Optional[BasePromptTemplate] = None,
    cypher_fix_prompt: Optional[BasePromptTemplate] = None,
    num_examples: int = 5,
    skip_qa_llm: bool = False,
    verbose: bool = False
) -> Callable[[str], ToolMessage]:
    
    if qa_prompt is None:
        qa_prompt = CYPHER_QA_PROMPT
    if cypher_generation_prompt is None:
        cypher_generation_prompt = CYPHER_GENERATION_PROMPT
    if cypher_fix_prompt is None:
        cypher_fix_prompt = CYPHER_FIX_PROMPT

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
        verbose=verbose
    )

    if embedder_model is not None:
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=text2cypher_example,
            embeddings=embedder_model,
            vectorstore_cls=InMemoryVectorStore,
            k=num_examples
        )

        few_shot_prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=text2cypher_example_prompt,
            prefix="",  # Mainkan prefix nya, tambahkan judul ## Example dan beberapa keterangan tambahan sebelum melihat example, contoh nya jelaskan bahwa ada 3 jenis peraturan yang tersedia di database, yaitu Peraturan Menteri Komunikasi dan Informatika disingkat PERMENKOMINFO, Undang-Undang adalah UU, dan Peraturan Pemerintah adalah PP
            suffix="",
            input_variables=["question"],
        )

        few_shot_selection = RunnableLambda(
            lambda x: {
                "query": x,
                "example": few_shot_prompt_template.invoke({
                    "question": x.lower()
                }).to_string()
            }
        )

        text2cypher_chain = few_shot_selection | text2cypher
    else:
        text2cypher_chain = text2cypher

    @tool(
        args_schema=SimpleQuery,
        response_format="content_and_artifact"
    )
    def text2cypher_retriever(
        query: str,
        # example: str = ""
    ) -> ToolMessage:
        """Text2Cypher Tool"""
        # TODO: PERBAIKI DESKRIPSI TOOL

        start_time = time.time()
        
        # selected_examples = "INI CONTOH "  # NANTI HAPUS
        # result = text2cypher_chain.invoke({"query": query, "example": selected_examples})
        result = text2cypher_chain.invoke(query)
        result = _exclude_keys_from_data(result, excluded_keys=["embedding"])
        response, artifact = _tool_result_formatter(result, skip_qa_llm)
        
        artifact["cypher_gen_usage_metadata"]["model"] = cypher_llm.model
        if not skip_qa_llm: artifact["qa_usage_metadata"]["model"] = qa_llm.model
        artifact["run_time"] = time.time() - start_time
        
        return response, artifact
    
    return text2cypher_retriever
