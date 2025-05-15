import time
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple
)
from langchain_core.tools import tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    ToolMessage
)
from langchain_neo4j import Neo4jGraph
from src.grag.retrievers.models import SimpleQuery
from src.grag.retrievers.text2cypher.prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_FIX_PROMPT,
    CYPHER_QA_PROMPT,
)
from src.grag.retrievers.text2cypher.cypher_mod import GraphCypherQAChainMod


def _tool_result_formatter(
    result: Dict[str, Any],
    is_skip_qa_llm: bool
) -> Tuple[str, Dict[str, Any]]:
    artifact = {
        "is_context_fetched": False,
        "cypher_gen_usage_metadata": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        },
        "qa_usage_metadata": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }

    if not result["cypher"][-1]:
        print("cypher None")
        result["cypher"][-1] = AIMessage(
            content="Tidak dapat membuat kode Cypher Neo4j berdasarkan permintaan query"
        )

    if is_skip_qa_llm:
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
                if key != "input_token_details":
                    artifact["qa_usage_metadata"][key] += value

        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{result['cypher'][-1].content}\n\n"
            "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
            f"{result['result'].content}"
        )
    
    for cypher_gen_ai_message in result["cypher"]:
        for key, value in cypher_gen_ai_message.usage_metadata.items():
            if key != "input_token_details":
                artifact["cypher_gen_usage_metadata"][key] += value
    
    return response, artifact


def create_text2cypher_retriever_tool(
    cypher_llm: BaseLanguageModel,
    qa_llm: BaseLanguageModel,
    neo4j_graph: Neo4jGraph,
    is_skip_qa_llm: Optional[bool] = False,
    verbose: Optional[bool] = False
) -> Callable[[str], ToolMessage]:
    text2cypher_chain = GraphCypherQAChainMod.from_llm(
        graph=neo4j_graph,
        qa_prompt=CYPHER_QA_PROMPT,
        cypher_generation_prompt=CYPHER_GENERATION_PROMPT,
        cypher_fix_prompt=CYPHER_FIX_PROMPT,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        exclude_types=["embedding"],
        return_intermediate_steps=not is_skip_qa_llm,
        return_direct=is_skip_qa_llm,
        allow_dangerous_requests=True,
        verbose=verbose
    )

    @tool(
        args_schema=SimpleQuery,
        response_format="content_and_artifact"
    )
    def text2cypher(
        query: str
    ) -> ToolMessage:
        """Text2Cypher Tool"""
        # TODO: PERBAIKI DESKRIPSI TOOL

        start_time = time.time()
        
        result = text2cypher_chain.invoke(query)
        response, artifact = _tool_result_formatter(result, is_skip_qa_llm)

        artifact["run_time"] = time.time() - start_time
        
        return response, artifact
    
    return text2cypher
