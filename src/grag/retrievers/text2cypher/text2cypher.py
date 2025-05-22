import time
from typing import (
    Any,
    Callable,
    Dict,
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
from ..models import SimpleQuery
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
    neo4j_graph: Neo4jGraph,
    embedder_model: Embeddings,
    cypher_llm: BaseLanguageModel,
    qa_llm: BaseLanguageModel,
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

    text2cypher_chain = GraphCypherQAChainMod.from_llm(
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

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=text2cypher_example,
        embeddings=embedder_model,
        vectorstore_cls=InMemoryVectorStore,
        k=num_examples
    )

    few_shot_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=text2cypher_example_prompt,
        suffix="",
        input_variables=["question"],
    )

    few_shot_selection_chain = RunnableLambda(
        lambda x: {
            "query": x,
            "example": few_shot_prompt_template.invoke({
                "question": x.lower()
            }).to_string()
        }
    )

    text2cypher_chain_with_few_shot = few_shot_selection_chain | text2cypher_chain

    @tool(
        args_schema=SimpleQuery,
        response_format="content_and_artifact"
    )
    def text2cypher(
        query: str,
        # example: str = ""
    ) -> ToolMessage:
        """Text2Cypher Tool"""
        # TODO: PERBAIKI DESKRIPSI TOOL

        start_time = time.time()
        
        # selected_examples = "INI CONTOH "  # NANTI HAPUS
        # result = text2cypher_chain.invoke({"query": query, "example": selected_examples})

        result = text2cypher_chain_with_few_shot.invoke(query)
        response, artifact = _tool_result_formatter(result, skip_qa_llm)

        artifact["run_time"] = time.time() - start_time
        
        return response, artifact
    
    return text2cypher
