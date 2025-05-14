from typing import (
    Callable,
    Optional,
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


def text2cypher_retriever_tool(
    cypher_llm: BaseLanguageModel,
    qa_llm: BaseLanguageModel,
    neo4j_graph: Neo4jGraph,
    *,
    skip_qa_llm: Optional[bool] = False,
    verbose: Optional[bool] = False
) -> Callable[[str], ToolMessage]:

    # Initialize the Neo4j Graph QA Chain
    text2cypher_chain = GraphCypherQAChainMod.from_llm(
        graph=neo4j_graph,
        qa_prompt=CYPHER_QA_PROMPT,
        cypher_generation_prompt=CYPHER_GENERATION_PROMPT,
        cypher_fix_prompt=CYPHER_FIX_PROMPT,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        exclude_types=["embedding"],
        return_intermediate_steps=not skip_qa_llm,
        return_direct=skip_qa_llm,
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

        result = text2cypher_chain.invoke(query)

        if not result["cypher"][-1]:
            print("cypher None")
            result["cypher"][-1] = AIMessage(
                content="Tidak dapat membuat kode Cypher Neo4j berdasarkan permintaan query"
            )

        if skip_qa_llm:
            artifact = {"is_context_fetched": bool(result["result"])}

            if not result["result"]:
                print("return_direct=True, but result=[]")
                result["result"] = ["Tidak dapat menemukan data yang sesuai dengan permintaan query"]

            response = (
                "### **Hasil Pembuatan Kode Cypher:**\n"
                f"{result['cypher'][-1].content}\n\n"
                "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
                f"{result['result']}"
            )
        
        else:
            artifact = {"is_context_fetched": bool(result["context"])}

            if not result["result"] or (hasattr(result["result"], "content") and result["result"].content == ""):
                print("return_direct=False, but result=''")
                result["result"] = AIMessage(
                    content="AAA Tidak dapat menemukan data yang sesuai dengan permintaan query"
                )

            response = (
                "### **Hasil Pembuatan Kode Cypher:**\n"
                f"{result['cypher'][-1].content}\n\n"
                "### **Hasil Eksekusi Kode Cypher ke Database:**\n"
                f"{result['result'].content}"
            )

        # TODO: Tentukan output artifact nya mau kaya bagaimana
        # return result
        return response, artifact
    
    return text2cypher