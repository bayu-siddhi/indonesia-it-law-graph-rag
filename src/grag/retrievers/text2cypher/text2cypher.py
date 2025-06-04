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
from ..models import SimpleQueryInput
from .prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_FIX_PROMPT,
    CYPHER_QA_PROMPT,
    FEW_SHOT_PREFIX_TEMPLATE
)
from .examples import (
    text2cypher_example,
    text2cypher_example_prompt
)
from .cypher_mod import GraphCypherQAChainMod


def _exclude_keys_from_data(
    data: Any,
    excluded_keys: List[str]
) -> Any:
    """
    Recursively excludes keys from dictionaries and lists.

    Args:
        data (Any): Any nested Python structure (dict, list, or scalar).
        excluded_keys (List[str]): Keys to remove from dictionaries.

    Returns:
        Any: Cleaned data with excluded keys removed.
    """
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


def _tool_result_formatter(
    result: Dict[str, Any],
    skip_qa_llm: bool
) -> Tuple[str, Dict[str, Any]]:
    """
    Formats the result from the text2cypher chain into a tool output.

    Args:
        result (Dict[str, Any]): The result dictionary from the text2cypher chain,
            including Cypher generation, execution, and optional QA result.
        skip_qa_llm (bool): Flag to skip the QA LLM (query explanation phase) 
            if True.

    Returns:
        Tuple[str, Dict[str, Any]]: A formatted response and a metadata artifact 
            dictionary.
    """
    artifact = {
        "is_context_fetched": False,
        "cypher_gen_usage_metadata": {},
        "qa_usage_metadata": {}
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
        if not result["result"]:
            result["result"] = [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]
        
        cypher = result['cypher'][-1].content.replace('`', '\`')
        
        response = (
            "### **Hasil Pembuatan Kode Cypher:**\n"
            f"{cypher}\n\n"
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

    # Aggregate usage metadata from all Cypher generation steps
    for cypher_gen_ai_message in result["cypher"]:
        for key, value in cypher_gen_ai_message.usage_metadata.items():
            if isinstance(value, (int, float)):
                artifact["cypher_gen_usage_metadata"][key] = \
                    artifact["cypher_gen_usage_metadata"].get(key, 0) + value
            else:
                artifact["cypher_gen_usage_metadata"][key] = value
    
    return response, artifact


def create_text2cypher_retriever_tool(
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseLanguageModel,
    qa_llm: BaseLanguageModel,
    embedder_model: Optional[Embeddings] = None,
    qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
    cypher_generation_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
    cypher_fix_prompt: BasePromptTemplate = CYPHER_FIX_PROMPT,
    few_shot_prefix_template: Optional[str] = FEW_SHOT_PREFIX_TEMPLATE,
    num_examples: int = 5,
    skip_qa_llm: bool = False,
    verbose: bool = False
) -> Callable[[str], ToolMessage]:
    """
    Create a text-to-Cypher retriever tool using Neo4j and LLMs.

    This tool converts natural language queries into Cypher code using LLM 
    prompting, executes the generated Cypher on a Neo4j database, and 
    optionally explains the result using a second QA LLM. The tool supports 
    few-shot prompting with semantic similarity if an embedding model is 
    provided.

    Args:
        neo4j_graph (Neo4jGraph): The Neo4j graph interface for running 
            Cypher queries.
        cypher_llm (BaseLanguageModel): LLM used to generate Cypher from 
            text input.
        qa_llm (BaseLanguageModel): LLM used to interpret and answer questions 
            based on results.
        embedder_model (Optional[Embeddings]): Model used for selecting 
            examples semantically.
        qa_prompt (Optional[BasePromptTemplate]): Prompt template for the 
            QA LLM.
        cypher_generation_prompt (Optional[BasePromptTemplate]): Prompt 
            template for generating Cypher.
        cypher_fix_prompt (Optional[BasePromptTemplate]): Prompt template 
            for fixing Cypher queries.
        num_examples (int): Number of few-shot examples to include in 
            prompting.
        skip_qa_llm (bool): Whether to skip the final QA stage and just 
            return raw data.
        verbose (bool): Whether to print debugging information.

    Returns:
        Callable[[str], ToolMessage]: A LangChain tool that receives a 
            user query and returns result.
    """
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
        verbose=verbose
    )

     # Add few-shot prompting with semantic similarity example selection
     # if an embedder model is provided
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
            prefix=few_shot_prefix_template,
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
        args_schema=SimpleQueryInput,
        response_format="content_and_artifact"
    )
    def text2cypher_retriever(
        query: str,
        # example: str = ""
    ) -> ToolMessage:
        """Text2Cypher Tool
        Jika pengguna meminta informasi tentang **isi pasal spesifik, struktur regulasi, pertimbangann hukum (consideration), dasar hukum (observation), hubungan antar pasal atau peraturan, atau apa pun yang dapat direpresentasikan sebagai Neo4j Cypher**, gunakan `text2cypher_retriever` ini.
        """
        # TODO: PERBAIKI DESKRIPSI TOOL

        start_time = time.time()

        result = text2cypher_chain.invoke(query)
        result = _exclude_keys_from_data(result, excluded_keys=["embedding"])
        response, artifact = _tool_result_formatter(result, skip_qa_llm)

        artifact["cypher_gen_usage_metadata"]["model"] = cypher_llm.model
        if not skip_qa_llm: artifact["qa_usage_metadata"]["model"] = qa_llm.model
        artifact["run_time"] = time.time() - start_time

        return response, artifact

    return text2cypher_retriever
