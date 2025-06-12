import copy
import uuid
from typing import (
    List,
    Optional,
    Tuple
)
from tqdm import tqdm
from ragas import EvaluationDataset
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import BasePromptTemplate
from langchain_core.language_models import BaseChatModel
from ..retrievers import (
    create_text2cypher_retriever_tool,
    extract_cypher
)


def text2cypher_eval_workflow(
    evaluation_dataset: EvaluationDataset,
    *,
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseChatModel,
    qa_llm: BaseChatModel,
    embedder_model: Optional[Embeddings] = None,
    qa_prompt: Optional[BasePromptTemplate] = None,
    cypher_generation_prompt: Optional[BasePromptTemplate] = None,
    cypher_fix_prompt: Optional[BasePromptTemplate] = None,
    few_shot_prefix_template: Optional[str] = None,
    num_examples: int = 5,
    skip_qa_llm: bool = True,
    verbose: bool = True
) -> Tuple[EvaluationDataset, List[str]]:
    evaluation_dataset = copy.deepcopy(evaluation_dataset)

    text2cypher_retriever = create_text2cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        embedder_model=embedder_model,
        qa_prompt=qa_prompt,
        cypher_generation_prompt=cypher_generation_prompt,
        cypher_fix_prompt=cypher_fix_prompt,
        few_shot_prefix_template=few_shot_prefix_template,
        num_examples=num_examples,
        skip_qa_llm=skip_qa_llm,
        verbose=False
    )

    generated_cypher_results = []

    for data in tqdm(
        iterable=evaluation_dataset,
        desc="Running text2cypher_retriever on evaluation dataset",
        disable=not verbose
    ):
        tool_result = text2cypher_retriever.invoke(
            ToolCall(
                name=text2cypher_retriever.model_dump()["name"],
                args={"query": data.user_input},
                id=f"run-{uuid.uuid4()}-0",  # required
                type="tool_call"             # required
            )
        )

        generated_cypher_results.append(
            extract_cypher(tool_result.content)
        )
        
        retrieved_contexts = []
        for context in tool_result.artifact["context"]:
            retrieved_contexts.append(str(context))
        
        data.retrieved_contexts = retrieved_contexts
    
    return evaluation_dataset, generated_cypher_results
