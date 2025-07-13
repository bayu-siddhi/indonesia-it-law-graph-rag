"""LLM text generation workflow"""

import copy
import uuid
from typing import List, Optional
from tqdm import tqdm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import Prompt
from ragas import EvaluationDataset
from ..agent import create_agent


VECTOR_CYPHER_CONTEXT_TEMPLATE = """
## **Daftar Pasal Peraturan Perundang-Undangan yang (Mungkin) Relevan untuk Menjawab Kueri:**
--------------------------------------------------------------------------------

{context}
""".strip()


TEXT2CYPHER_CONTEXT_TEMPLATE = """
### **Hasil Pembuatan Kode Cypher:**
```{cypher}```

### **Hasil Eksekusi Kode Cypher ke Database:**
{context}
""".strip()


def run_text_generation_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    expected_tool_call_names: List[str],
    generated_cypher_results: List[str],
    llm: BaseChatModel,
    prompt: Optional[Prompt] = None,
    verbose: bool = True,
) -> EvaluationDataset:
    """
    Runs a text generation workflow to evaluate the LLM's ability to 
    generate responses based on retrieved contexts.

    Args:
        evaluation_dataset (EvaluationDataset): The EvaluationDataset 
            object containing questions and retrieved contexts for evaluation.
        experiment_name (str): The name of the experiment.
        expected_tool_call_names (List[str]): A list of expected tool 
            call names corresponding to each question.
        generated_cypher_results (List[str]): A list of generated Cypher 
            queries (if applicable) corresponding to each question.
        llm (BaseChatModel): The language model used for text generation.
        prompt (Optional[Prompt], optional): An optional prompt to use for 
            the LLM. Defaults to None.
        verbose (bool, optional): Whether to display progress information. 
            Defaults to True.

    Returns:
        result (EvaluationDataset): The modified EvaluationDataset object 
            with the generated responses added to each data point.
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)
    agent = create_agent(model=llm, tools=[], prompt=prompt)

    for data, tool_name, cypher in tqdm(
        iterable=list(
            zip(evaluation_dataset, expected_tool_call_names, generated_cypher_results)
        ),
        desc=f"Running llm_text_generation: `{experiment_name}`",
        disable=not verbose,
    ):
        tool_call_id = f"run-{uuid.uuid4()}-0"

        # Formatting ToolMessage content
        if "text2cypher" in tool_name and cypher:
            tool_message_content = TEXT2CYPHER_CONTEXT_TEMPLATE.format(
                cypher=cypher, context="[" + " ".join(data.retrieved_contexts) + "]"
            )
        elif "vector" in tool_name:
            tool_message_content = VECTOR_CYPHER_CONTEXT_TEMPLATE.format(
                context="\n\n".join(data.retrieved_contexts)
            )
        else:
            print("Unknown `tool_name`, skipping `user_input`: " f"{data.user_input}")
            continue

        tool_message_content += (
            "\n\nJawab apa adanya berdasarkan data atau informasi yang telah diambil ini."
        )

        # Create messages state history
        state = {
            "messages": [
                HumanMessage(content=data.user_input),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"query": data.user_input},
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=tool_message_content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                ),
            ]
        }

        # Run LLM
        response = agent.invoke(state)
        data.response = str(response["messages"][-1].content)

    return evaluation_dataset
