"""LLM text generation evaluation workflow"""

import copy
import uuid
from typing import List
from tqdm import tqdm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset
from ..agent import create_agent

# from langgraph.graph import MessagesState
# from langgraph.prebuilt.chat_agent_executor import _get_prompt_runnable


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


def text_generation_eval_workflow(
    evaluation_dataset: EvaluationDataset,
    tool_names: List[str],
    generated_cypher_results: List[str],
    llm: BaseChatModel,
    verbose: bool = True,
) -> EvaluationDataset:
    """
    TODO: Docstring
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)
    agent = create_agent(model=llm, tools=[])

    for data, tool_name, cypher in tqdm(
        iterable=list(zip(evaluation_dataset, tool_names, generated_cypher_results)),
        desc="Running llm_text_generation on evaluation dataset",
        disable=not verbose,
    ):
        tool_call_id = f"run-{uuid.uuid4()}-0"

        # Formatting ToolMessage content
        if "text2cypher" in tool_name and cypher:
            tool_message_content = TEXT2CYPHER_CONTEXT_TEMPLATE.format(
                cypher=cypher, context="[" + " ".join(data.reference_contexts) + "]"
            )
        elif "vector" in tool_name:
            tool_message_content = VECTOR_CYPHER_CONTEXT_TEMPLATE.format(
                context="\n\n".join(data.reference_contexts)
            )
        else:
            print("Unknown `tool_name`, skipping `user_input`: " f"{data.user_input}")
            continue

        # Create fake "messages" state history
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

        response = agent.invoke(state)
        data.response = response["messages"][-1].content

    return evaluation_dataset
