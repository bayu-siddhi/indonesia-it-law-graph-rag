"""LLM tools selection workflow"""

import copy
from typing import (
    Callable,
    List,
    Sequence,
    Union,
)
from tqdm import tqdm
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.tool_node import ToolNode
import langchain_core.messages as langchain_m
import ragas.messages as ragas_m
from ragas import EvaluationDataset, MultiTurnSample
from ..workflow import create_graph_rag_workflow


def _convert_to_ragas_messages(
    messages: List[langchain_m.BaseMessage], multi_turn_sample: MultiTurnSample
) -> None:
    """
    Converts a list of langchain messages to Ragas messages for evaluation, 
    enriching the MultiTurnSample with tool call information.

    Args:
        messages (List[langchain_m.BaseMessage]): A list of langchain messages 
            representing the conversation history.
        multi_turn_sample (MultiTurnSample): The MultiTurnSample object to enrich 
            with tool call information.
    
    Returns:
        None
    """
    for current_message in messages:
        if isinstance(current_message, langchain_m.AIMessage):
            for tool_call in current_message.tool_calls:
                multi_turn_sample.user_input.append(
                    ragas_m.AIMessage(
                        content=str(current_message.content),
                        tool_calls=[
                            ragas_m.ToolCall(
                                name=tool_call["name"], args=tool_call["args"]
                            )
                        ],
                    )
                )

    # If no tool call and tool message, then create dummy tool call
    if not messages[1].tool_calls:
        multi_turn_sample.user_input.insert(
            1,
            ragas_m.AIMessage(
                content="",
                tool_calls=[ragas_m.ToolCall(name="no_tool_call", args={"query": ""})],
            ),
        )


def run_tools_selection_workflow(
    multi_turn_evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    model: BaseChatModel,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    verbose: bool = True,
) -> EvaluationDataset:
    """
    Runs a tools selection workflow to evaluate the agent's ability to 
    choose the correct tools.

    Args:
        multi_turn_evaluation_dataset (EvaluationDataset): The EvaluationDataset 
            object containing multi-turn conversations.
        experiment_name (str): The name of the experiment.
        model (BaseChatModel): The language model to use for the agent.
        tools (Union[Sequence[Union[BaseTool, Callable]], ToolNode]): The list 
            of tools the agent can use.
        verbose (bool, optional): Whether to display progress information. 
            Defaults to True.

    Returns:
        result (EvaluationDataset): The modified EvaluationDataset object 
            with ragas messages containing tool call information.
    """
    multi_turn_evaluation_dataset = copy.deepcopy(multi_turn_evaluation_dataset)
    workflow = create_graph_rag_workflow(model=model, tools=tools)

    for data in tqdm(
        iterable=multi_turn_evaluation_dataset,
        desc=f"Running retrieval tool selection: `{experiment_name}`",
        disable=not verbose,
    ):
        response = {"messages": []}

        for s in workflow.stream(
            {"messages": data.user_input[0].content}, stream_mode="values"
        ):
            message = s["messages"][-1]
            response["messages"].append(message)
            if len(response["messages"]) == 2:
                break

        _convert_to_ragas_messages(response["messages"], data)

    return multi_turn_evaluation_dataset
