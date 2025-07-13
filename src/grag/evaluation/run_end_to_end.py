"""End-to-end Graph-RAG workflow"""

import os
import copy
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from tqdm import tqdm
import pandas as pd
from neo4j import RoutingControl
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.tool_node import ToolNode
import langchain_core.messages as langchain_m
import ragas.messages as ragas_m
from ragas import EvaluationDataset, MultiTurnSample
from ..workflow import create_graph_rag_workflow
from ..fallback import BaseFallbackToolCalling
from ...prep.encodings import REGULATION_CODES


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
    ai_message = None
    tool_messages = []

    for current_message in messages:
        if isinstance(current_message, langchain_m.AIMessage):
            if ai_message is None:
                ai_message = current_message
            else:
                for tool_call, tool_message in zip(
                    ai_message.tool_calls, tool_messages
                ):
                    multi_turn_sample.user_input.append(
                        ragas_m.AIMessage(
                            content=str(ai_message.content),
                            tool_calls=[
                                ragas_m.ToolCall(
                                    name=tool_call["name"], args=tool_call["args"]
                                )
                            ],
                        )
                    )

                    multi_turn_sample.user_input.append(
                        ragas_m.ToolMessage(content=tool_message.content)
                    )
                ai_message = current_message
                tool_messages = []
        elif isinstance(current_message, langchain_m.ToolMessage):
            tool_messages.append(current_message)
    # For last AIMessagee
    multi_turn_sample.user_input.append(ragas_m.AIMessage(content=ai_message.content))

    # If no tool call and tool message, then create dummy tool call
    if len(multi_turn_sample.user_input) <= 2:
        multi_turn_sample.user_input.insert(
            1,
            ragas_m.AIMessage(
                content="",
                tool_calls=[ragas_m.ToolCall(name="no_tool_call", args={"query": ""})],
            ),
        )
        multi_turn_sample.user_input.insert(2, ragas_m.ToolMessage(content=""))


def _save_checkpoint(
    dataset: Tuple[EvaluationDataset, EvaluationDataset],
    experiment_name: str,
    checkpoint_path: str,
    checkpoint_num: int,
) -> None:
    """
    Saves a checkpoint of the evaluation dataset to a JSON file.

    Args:
        dataset (Tuple[EvaluationDataset, EvaluationDataset]): A tuple containing 
            the single-turn and multi-turn EvaluationDatasets.
        experiment_name (str): The name of the experiment.
        checkpoint_path (str): The path to the directory where the checkpoint file 
            should be saved.
        checkpoint_num (int): A number to append to the filename (e.g. step number).

    Returns:
        None
    """
    single_turn_result, multi_turn_result = dataset

    single_df = single_turn_result.to_pandas()
    single_df = single_df.loc[:, single_df.columns != "user_input"]

    single_df_cols = [
        "response",
        "reference",
        "retrieved_contexts",
        "reference_contexts",
    ]

    multi_df_cols = ["user_input", "reference_tool_calls"]
    multi_df = multi_turn_result.to_pandas().loc[:, multi_df_cols]
    multi_df["reference_tool_calls"] = multi_df["reference_tool_calls"].apply(
        lambda x: x[0]["name"]
    )

    result_df = pd.concat([single_df, multi_df], axis=1)
    result_df = result_df.loc[
        :, multi_df_cols[0:1] + single_df_cols + multi_df_cols[1:2]
    ]

    result_df.to_json(
        os.path.join(checkpoint_path, f"{experiment_name}_{checkpoint_num}.json"),
        orient="records",
    )


def run_end_to_end_graph_rag_workflow(
    single_turn_evaluation_dataset: EvaluationDataset,
    multi_turn_evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    model: BaseChatModel,
    neo4j_graph: Neo4jGraph,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    checkpoint_path: str,
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None,
    total_definition_limit: int = 5,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, EvaluationDataset]:
    """
    Runs the end-to-end Graph-RAG workflow, evaluating performance with both 
    single-turn and multi-turn datasets.

    This function orchestrates the complete Graph-RAG (Retrieval Augmented 
    Generation) workflow, starting from retrieving relevant context using a 
    combination of graph-based and semantic search, and ending with generating 
    a final answer using a language model. The process is evaluated using both 
    single-turn and multi-turn evaluation datasets, and a fallback tool calling 
    mechanism is incorporated. Periodic checkpoints are saved to enable restarting 
    interrupted evaluations.

    Args:
        single_turn_evaluation_dataset (EvaluationDataset): The EvaluationDataset 
            object containing single-turn questions and answers.
        multi_turn_evaluation_dataset (EvaluationDataset): The EvaluationDataset 
            object containing multi-turn conversations.
        experiment_name (str): The name of the experiment.
        model (BaseChatModel): The language model used for generating final answers.
        neo4j_graph (Neo4jGraph): The Neo4j graph instance for graph-based retrieval.
        tools (Union[Sequence[Union[BaseTool, Callable]], ToolNode]): The list of 
            tools that the agent can use.
        checkpoint_path (str): The directory to save checkpoint files to during 
            long running processes.
        fallback_tool_calling_cls (Optional[Type[BaseFallbackToolCalling]], optional): 
            An optional custom fallback tool calling class. Defaults to None.
        total_definition_limit (int, optional): The maximum number of definitions 
            to retrieve. Defaults to 5.
        verbose (bool, optional): Whether to display progress information. Defaults 
            to True.

    Returns:
        results (Tuple[EvaluationDataset, EvaluationDataset]): A tuple containing 
            the modified single-turn and multi-turn EvaluationDataset objects, 
            enriched with generated responses and retrieved contexts.
    """
    single_turn_evaluation_dataset = copy.deepcopy(single_turn_evaluation_dataset)
    multi_turn_evaluation_dataset = copy.deepcopy(multi_turn_evaluation_dataset)

    workflow = create_graph_rag_workflow(
        model=model, tools=tools, fallback_tool_calling_cls=fallback_tool_calling_cls
    )

    counter = 0

    for single, multi in tqdm(
        iterable=list(
            zip(single_turn_evaluation_dataset, multi_turn_evaluation_dataset)
        ),
        desc=f"Running end-to-end graph_rag: `{experiment_name}`",
        disable=not verbose,
    ):
        if counter % 3 == 0 and counter != 0:
            _save_checkpoint(
                dataset=(single_turn_evaluation_dataset, multi_turn_evaluation_dataset),
                experiment_name=experiment_name,
                checkpoint_path=checkpoint_path,
                checkpoint_num=counter,
            )

        # Running Graph-RAG
        response = {"messages": []}
        for s in workflow.stream({"messages": single.user_input}, stream_mode="values"):
            message = s["messages"][-1]
            response["messages"].append(message)

        _convert_to_ragas_messages(response["messages"], multi)

        single.response = str(response["messages"][-1].content)
        single.retrieved_contexts = []

        tool_messages = [
            (message.name, message.artifact)
            for message in response["messages"]
            if (
                isinstance(message, langchain_m.ToolMessage)
                and isinstance(message.artifact, dict)
            )
        ]

        if tool_messages:
            for name, artifact in tool_messages:
                if (
                    "text2cypher" in name
                    and isinstance(artifact, dict)
                    and artifact.get("is_context_fetched")
                    and artifact.get("context")
                ):
                    retrieved_contexts = []
                    for context in artifact["context"]:
                        retrieved_contexts.append(str(context))
                    single.retrieved_contexts += retrieved_contexts

                    continue
                if (
                    "vector" in name
                    and isinstance(artifact, dict)
                    and artifact.get("is_context_fetched")
                    and artifact.get("node_ids")
                ):
                    article_node_ids = artifact["node_ids"][:-total_definition_limit]
                    definition_node_ids = artifact["node_ids"][-total_definition_limit:]
                    art_definition_node_ids = []

                    for def_node_id in definition_node_ids:
                        new_definition_id = int(
                            str(def_node_id)[:-6]
                            + REGULATION_CODES["section"]["article"]
                            + "00100"
                        )
                        art_definition_node_ids.append(new_definition_id)

                    current_article_node_ids = list(
                        dict.fromkeys(article_node_ids + art_definition_node_ids)
                    )

                    query_result = neo4j_graph._driver.execute_query(
                        query_="""
                            UNWIND $node_ids AS node_id
                            MATCH (n)
                            WHERE n.id = node_id
                            RETURN n.text AS text
                        """,
                        parameters_={"node_ids": current_article_node_ids},
                        routing_=RoutingControl.READ,
                        database_=neo4j_graph._database,
                    )

                    retrieved_contexts = []
                    for record in query_result.records:
                        retrieved_contexts.append(record["text"])
                    single.retrieved_contexts += retrieved_contexts

                    continue

                single.retrieved_contexts += [
                    "Tidak dapat menemukan data yang sesuai dengan permintaan query"
                ]
        else:
            single.retrieved_contexts += [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]

        counter += 1

    return single_turn_evaluation_dataset, multi_turn_evaluation_dataset
