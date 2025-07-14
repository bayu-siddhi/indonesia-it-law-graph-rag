"""Graph-RAG chat process"""

import os
import uuid
from typing import Any, Callable, Dict, List
import aiofiles
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import chainlit as cl


async def show_graph_viz(
    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]],
    state_messages: List[BaseMessage],
) -> None:
    """
    Displays a graph visualization based on the tool messages in the
    conversation history.

    This function extracts ToolMessage objects from the state_messages,
    invokes the graph_visualizer_tool to generate a graph visualization for
    each, and then displays these visualizations as custom elements within
    Chainlit messages. It generates a unique path for each graph visualization
    HTML file, renders the visualization, and sends a message containing the
    custom element to the Chainlit UI.

    Args:
        graph_visualizer_tool (Callable[[ToolMessage], Dict[str, Any]]): A
            callable that takes a ToolMessage and returns a dictionary
            containing the graph visualization (HTML), runtime, and artifacts.
        state_messages (List[BaseMessage]): A list of BaseMessage objects
            representing the conversation history.

    Returns:
        None
    """
    tool_messages = []
    tool_message_idxs = []

    for idx in range(-1, (-len(state_messages) - 1), -1):
        if isinstance(state_messages[idx], ToolMessage):
            tool_message_idxs.append(idx)
        elif isinstance(state_messages[idx], HumanMessage):
            if tool_message_idxs:
                tool_message_idxs.sort()
                tool_messages = [state_messages[i] for i in tool_message_idxs]
            break

    for message in tool_messages:
        graph_viz = graph_visualizer_tool(tool_message=message)

        if graph_viz["viz"]:
            graph_viz_path = str(uuid.uuid4()) + ".html"
            graph_viz_path = os.path.join("public", "graph_viz", graph_viz_path)

            async with aiofiles.open(graph_viz_path, "w", encoding="utf-8") as f:
                await f.write(graph_viz["viz"].render(height="100%").data.strip())

            element = cl.CustomElement(name="Neo4jViz", props={"src": graph_viz_path})

            await cl.Message(content="", elements=[element]).send()


async def graph_rag_on_message(
    input_msg: cl.Message,
    llm_model_name: str,
    graph_workflow: CompiledStateGraph,
    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]],
    streaming_response: bool,
    config: Dict[str, Any],
) -> None:
    """
    Handles incoming messages in a Graph-RAG chat application.

    This function orchestrates the Graph-RAG (Retrieval Augmented Generation)
    process for each incoming user message. It first invokes the graph_workflow
    to process the message and generate a response. Then, based on streaming_response
    boolean, either steams the message back to the user or send the non streaming
    one. After the response has been generated, calls the graph_visualizer_tool to
    generate and display the graph to the user.

    Args:
        input_msg (cl.Message): The incoming Chainlit message from the user.
        llm_model_name (str): The name of the language model being used.
        graph_workflow (CompiledStateGraph): The compiled LangGraph state graph
            representing the Graph-RAG workflow.
        graph_visualizer_tool (Callable[[ToolMessage], Dict[str, Any]]): A callable
            for generating graph visualizations.
        streaming_response (bool): Whether to stream the LLM response back to the
            user or to wait and send it whole at the end.
        config (Dict[str, Any]): Configuration for the Langchain RunnableConfig.

    Returns:
        None
    """
    cb = cl.LangchainCallbackHandler()

    if streaming_response:
        output_msg = cl.Message(content="")
        is_claude_llm = llm_model_name.startswith("claude")

        for msg, metadata in graph_workflow.stream(
            {"messages": [HumanMessage(content=input_msg.content)]},
            stream_mode="messages",
            config=RunnableConfig(callbacks=[cb], **config),
        ):
            if (
                isinstance(msg, AIMessageChunk)
                and metadata["langgraph_node"] == "agent"
            ):
                if (
                    is_claude_llm
                    and isinstance(msg.content, list)
                    and msg.content
                    and msg.content[0].get("text")
                ):
                    await output_msg.stream_token(str(msg.content[0]["text"]))
                elif not is_claude_llm and msg.content:
                    await output_msg.stream_token(str(msg.content))
            else:
                # Remove tool call explanation from UI
                output_msg.content = ""

        await output_msg.send()
    else:
        response = graph_workflow.invoke(
            {"messages": [HumanMessage(content=input_msg.content)]},
            config=RunnableConfig(callbacks=[cb], **config),
        )
        await cl.Message(content=str(response["messages"][-1].content)).send()

    state_messages = graph_workflow.get_state(config).values["messages"]
    await show_graph_viz(graph_visualizer_tool, state_messages)
