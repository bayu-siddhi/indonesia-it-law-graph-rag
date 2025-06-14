import os
import uuid
import chainlit as cl
from typing import Any, Callable, Dict, List
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


async def show_graph_viz(
    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]],
    state_messages: List[BaseMessage],
) -> None:
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

        # If graph_viz["viz"] is not False
        if graph_viz["viz"]:
            graph_viz_path = str(uuid.uuid4()) + ".html"
            graph_viz_path = os.path.join("public", "graph_viz", graph_viz_path)

            with open(graph_viz_path, "w", encoding="utf-8") as f:
                f.write(
                    graph_viz["viz"]
                    .render(
                        height="100%",
                        # show_hover_tooltip=True
                        # Tunggu versi neo4j-viz==0.2.7 publish
                    )
                    .data.strip()
                )

            element = cl.CustomElement(name="Neo4jViz", props={"src": graph_viz_path})

            await cl.Message(content="Visualisasi Neo4j:", elements=[element]).send()


async def graph_rag_on_message(
    workflow: CompiledStateGraph,
    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]],
    input_msg: cl.Message,
    config: Dict[str, Any],
) -> None:
    cb = cl.LangchainCallbackHandler()
    output_msg = cl.Message(content="")

    print(input_msg.content)

    # from langgraph.prebuilt import create_react_agent

    for msg, metadata in workflow.stream(
        {"messages": [HumanMessage(content=input_msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
        # config=config
    ):
        # print("MASUK")
        # print(msg.content)
        if (
            msg.content
            and isinstance(msg, AIMessageChunk)
            and metadata["langgraph_node"] == "agent"
        ):
            await output_msg.stream_token(msg.content)

    await output_msg.send()

    state_messages: List[BaseMessage] = workflow.get_state(config).values["messages"]

    await show_graph_viz(graph_visualizer_tool, state_messages)
