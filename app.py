"""Chainlit main app"""

from typing import Any, Dict
from dotenv import load_dotenv
import chainlit as cl
from ui.grag import (
    GRAPH_RAG_SETTINGS,
    GRAPH_RAG_STARTERS,
    configure_graph_rag,
    graph_rag_on_message,
    initialize_system,
)
# from ui.tag import ...


@cl.set_chat_profiles
async def set_chat_profile():
    """
    Defines the available chat profiles for the Chainlit application.

    This function sets up the available chat profiles, including Graph-RAG and TAG, 
    with their respective descriptions, icons, and starter messages.

    Returns:
        results (List[cl.ChatProfile]): A list of ChatProfile objects representing 
            the available chat profiles.
    """
    return [
        cl.ChatProfile(
            name="Graph-RAG",
            markdown_description="# **Graph-RAG**",
            icon="https://picsum.photos/200",
            starters=GRAPH_RAG_STARTERS,
            default=True,
        ),
        # cl.ChatProfile(
        #     name="TAG",
        #     markdown_description="# **Table-Augmented Generation**",
        #     icon="https://picsum.photos/250",
        #     starters=TAG_STARTERS,
        # ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the chat session when a user connects, configuring either 
    the Graph-RAG or TAG workflow.

    This function is called when a new user connects to the Chainlit application. 
    It determines the selected chat profile and configures the appropriate 
    workflow (Graph-RAG or TAG) by initializing the necessary components and 
    setting them in the user session.
    
    Returns:
        None
    """
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        settings = await cl.ChatSettings(GRAPH_RAG_SETTINGS).send()

        neo4j_graph, embedder_model = initialize_system()
        cl.user_session.set("neo4j_graph", neo4j_graph)
        cl.user_session.set("embedder_model", embedder_model)

        graph_workflow, graph_visualizer_tool = configure_graph_rag(
            llm_name=settings["llm_model_name"],
            neo4j_graph=cl.user_session.get("neo4j_graph"),
            embedder_model=cl.user_session.get("embedder_model"),
        )

        cl.user_session.set("stream", settings["stream"])
        cl.user_session.set("llm_model_name", settings["llm_model_name"])
        cl.user_session.set("graph_workflow", graph_workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)

    # elif chat_profile == "TAG":
    #     pass


@cl.on_settings_update
async def setup_agent(settings: Dict[str, Any]):
    """
    Updates the chat session based on new settings selected by the user.

    This function is called when the user updates the settings for the chat 
    session. It reconfigures the appropriate workflow (Graph-RAG or TAG) by 
    initializing the necessary components and setting them in the user session.
    
    Args:
        settings (Dict[str, Any]): The new chat settings object.

    Returns:
        None
    """
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        graph_workflow, graph_visualizer_tool = configure_graph_rag(
            llm_name=settings["llm_model_name"],
            neo4j_graph=cl.user_session.get("neo4j_graph"),
            embedder_model=cl.user_session.get("embedder_model"),
        )

        cl.user_session.set("stream", settings["stream"])
        cl.user_session.set("llm_model_name", settings["llm_model_name"])
        cl.user_session.set("graph_workflow", graph_workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)

    # elif chat_profile == "TAG":
    #     pass


@cl.on_message
async def on_message(input_msg: cl.Message):
    """
    Handles incoming messages in the Chainlit application and routes them to 
    the appropriate workflow (Graph-RAG or TAG).

    This function is the main entry point for handling user messages. It 
    retrieves the selected chat profile from the user session and routes the 
    message to either the Graph-RAG or TAG workflow for processing.

    Args:
        input_msg (cl.Message): The incoming Chainlit message from the user.

    Returns:
        None
    """
    chat_profile = cl.user_session.get("chat_profile")
    config = {"configurable": {"thread_id": cl.context.session.id}}

    if chat_profile == "Graph-RAG":
        await graph_rag_on_message(
            input_msg=input_msg,
            llm_model_name=cl.user_session.get("llm_model_name"),
            graph_workflow=cl.user_session.get("graph_workflow"),
            graph_visualizer_tool=cl.user_session.get("graph_visualizer_tool"),
            streaming_response=cl.user_session.get("stream"),
            config=config,
        )

    # elif chat_profile == "TAG":
    #     pass  


if __name__ == "__main__":
    load_dotenv()
