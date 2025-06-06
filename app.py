import chainlit as cl
from dotenv import load_dotenv
from src.grag.chainlit import (
    GRAPH_RAG_DESC,
    GRAPH_RAG_SETTINGS,
    GRAPH_RAG_STARTERS,
    graph_rag_on_message,
    prepare_graph_rag,
)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Graph-RAG",
            markdown_description=GRAPH_RAG_DESC,
            icon="https://picsum.photos/200",
            starters=GRAPH_RAG_STARTERS
        ),
        cl.ChatProfile(
            name="TAG",
            markdown_description="TODO",
            icon="https://picsum.photos/250",
            # default=True
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        settings = await cl.ChatSettings(GRAPH_RAG_SETTINGS).send()
        workflow, graph_visualizer_tool = prepare_graph_rag(
            llm_name = settings["llm_model"]
        )
        cl.user_session.set("workflow", workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)

    elif chat_profile == "TAG":
        pass


@cl.on_settings_update
async def setup_agent(settings):
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        workflow, graph_visualizer_tool = prepare_graph_rag(
            llm_name = settings["llm_model"]
        )
        cl.user_session.set("workflow", workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)

    elif chat_profile == "TAG":
        pass


@cl.on_message
async def on_message(input_msg: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    config = {"configurable": {"thread_id": cl.context.session.id}}
    
    if chat_profile == "Graph-RAG":
        workflow = cl.user_session.get("workflow")
        graph_visualizer_tool = cl.user_session.get("graph_visualizer_tool")

        await graph_rag_on_message(
            workflow=workflow,
            graph_visualizer_tool=graph_visualizer_tool,
            input_msg=input_msg,
            config=config
        )

    elif chat_profile == "TAG":
        pass


if __name__ == "__main__":
    load_dotenv()
