import os
import uuid
import chainlit as cl
from typing import (
    Any,
    Dict,
    List
)
from pprint import pprint
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage
)
from IPython.display import HTML
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from src.grag import (
    create_grag_workflow,
    create_graph_visualizer_tool,
    create_vector_cypher_retriever_tool,
    create_text2cypher_retriever_tool,
    Text2CypherFallback,
)


load_dotenv(".env")

graph_rag_markdown_desc = """
### ⚙️ Graph-RAG

> **Graph-RAG** adalah sistem tanya-jawab cerdas yang menggabungkan kekuatan **Large Language Model (LLM)** dan **Graph Database** untuk menghasilkan jawaban yang lebih **akurat**, **kontekstual**, dan **terhubung**. Dengan memanfaatkan **LLM** dan **graf pengetahuan**, Graph-RAG mampu memahami pertanyaan kompleks serta menelusuri hubungan antar data secara mendalam.

🎯 **Coba sekarang!** Ajukan pertanyaanmu dan temukan jawaban cerdas dari Graph-RAG.
""".strip()

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Graph-RAG",
            markdown_description=graph_rag_markdown_desc,
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="TAG",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]

URI = os.environ["DATABASE_HOST"]
DATABASE = os.environ["DATABASE_SMALL"]
USERNAME = os.environ["DATABASE_USERNAME"]
PASSWORD = os.environ["DATABASE_PASSWORD"]
DATABASE = os.environ["DATABASE_SMALL"]

neo4j_config = {
    "DATABASE_NAME": DATABASE,
    "ARTICLE_VECTOR_INDEX_NAME": os.environ["ARTICLE_VECTOR_INDEX_NAME"],
    "ARTICLE_FULLTEXT_INDEX_NAME": os.environ["ARTICLE_FULLTEXT_INDEX_NAME"],
    "DEFINITION_VECTOR_INDEX_NAME": os.environ["DEFINITION_VECTOR_INDEX_NAME"],
    "DEFINITION_FULLTEXT_INDEX_NAME": os.environ["DEFINITION_FULLTEXT_INDEX_NAME"],
}

neo4j_graph = Neo4jGraph(
    url=URI,
    username=USERNAME,
    password=PASSWORD,
    database=DATABASE,
    enhanced_schema=True
)

neo4j_driver = neo4j_graph._driver

embedder_model = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    api_key=os.environ["GOOGLE_API_KEY"],
    streaming=True
)

vector_cypher_retriever = create_vector_cypher_retriever_tool(
    embedder_model=embedder_model,
    neo4j_driver=neo4j_driver,
    neo4j_config=neo4j_config,
    total_definition_limit=5,
    top_k_initial_article=5,
    max_k_expanded_article=-1,
    total_article_limit=None,
)

text2cypher_retriever = create_text2cypher_retriever_tool(
    neo4j_graph=neo4j_graph,
    embedder_model=embedder_model,
    cypher_llm=llm,
    qa_llm=llm,
    skip_qa_llm=True,
    verbose=False
)

graph_visualizer = create_graph_visualizer_tool(
    llm=llm,
    neo4j_graph=neo4j_graph,
    autocomplete_relationship=True,
    verbose=False
)

checkpointer = MemorySaver()

workflow = create_grag_workflow(
    llm, [text2cypher_retriever, vector_cypher_retriever],
    checkpointer=checkpointer,
    fallback_tool_calling_cls=Text2CypherFallback
)

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

async def graph_rag_on_message(
    input_msg: cl.Message,
    config: Dict[str, Any]
) -> None:
    cb = cl.LangchainCallbackHandler()
    output_msg = cl.Message(content="")
    
    for msg, metadata in workflow.stream(
        {"messages": [HumanMessage(content=input_msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config)
        # config=config
    ):
        if (
            metadata["langgraph_node"] == "agent"
            and msg.content
        ):
            await output_msg.stream_token(msg.content)
        
    await output_msg.send()

    # chat_history: List[BaseMessage] = workflow\
    #     .get_state(config)\
    #     .values["messages"]
    
    # tool_messages = []
    # tool_message_idxs = []

    # for idx in range(-1, (-len(chat_history) - 1), -1):
    #     if isinstance(chat_history[idx], ToolMessage):
    #         tool_message_idxs.append(idx)
    #     else:
    #         if isinstance(chat_history[idx], HumanMessage):
    #             if tool_message_idxs:
    #                 tool_message_idxs.sort()
    #                 tool_messages = [
    #                     chat_history[i] for i in tool_message_idxs
    #                 ]
    #             break
    
    # for message in tool_messages:
    #     graph_viz = graph_visualizer(tool_message=message)
    #     graph_viz_path = str(uuid.uuid4()) + ".html"
    #     graph_viz_path = os.path.join("graph_viz", graph_viz_path)
    #     with open(graph_viz_path, "w", encoding="utf-8") as f:
    #         f.write(graph_viz["viz"].render(height="300px").data.strip())
    #     with open(graph_viz_path, "r", encoding="utf-8") as f:
    #         html_string = f.read()

    #     await cl.Message(content=html_string).send()

    # pprint(workflow.get_state(config)[0])



@cl.on_message
async def on_message(input_msg: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    config = {"configurable": {"thread_id": cl.context.session.id}}
    
    if chat_profile == "Graph-RAG":
        await graph_rag_on_message(input_msg, config)
    elif chat_profile == "TAG":
        pass

    
# import chainlit as cl

# @cl.on_chat_start
# async def start():
#     # html_string adalah string HTML yang dihasilkan oleh neo4j_viz
#     with open("graph_viz/1.html", "r", encoding="utf-8") as f:
#         html_string = f.read()
    
#     # html_string = "<h1 style='color:red;'>Tes render</h1><script>console.log('Script jalan');</script>"

#     await cl.Message(
#         content=html_string
#     ).send()
