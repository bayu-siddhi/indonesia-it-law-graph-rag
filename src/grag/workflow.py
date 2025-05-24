from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Type,
    Union,
    get_type_hints
)
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import (
    Checkpointer,
    Send
)
from langgraph.store.base import BaseStore
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchema,
    StateSchemaType,
    _get_state_value
)
from .agent import create_agent
from .fallback import BaseFallbackToolCalling


def create_workflow(
    model: BaseChatModel,
    tools: Sequence[Union[BaseTool, Callable]],
    *,
    prompt: Optional[Prompt] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: Optional[str] = None,
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None
):
    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if state_schema is None:
        state_schema = AgentState  # Pakai AgentStateWithStructuredResponse jika jadi pakai Agent Perangkum

    tool_node = ToolNode(tools)
    
    # Define the function that determines whether to continue or not
    def should_continue(state: StateSchema) -> Union[str, list]:
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        # Otherwise if there is, we continue
        else:
            # return "tools"
            # NGGA TAHU KENAPA KALAU PAKAI V2 ITU MALAH ERROR
            tool_calls = [
                tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                for call in last_message.tool_calls
            ]
            return [Send("tools", [tool_call]) for tool_call in tool_calls]
    
    # MEMBUAT WORKFLOW
    # Kurang guardrails, planner (jika jadi), dan final_answer
    # final_answer tujuannya cuma untuk mengambil pesan terakhir di dalam messages
    # untuk diberikan sebagai output ke pengguna. Jadi ngga semua history pesan
    # diberikan ke pengguna seperti pada ReAct Agent biasanya.
    # Tujuannya agar data yang masuk ke memory dan dipakai di input llm pesan selanjutnya
    # bisa berkurang hanya query pengguna dan jawaban akhir llm saja (selama ini kan
    # semua data dimasukkan ke memory, seperti ToolMessage text2cyper, dkk.)
    #########################################################################################
    
    # Define a new graph
    workflow = StateGraph(state_schema or AgentState, config_schema=config_schema)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", create_agent(
            model=model,
            tools=tool_node,
            prompt=prompt,
            name="agent",
            fallback_tool_calling_cls=fallback_tool_calling_cls
        )
    )
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")
    
    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        path_map=["tools", END]
    )

    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        name=name
    )
