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
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchema,
    StateSchemaType,
    _get_state_value
)
from ..agent import create_agent
from ..fallback import BaseFallbackToolCalling


def create_workflow(
    model: BaseChatModel,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: Optional[str] = None,
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None
) -> CompiledStateGraph:
    """
    Creates a compiled state graph workflow for an agent that can use tools.

    Args:
        model (BaseChatModel): The chat-based language model to use for the agent.
        tools (Union[Sequence[Union[BaseTool, Callable]], ToolNode]):
            A list of tools (functions, APIs, etc.) the agent can call, or an 
            existing ToolNode.
        prompt (Optional[Prompt], optional): A custom prompt for the agent. 
            Defaults to None.
        state_schema (Optional[StateSchemaType], optional):
            The schema for the agent's state. If not provided, `AgentState` is used 
            by default.
        config_schema (Optional[Type[Any]], optional):
            An optional configuration schema for the workflow. Defaults to None.
        checkpointer (Optional[Checkpointer], optional):
            Optional checkpointer for saving state across steps. Useful for resuming 
            or debugging. Defaults to None.
        store (Optional[BaseStore], optional):
            Optional store to persist or retrieve additional data used in the workflow. 
            Defaults to None.
        name (Optional[str], optional): Optional name for the compiled workflow. 
            Defaults to None.
        fallback_tool_calling_cls (Optional[Type[BaseFallbackToolCalling]], optional):
            A fallback tool-calling class in case the agent fails to handle a tool 
            call properly. Defaults to None.

    Returns:
        CompiledStateGraph: A compiled StateGraph object that defines the full agent 
            and tool execution loop.
    """

    # Use default state schema if none is provided
    if state_schema is None:
        state_schema = AgentState
    # Validate state schema if provided
    else:
        required_keys = {"messages", "remaining_steps"}
        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    # If tools is not already a ToolNode, create one
    if not isinstance(tools, ToolNode):
        tool_node = ToolNode(tools)
    
    def should_continue(state: StateSchema) -> Union[str, list]:
        """Function that determines whether to continue or not. 
        Continues if the last message had tool calls.
        """
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        # Otherwise if there is, we continue
        else:
            tool_calls = [
                tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                for call in last_message.tool_calls
            ]
            return [Send("tools", [tool_call]) for tool_call in tool_calls]
    
    # Define a new graph workflow
    workflow = StateGraph(state_schema or AgentState, config_schema=config_schema)

    # Define the nodes
    workflow.add_node("agent", create_agent(
            model=model,
            tools=tool_node,
            name="agent",
            prompt=prompt,
            fallback_tool_calling_cls=fallback_tool_calling_cls
        )
    )
    workflow.add_node("tools", tool_node)
    
    # Define the edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["tools", END]
    )
    workflow.add_edge("tools", "agent")

    # Compile the graph workflow
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        name=name
    )
