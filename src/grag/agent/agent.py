"""Agent to use tools or generate final answer"""

from copy import deepcopy
from typing import cast, Callable, List, Optional, Sequence, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt.chat_agent_executor import (
    Prompt,
    StateSchema,
    _get_prompt_runnable,
    _get_state_value,
    _should_bind_tools,
    _validate_chat_history,
)
from .prompts import AGENT_SYSTEM_PROMPT
from ..fallback import BaseFallbackToolCalling


def create_agent(
    model: BaseChatModel,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    *,
    name: str = "agent",
    prompt: Optional[Prompt] = None,
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None,
) -> RunnableCallable:
    """
    Creates a runnable agent that processes chat messages, optionally uses tools,
    and supports fallback tool-calling behavior if a tool fails to provide the
    desired output.

    Args:
        model (BaseChatModel): The chat model that powers the agent (e.g., OpenAI,
            Anthropic).
        tools (Union[Sequence[Union[BaseTool, Callable]], ToolNode]):
            A list of tools the agent can use, or a pre-defined ToolNode.
        name (str, optional): The name of the agent. Used for tagging responses.
            Defaults to "agent".
        prompt (Optional[Prompt], optional): A custom system prompt for the agent.
            Defaults to `AGENT_SYSTEM_PROMPT`.
        fallback_tool_calling_cls (Optional[Type[BaseFallbackToolCalling]], optional):
            An optional class that handles fallback tool calls if a previous call
            fails. Defaults to None.

    Returns:
        result (RunnableCallable): A callable object that processes agent
            state and returns the next agent message.
    """
    if not prompt:
        prompt = AGENT_SYSTEM_PROMPT

    # Convert tools to ToolNode if necessary and extract tool classes
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
    else:
        tool_node = ToolNode(tools)
        tool_classes = list(tool_node.tools_by_name.values())

    tool_calling_enabled = len(tool_classes) > 0

    # Bind tools to model if the model supports it
    if _should_bind_tools(model, tool_classes) and tool_calling_enabled:
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    # Combine the prompt with the model to make a runnable chain
    model_runnable = _get_prompt_runnable(prompt) | model

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        """
        Determine whether more steps are needed based on the agent's response.

        Conditions for continuing:
        - There are tool calls AND
        - Either:
            * remaining_steps is None and it's the last step, OR
            * remaining_steps < 2

        Args:
            state (StateSchema): The current state of the workflow.
            response (BaseMessage): The latest message from the agent.

        Returns:
            result (bool): True if the workflow should
                continue with more steps, False otherwise.
        """
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (remaining_steps is None and is_last_step and has_tool_calls) or (
            remaining_steps is not None and remaining_steps < 2 and has_tool_calls
        )

    def _check_fallback(messages: List[BaseMessage]) -> Union[StateSchema, bool]:
        """
        Checks if a fallback tool call is needed based on recent messages.

        Args:
            messages (List[BaseMessage]): List of recent messages in the workflow trace.

        Returns:
            result (Union[StateSchema, bool]): A dictionary `{'messages': [AIMessage]}`
                containing a new message with fallback tool calls if a fallback is
                triggered, otherwise False.
        """
        tool_call_message = None
        tool_message_idxs = []
        tool_messages = []

        # Get the tool call and tool message
        for idx in range(-1, (-len(messages) - 1), -1):
            if isinstance(messages[idx], ToolMessage):
                tool_message_idxs.append(idx)
            else:
                if isinstance(messages[idx], AIMessage) and tool_message_idxs:
                    tool_call_message = messages[idx]
                    tool_message_idxs.sort()
                    tool_messages = [messages[i] for i in tool_message_idxs]
                    break

        if tool_call_message:
            # Check if the previous tool failed to get data
            fallback_tool_status = fallback_tool_calling_cls.check(
                tool_messages=tool_messages
            )

            # If any tool failed to get data, then do fallback
            if any(fallback_tool_status):
                new_tool_call_message = fallback_tool_calling_cls.tool_call(
                    prev_tool_call=tool_call_message,
                    fallback_tool_status=fallback_tool_status,
                    name=name,
                )

                return {"messages": [new_tool_call_message]}

        return False

    def _query_injection(
        ai_message: AIMessage,
        human_message: HumanMessage
    ):
        if ai_message.tool_calls:
            tool_calls = []
            for tool_call in ai_message.tool_calls:
                tool_call_copy = deepcopy(tool_call)
                tool_call_copy["args"]["query"] = str(human_message.content)
                tool_calls.append(tool_call_copy)
            ai_message.tool_calls = tool_calls

        return ai_message

    # Define the function that calls the model
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        """
        Call the model and return the updated state (synchronously).

        Args:
            state (StateSchema): The current agent state, including message history
                and step metadata.
            config (RunnableConfig): Additional configuration for invoking the model
                (e.g., callbacks, tags).

        Returns:
            result (StateSchema): The updated state including the new message
                (either from model or fallback tool).
        """
        messages = _get_state_value(state, "messages")
        _validate_chat_history(messages)

        # Check fallback status
        if fallback_tool_calling_cls is not None:
            fallback: Union[StateSchema, bool] = _check_fallback(messages=messages)
            if fallback:
                return fallback

        # Generate tool calling or final answer
        response = cast(AIMessage, model_runnable.invoke(state, config))
        response.name = name

        if isinstance(messages[-1], HumanMessage):
            response = _query_injection(response, messages[-1])

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content=(
                            "Maaf, diperlukan langkah lebih lanjut untuk memproses "
                            "permintaan ini."
                        ),
                    )
                ]
            }

        return {"messages": [response]}

    async def acall_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        """
        Call the model and return the updated state (asynchronously).

        Args:
            state (StateSchema): The current agent state, including message history
                and step metadata.
            config (RunnableConfig): Additional configuration for invoking the model
                (e.g., callbacks, tags).

        Returns:
            result (StateSchema): The updated state including the new message
                (either from model or fallback tool).
        """
        messages = _get_state_value(state, "messages")
        _validate_chat_history(messages)

        # Check fallback status
        if fallback_tool_calling_cls is not None:
            fallback: Union[StateSchema, bool] = _check_fallback(messages=messages)
            if fallback:
                return fallback

        # Generate tool calling or final answer
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
        response.name = name

        if isinstance(messages[-1], HumanMessage):
            response = _query_injection(response, messages[-1])

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content=(
                            "Maaf, diperlukan langkah lebih lanjut untuk memproses "
                            "permintaan ini."
                        ),
                    )
                ]
            }

        return {"messages": [response]}

    return RunnableCallable(call_model, acall_model)
