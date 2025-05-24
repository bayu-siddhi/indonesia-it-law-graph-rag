from typing import (
    Callable,
    Optional,
    Sequence,
    Type,
    Union,
    cast
)
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage
)
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt.chat_agent_executor import (
    Prompt,
    StateSchema,
    _get_prompt_runnable,
    _get_state_value,
    _should_bind_tools,
    _validate_chat_history
)
from .fallback import BaseFallbackToolCalling


def create_agent(
    model: BaseChatModel,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    name: str = "agent",
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None,
):
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
    else:
        tool_node = ToolNode(tools)
        # get the tool functions wrapped in a tool class from the ToolNode
        tool_classes = list(tool_node.tools_by_name.values())

    tool_calling_enabled = len(tool_classes) > 0

    if _should_bind_tools(model, tool_classes) and tool_calling_enabled:
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    model_runnable = _get_prompt_runnable(prompt) | model
    
    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )

    # Define the function that calls the model
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        messages = _get_state_value(state, "messages")
        _validate_chat_history(messages)

        ######################## BUATANKU ##########################
        # Cek jika tool sebelumnya gagal mendapatkan data
        last_message = messages[-1]
        if fallback_tool_calling_cls is not None and isinstance(last_message, ToolMessage):
            if fallback_tool_calling_cls.check(last_message):
                print("FOLLBACK")  # Nanti hapus
                tool_call_message = fallback_tool_calling_cls.tool_call(
                    prev_tool_call=messages[-2], name=name
                )
                return {
                    "messages": [tool_call_message],
                    "steps": [name]  # Samain kaya OverallState Workflow nya
                }
        print("NO FOLLBACK")  # Nanti hapus
        ############################################################
        # TODO: BARU SAMPAI SINI, BELUM TESTING
        
        response = cast(AIMessage, model_runnable.invoke(state, config))
        # add agent name to the AIMessage
        response.name = name

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
                ],
                "steps": [name]  # Samain kaya OverallState Workflow nya
            }
        # We return a list, because this will get added to the existing list
        return {
            "messages": [response],
            "steps": [name]  # Samain kaya OverallState Workflow nya
        }

    return RunnableCallable(call_model)
