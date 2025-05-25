from typing import (
    cast,
    Callable,
    List,
    Optional,
    Sequence,
    Type,
    Union
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
    
    from langchain_core.messages import trim_messages

    # Define the function that calls the model
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        messages = _get_state_value(state, "messages")
        _validate_chat_history(messages)

        # https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_window_memory/#modern-usage-with-langgraph
        # Ngga tahu cara lihat berhasil atau tidaknya bagaimana
        # state["messages"] = trim_messages(
        #     messages,
        #     token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        #     max_tokens=10,  # <-- allow up to 5 messages.
        #     strategy="last",
        #     # Most chat models expect that chat history starts with either:
        #     # (1) a HumanMessage or
        #     # (2) a SystemMessage followed by a HumanMessage
        #     # start_on="human" makes sure we produce a valid chat history
        #     start_on="human",
        #     # Usually, we want to keep the SystemMessage
        #     # if it's present in the original history.
        #     # The SystemMessage has special instructions for the model.
        #     include_system=True,
        #     allow_partial=False,
        # )

        ######################## BUATANKU ##########################
        # Cek jika tool sebelumnya gagal mendapatkan data
        tool_message_idxs = []
        tool_call_message = []
        tool_messages = None
        for idx in range(-1, (-len(messages) - 1), -1):
            if isinstance(messages[idx], ToolMessage):
                tool_message_idxs.append(idx)
            else:
                # if tool_message_idxs:
                if tool_message_idxs and isinstance(messages[idx], AIMessage):
                    tool_call_message = messages[idx]
                    tool_message_idxs.sort()
                    tool_messages = [messages[i] for i in tool_message_idxs]
                    break
                # else:
                    # break
        
        if fallback_tool_calling_cls is not None and tool_call_message:
            fallback_tool_status: List[bool] = fallback_tool_calling_cls.check(
                tool_messages=tool_messages
            )
            if any(fallback_tool_status):
                print("FOLLBACK")  # Nanti hapus
                # fallback_tool_messages = [
                #     message
                #     for message, call_alternate in zip(tool_messages, fallback_tool_status)
                #     if call_alternate
                # ]
                new_tool_call_message = fallback_tool_calling_cls.tool_call(
                    prev_tool_call=tool_call_message,
                    fallback_tool_status=fallback_tool_status,
                    name=name
                )
                return {
                    "messages": [new_tool_call_message],
                    "steps": [name]  # Samain kaya OverallState Workflow nya
                }
        print("NO FOLLBACK")  # Nanti hapus

        # last_message = messages[-1]
        # if fallback_tool_calling_cls is not None and isinstance(last_message, ToolMessage):
        #     if fallback_tool_calling_cls.check(last_message):
        #         print("FOLLBACK")  # Nanti hapus
        #         tool_call_message = fallback_tool_calling_cls.tool_call(
        #             prev_tool_call=messages[-2], name=name
        #         )
        #         return {
        #             "messages": [tool_call_message],
        #             "steps": [name]  # Samain kaya OverallState Workflow nya
        #         }
        # print("NO FOLLBACK")  # Nanti hapus
        ############################################################
        # TODO: BARU SAMPAI SINI, BELUM TESTING
        
        response = cast(AIMessage, model_runnable.invoke(state, config))
        # response = cast(AIMessage, model_runnable.invoke(selected_messages, config))
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
