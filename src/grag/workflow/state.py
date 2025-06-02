from operator import add
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Sequence,
    TypedDict,
    Union
)
from langgraph.graph import add_messages
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage
) 


# MASIH BELUM FIX

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class OverallState(TypedDict):
    next_step: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: Annotated[List[str], add]
    usage_metadata: Annotated[List[Dict[str, Union[str, Dict[str, Any]]]], add]
    # is_last_step: IsLastStep  # Nanti kodingan create_react_agent yang implementasi (otomatis ditambahkan)
    # remaining_steps: RemainingSteps  # Nanti kodingan create_react_agent yang implementasi (otomatis ditambahkan)

class OutputState(TypedDict):
    output: AIMessage  # Ternyata jangan pakai nama `messages`` lagi. Nanti coba lagi
    steps_history: List[str]
    llm_usage_metadata: List[Dict[str, Union[str, Dict[str, Any]]]]
