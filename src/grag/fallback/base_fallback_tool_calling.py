from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Dict,
    Optional
)
from langchain_core.messages import (
    AIMessage,
    ToolMessage
)


class BaseFallbackToolCalling(ABC):
    """Abstract base class for handling fallback when tools fails to get data"""

    tool_map: Dict[str, str] = {}

    @classmethod
    @abstractmethod
    def check(
        cls,
        tool_message: ToolMessage
    ) -> bool:
        """
        Check whether a fallback tool should be triggered based on the tool message.
        """
        pass

    @classmethod
    @abstractmethod
    def tool_call(
        cls,
        prev_tool_call: ToolMessage,
        name: Optional[str] = None
    ) -> AIMessage:
        """
        Create a new tool call message as a fallback attempt.
        """
        pass
