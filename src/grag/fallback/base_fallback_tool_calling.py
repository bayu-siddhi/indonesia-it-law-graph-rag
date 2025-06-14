"""Base (abstract) class for fallback tool callling"""

from abc import ABC, abstractmethod
from typing import Dict, List
from langchain_core.messages import AIMessage, ToolMessage


class BaseFallbackToolCalling(ABC):
    """Abstract base class for handling fallback when tools fails to get data"""

    # Maps original tool name to its fallback tool name
    tool_map: Dict[str, str] = {}

    @classmethod
    @abstractmethod
    def check(cls, tool_messages: List[ToolMessage]) -> List[bool]:
        """
        Check whether a fallback tool should be triggered based on the tool message.

        Args:
            tool_messages (List[ToolMessage]): List of results from previous tool
                calls.

        Returns:
            fallback_tool_status (List[bool]): List indicating True for tool calls
                needing fallback.
        """

    @classmethod
    @abstractmethod
    def tool_call(
        cls, prev_tool_call: AIMessage, fallback_tool_status: List[bool]
    ) -> AIMessage:
        """
        Creates a new AIMessage containing calls to fallback tools for those that 
        failed.

        Args:
            prev_tool_call (AIMessage): The original AI message containing tool 
                calls.
            fallback_tool_status (List[bool]): Status list from `check` indicating 
                which calls failed.

        Returns:
            ai_message (AIMessage): A new message with fallback tool calls.
        """
