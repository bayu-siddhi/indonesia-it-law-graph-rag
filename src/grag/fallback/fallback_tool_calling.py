"""Implementation class for fallback tool callling"""

import uuid
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, ToolMessage
from .base_fallback_tool_calling import BaseFallbackToolCalling


class FallbackToolCalling(BaseFallbackToolCalling):
    """Class for handling fallback when tools fails to get data"""

    # Maps original tool name to its fallback tool name
    tool_map: Dict[str, str] = {"text2cypher_retriever": "vector_cypher_retriever"}

    @classmethod
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
        fallback_tool_status = []
        for tool_message in tool_messages:
            for availabel_tool_name in cls.tool_map:
                if tool_message.name in availabel_tool_name:
                    fallback_tool_status.append(
                        not tool_message.artifact.get("is_context_fetched", False)
                    )
                    break
                fallback_tool_status.append(False)

        return fallback_tool_status

    @classmethod
    def tool_call(
        cls,
        prev_tool_call: AIMessage,
        fallback_tool_status: List[bool],
        name: Optional[str] = None,
    ) -> AIMessage:
        """
        Creates a new AIMessage containing calls to fallback tools for those that
        failed.

        Args:
            prev_tool_call (AIMessage): The original AI message containing tool
                calls.
            fallback_tool_status (List[bool]): Status list from `check` indicating
                which calls failed.
            name (Optional[str]): Optional name for the new message.

        Returns:
            ai_message (AIMessage): A new message with fallback tool calls.
        """
        prev_tool_name: List[str] = []
        prev_tool_call_args: List[Dict[str, Any]] = []
        alternate_tool_name: List[str] = []

        for tool_call, status in zip(prev_tool_call.tool_calls, fallback_tool_status):
            if status is True:
                for tool, alternate_tool in cls.tool_map.items():
                    if tool_call["name"] in tool:
                        prev_tool_name.append(tool)
                        prev_tool_call_args.append(tool_call["args"])
                        alternate_tool_name.append(alternate_tool)

        new_tool_calls: List[Dict[str, Any]] = []

        for tool_name, tool_args in zip(alternate_tool_name, prev_tool_call_args):
            new_tool_calls.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }
            )

        return AIMessage(
            content=(""),
            additional_kwargs={},
            response_metadata={},
            type=prev_tool_call.type,
            name=name,
            id=f"run-{uuid.uuid4()}-0",
            example=prev_tool_call.example,
            tool_calls=new_tool_calls,
            invalid_tool_calls=prev_tool_call.invalid_tool_calls,
        )

        # "Tidak dapat menemukan data yang sesuai untuk request: "
        # f"{prev_tool_call_args} dengan menggunakan tool {prev_tool_name}. "
        # "Mencoba ulang pencarian data dengan menggunakan tool alternatif "
        # f"{alternate_tool_name} untuk mendapatkan konteks tambahan."
