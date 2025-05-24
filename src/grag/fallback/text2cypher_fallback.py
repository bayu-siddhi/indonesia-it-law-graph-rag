import uuid
import json
from typing import (
    Dict,
    Optional
)
from langchain_core.messages import (
    AIMessage,
    ToolMessage
)
from .base_fallback_tool_calling import BaseFallbackToolCalling


class Text2CypherFallback(BaseFallbackToolCalling):
    """Class for handling fallback when tools fails to get data"""

    tool_map: Dict[str, str] = {
        "text2cypher_retriever": "hybrid_cypher_retriever"
    }

    @classmethod
    def check(
        cls,
        tool_message: ToolMessage
    ) -> bool:
        """
        Check whether a fallback tool should be triggered based on the tool message.
        """
        for availabel_tool_name in cls.tool_map.keys():
            if tool_message.name in availabel_tool_name:
                should_call_alternate_tool = not tool_message.artifact.get("is_context_fetched", False)
                return should_call_alternate_tool
        return False

    @classmethod
    def tool_call(
        cls,
        prev_tool_call: ToolMessage,
        name: Optional[str] = None
    ) -> AIMessage:
        """
        Create a new tool call message as a fallback attempt.
        """
        for availabel_tool_name in cls.tool_map.keys():
            if prev_tool_call.tool_calls[0]['name'] in availabel_tool_name:
                alternate_tool = cls.tool_map[availabel_tool_name]

        return AIMessage(
            content=(
                "Tidak dapat menemukan data yang sesuai dengan permintaan query: "
                f"{prev_tool_call.additional_kwargs['function_call']['arguments']} dengan "
                f"menggunakan tool {prev_tool_call.tool_calls[0]['name']}. Mencoba ulang "
                f"pencarian data dengan menggunakan tool alternatif: {alternate_tool}"
            ),
            additional_kwargs={
                "function_call": {
                    "name": alternate_tool,
                    "arguments": prev_tool_call.additional_kwargs["function_call"]["arguments"]
                }
            },
            response_metadata={},
            type=prev_tool_call.type,
            name=name,
            id=f"run-{uuid.uuid4()}-0",
            example=prev_tool_call.example,
            tool_calls=[{
                "name": alternate_tool,
                # TODO: Di langchain ada schema parser, coba itu buat parse str ini menjadi dict,
                # dan bukan pakai json.loads()
                "args": json.loads(prev_tool_call.additional_kwargs["function_call"]["arguments"]),
                "id": str(uuid.uuid4()),
                "type": "tool_call"
            }],
            invalid_tool_calls=prev_tool_call.invalid_tool_calls,
            usage_metadata = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                # TODO: Bagian di bawah ini ternyata di hasil ChatOllama tidak ada
                # Coba cek kalau ChatGPT bagaimana usage_metadata nya
                # Dan apakah tidak apa jika berbeda sedikit?
                'input_token_details': {
                    'cache_read': 0
                }
            }
        )