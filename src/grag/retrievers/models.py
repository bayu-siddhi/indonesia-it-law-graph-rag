"""Base model for Graph-RAG retrievers"""

from pydantic import BaseModel, Field


class SimpleQueryInput(BaseModel):
    query: str = Field(
        description=(
            "Kueri pengguna, secara opsional termasuk informasi dari alat (tool) lain."
        )
    )
