from pydantic import Field
from pydantic import BaseModel


class SimpleQuery(BaseModel):
    query: str = Field(description="Kueri pengguna, secara opsional termasuk informasi dari alat (tool) lain.")
