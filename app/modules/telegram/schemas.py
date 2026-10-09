from typing import Any

from pydantic import BaseModel, ConfigDict


class RawUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")
    update_id: int
    message: dict[str, Any] | None = None
    edited_message: dict[str, Any] | None = None
    my_chat_member: dict[str, Any] | None = None
