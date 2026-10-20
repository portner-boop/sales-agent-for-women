from pydantic import BaseModel


class IncomingMessage(BaseModel):
    external_id: int
    text: str
    event_time: float
    reply_to_external_id: int | None = None
    forwarded: bool = False
    is_edit: bool = False
