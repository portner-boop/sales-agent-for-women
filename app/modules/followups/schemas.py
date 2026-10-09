from pydantic import BaseModel, ConfigDict


class FollowupJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    followup_id: str
