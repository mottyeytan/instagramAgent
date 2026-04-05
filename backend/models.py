from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    target_description: str
    seed_username: str | None = None
    seed_name: str | None = None
    photo_paths: list[str] = Field(default_factory=list)
    time_limit_minutes: int = 10


class ResumeRequest(BaseModel):
    answer: str


class SSEEvent(BaseModel):
    event: str
    data: dict = Field(default_factory=dict)


class InvestigationResponse(BaseModel):
    investigation_id: str
    status: str
    stream_url: str | None = None
