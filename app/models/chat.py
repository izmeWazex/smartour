from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime
import uuid


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to continue an existing conversation. "
                    "If None, a new session is created.",
    )
    message: str = Field(..., min_length=1, max_length=4096, description="User message text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "abc-123",
                "message": "What are the best tours in Kuala Lumpur?",
            }
        }
    }


class ClearHistoryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to clear")


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessageResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessage]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]
    message_count: int


class ClearHistoryResponse(BaseModel):
    session_id: str
    message: str
