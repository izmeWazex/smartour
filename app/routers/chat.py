import uuid
from typing import Dict, List
from fastapi import APIRouter, HTTPException, status

from app.models.chat import (
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryResponse,
    ClearHistoryResponse,
    MessageRole,
)
from app.ai.engine import smartour_ai

router = APIRouter(prefix="/chat", tags=["Chat"])

# In-memory session store (TODO: persist to DB)
_sessions: Dict[str, List[ChatMessage]] = {}


def _get_or_create_session(session_id: str | None) -> str:
    """Return the existing session ID, or create a new one."""
    if session_id:
        if session_id not in _sessions:
            _sessions[session_id] = []  # create with the provided ID
        return session_id
    new_id = str(uuid.uuid4())
    _sessions[new_id] = []
    return new_id


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and get a reply from Smartour AI",
)
async def send_message(body: ChatMessageRequest) -> ChatMessageResponse:
    """Send a user message; returns the reply and full conversation history."""
    session_id = _get_or_create_session(body.session_id)

    # Append user message
    user_msg = ChatMessage(role=MessageRole.user, content=body.message)
    _sessions[session_id].append(user_msg)

    # Build history context for AI (list of dicts)
    history_context = [
        {"role": msg.role, "content": msg.content}
        for msg in _sessions[session_id]
    ]

    # Get AI reply
    reply_text = smartour_ai.respond(body.message, history_context)

    # Append assistant message
    assistant_msg = ChatMessage(role=MessageRole.assistant, content=reply_text)
    _sessions[session_id].append(assistant_msg)

    return ChatMessageResponse(
        session_id=session_id,
        reply=reply_text,
        history=_sessions[session_id],
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get chat history for a session",
)
async def get_history(session_id: str) -> ChatHistoryResponse:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    history = _sessions[session_id]
    return ChatHistoryResponse(
        session_id=session_id,
        history=history,
        message_count=len(history),
    )


@router.delete(
    "/history/{session_id}",
    response_model=ClearHistoryResponse,
    summary="Clear chat history for a session",
)
async def clear_history(session_id: str) -> ClearHistoryResponse:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    del _sessions[session_id]
    return ClearHistoryResponse(
        session_id=session_id,
        message="Chat history cleared successfully.",
    )


@router.get(
    "/sessions",
    summary="List all active session IDs",
)
async def list_sessions() -> dict:
    return {
        "sessions": [
            {"session_id": sid, "message_count": len(msgs)}
            for sid, msgs in _sessions.items()
        ]
    }
