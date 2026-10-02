"""Schemas for the chat app (CHAT_UI_PLAN.md section 6): Options, error_body and the disclaimer copy."""
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# message_stop.disclaimer and the export header: the one copy (app/engine.py and app/store.py import it).
DISCLAIMER = "Research prototype; not for clinical use."


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")          # D10: retrieval_k is rejected
    model: Optional[str] = None                          # None = the server's default engine
    decode: Literal["beam", "greedy"] = "beam"
    beam_size: Annotated[int, Field(ge=1, le=8)] = 3
    max_new_tokens: Annotated[int, Field(ge=16, le=200)] = 100
    cached_decode: bool = True
    compile: bool = False                                # refused unless the server allows it (P1-D)
    k_images: Annotated[int, Field(ge=0, le=12)] = 4
    k_reports: Annotated[int, Field(ge=0, le=10)] = 3
    label: bool = True
    reference: Optional[str] = Field(default=None, max_length=20000)   # private mode only
    display_repair: bool = False
    test_row: Optional[Annotated[int, Field(ge=0)]] = None             # private mode only (picker)


def error_body(kind: str, message: str) -> Dict[str, Any]:
    """Return error envelope: {"type": "error", "error": {"type": kind, "message": message}}."""
    return {"type": "error", "error": {"type": kind, "message": message}}
