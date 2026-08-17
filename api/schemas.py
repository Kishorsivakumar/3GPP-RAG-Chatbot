from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the 3GPP RAG system.",
    )


class Source(BaseModel):
    specification: str
    version: str
    release: str
    section: str
    section_title: str
    content_type: str
    score: float


class Claim(BaseModel):
    text: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    allowed: bool
    reason: str
    confidence: float
    sources: List[Source] = []
    claims: List[Claim] = []
    claim_validation: Dict[str, Any] = {}
    completeness_validation: Dict[str, Any] = {}