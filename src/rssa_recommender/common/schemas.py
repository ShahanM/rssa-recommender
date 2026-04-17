"""Pydantic models for recommendations requests."""

from typing import Literal, Optional

from pydantic import BaseModel


class MovieLensRating(BaseModel):
    """A Pydantic model for the rating schema."""

    item_id: int
    rating: float


class RecommendationRequestPayloadSchema(BaseModel):
    """Request pauyload consisting of rated items."""

    user_id: str
    ratings: list[MovieLensRating]
    n: int


MovieLensID = int


class RecommendationResponsePayloadSchema(BaseModel):
    """Response payload consisting of recommendations."""

    user_id: str
    recommendations: list[MovieLensID]


class PrefVizItem(BaseModel):
    """Response payload for the preference visualization study."""

    item_id: str
    community_score: float
    score: float
    community_label: int
    label: int
    cluster: int
    title: Optional[str] = None


class EmotionContinuousInputSchema(BaseModel):
    """Payload for the emotions study slider controls."""

    emotion: Literal['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    switch: Literal['ignore', 'diverse', 'specified']
    weight: float


class EmotionDiscreteInputSchema(BaseModel):
    """Payload for the emotions study button controls."""

    emotion: Literal['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    weight: Literal['low', 'high', 'diverse', 'ignore']
