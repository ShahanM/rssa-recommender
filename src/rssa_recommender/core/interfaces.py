"""Interfaces for different algorithms."""

from typing import Any, List, Protocol

from rssa_recommender.common.schemas import MovieLensRating


class RecommenderServiceProtocol(Protocol):
    """Centralized generic protocol to standardize calls signature."""

    def predict_top_n(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the Top N recommendation using the BiasedMF model."""
        ...

    def predict_diverse_n(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the emotions diversified Top N recommendation using the ImplicitMF model."""
        ...

    def predict_with_community_scores(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the community scored preditions using the BiasedMF model."""
        ...

    def predict_discounted_top_n(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the Top N recommendation using the ImplicitMF model."""
        ...

    def predict_controversial_items(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the Top N recommendation using the ImplicitMF model."""
        ...

    def predict_hate_items(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the predicted to dislike recommendation using the ImplicitMF model."""
        ...

    def predict_hip_items(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the Top N recommendation using the ImplicitMF model."""
        ...

    def predict_no_clue_items(self, user_id: str, ratings: List[MovieLensRating], limit: int, **kwargs) -> Any:
        """Returns the Top N recommendation using the ImplicitMF model."""
        ...
