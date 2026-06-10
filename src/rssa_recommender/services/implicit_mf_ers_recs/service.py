"""Implicit MF ERS Recommender Service."""

import logging
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.common.mf_base import RSSABase
from rssa_recommender.common.schemas import EmotionContinuousInputSchema, EmotionDiscreteInputSchema, MovieLensRating

setup_logging()
log = logging.getLogger(__name__)


EmotionInputUnion = Union[list[EmotionDiscreteInputSchema], list[EmotionContinuousInputSchema]]
EmoTagInput = tuple[str, float]


class ImplicitMFErsRecsService(RSSABase):
    """Implicit MF ERS Recommender Service."""

    norm = 'L1'

    def __init__(self, asset_root: str, asset_bundle_key: str):
        """Initialize the Implicit MF ERS Recommender Service."""
        super().__init__(asset_root, asset_bundle_key)
        self.emotion_tags = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        self.discounting_coefficient = 0.5
        self.emotion_lookup_asset = None

    def _load_emotion_lookup_asset(self) -> pd.DataFrame:
        """Load the item-emotion lookup asset."""
        if self.emotion_lookup_asset is not None:
            return self.emotion_lookup_asset
        emotions_path = f'{self.path}/item_emotion_lookup.parquet'
        emotions_df = pd.read_parquet(emotions_path)
        self.emotion_lookup_asset = emotions_df
        return emotions_df

    def predict_with_emotions(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        limit: int = 10,
        *,
        strategy: Literal['top_n', 'diverse_n'] = 'top_n',
        candidate_pool_size: int = 500,
        sampling_size: int = 50,
        emotion_input: Optional[list[Union[dict, Any]]] = None,
        ranking_strategy: str = 'distance',
        emotion_discrete_cutoffs: tuple[float, float] = (0.3, 0.8),
        div_criteria: str = 'unspecified',
    ) -> list[int]:
        """Get emotion-based recommendations."""
        if not emotion_input:
            if strategy == 'top_n':
                return self.predict_discounted(user_id, ratings, limit=limit)['item'].astype(int).to_list()
            elif strategy == 'diverse_n':
                diverse_n = self._diversify_with_emotions(
                    user_id, ratings, candidate_pool_size=candidate_pool_size, sampling_size=sampling_size
                )
                return diverse_n['item'].head(limit).astype(int).to_list()
        recs = self._generate_tuned_recs(
            user_id,
            ratings,
            limit=limit,
            emotion_input=emotion_input,
            diversify=True,
            candidate_pool_size=candidate_pool_size,
            sampling_size=sampling_size,
            ranking_strategy=ranking_strategy,
            emotion_discrete_cutoffs=emotion_discrete_cutoffs,
            div_criteria=div_criteria,
        )
        return recs['item'].head(limit).astype(int).to_list()

    # def predict_top_n(
    #     self,
    #     user_id: str,
    #     ratings: list[MovieLensRating],
    #     limit: int = 10,
    #     *,
    #     emotion_input: Optional[list[Union[dict, Any]]] = None,
    #     candidate_pool_size: int = 500,
    #     ranking_strategy: str = 'distance',
    #     emotion_discrete_cutoffs: tuple[float, float] = (0.3, 0.8),
    # ) -> list[int]:
    #     """Predict Top-N items or Tuned Top-N items."""
    #     if not emotion_input:
    #         top_n_preds = self.predict_discounted(user_id, ratings, limit=limit)
    #         return top_n_preds['item'].astype(int).to_list()

    #     recs = self._generate_tuned_recs(
    #         user_id,
    #         ratings,
    #         limit=limit,
    #         emotion_input=emotion_input,
    #         candidate_pool_size=candidate_pool_size,
    #         ranking_strategy=ranking_strategy,
    #         emotion_discrete_cutoffs=emotion_discrete_cutoffs,
    #     )
    #     return recs['item'].head(limit).astype(int).to_list()

    # def predict_diverse_n(
    #     self,
    #     user_id: str,
    #     ratings: list[MovieLensRating],
    #     limit: int = 10,
    #     *,
    #     emotion_input: Optional[list[Union[dict, Any]]] = None,
    #     candidate_pool_size: int = 500,
    #     sampling_size: int = 50,
    #     ranking_strategy: str = 'distance',
    #     emotion_discrete_cutoffs: tuple[float, float] = (0.3, 0.8),
    #     div_criteria: str = 'unspecified',
    # ) -> list[int]:
    #     """Predict Diverse-N items or Tuned Diverse-N items."""
    #     if not emotion_input:
    #         diverse_n = self._diversify_with_emotions(
    #             user_id, ratings, candidate_pool_size=candidate_pool_size, sampling_size=sampling_size
    #         )
    #         return diverse_n['item'].head(limit).astype(int).to_list()

    #     recs = self._generate_tuned_recs(
    #         user_id,
    #         ratings,
    #         limit=limit,
    #         emotion_input=emotion_input,
    #         diversify=True,
    #         candidate_pool_size=candidate_pool_size,
    #         sampling_size=sampling_size,
    #         ranking_strategy=ranking_strategy,
    #         emotion_discrete_cutoffs=emotion_discrete_cutoffs,
    #         div_criteria=div_criteria,
    #     )
    #     return recs['item'].head(limit).astype(int).to_list()

    def _get_candidate_item(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        limit: int = 500,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get candidate items and their emotion features."""
        preds = self.predict_discounted(user_id, ratings, limit=limit)
        candidates = preds.reset_index()
        candidate_ids = candidates.item.unique()
        emotions_lookup = self._load_emotion_lookup_asset()

        valid_ids = [i for i in candidate_ids if i in emotions_lookup.index]
        candidate_items_emotions = emotions_lookup.loc[valid_ids]
        return candidates, candidate_items_emotions

    def _process_emotion_input(
        self,
        emotion_input: list[Union[dict, Any]],
        emotion_cutoffs: tuple[float, float] = (0.3, 0.8),
    ) -> tuple[list[EmoTagInput], list[str]]:
        """Process emotion input safely from either dicts or Pydantic models."""
        extracted_input: list[EmoTagInput] = []
        unspecified_tags: list[str] = []

        lowval, highval = emotion_cutoffs
        lowval = max(0.0, lowval)
        highval = min(1.0, highval)

        emo_dict = {}
        for e in emotion_input:
            if isinstance(e, dict):
                emotion_key = e.get('emotion', '').lower()
                weight_val = e.get('weight', '')
            else:
                emotion_key = getattr(e, 'emotion', '').lower()
                weight_val = getattr(e, 'weight', '')

            if emotion_key:
                emo_dict[emotion_key] = weight_val

        for emo in self.emotion_tags:
            w = emo_dict.get(emo, '')

            if isinstance(w, str):
                w = w.strip().lower()

            if w == 'low':
                extracted_input.append((emo, lowval))
            elif w == 'high':
                extracted_input.append((emo, highval))
            elif w != '':
                try:
                    extracted_input.append((emo, float(w)))
                except ValueError:
                    unspecified_tags.append(emo)
            else:
                unspecified_tags.append(emo)

        return extracted_input, unspecified_tags

    def _get_distance_to_input(
        self,
        candidate_items: pd.DataFrame,
        emotion_inputs: list[EmoTagInput],
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate distance of candidate items to the emotion input vector."""
        emo_tags, emo_vals = zip(*emotion_inputs)
        candidate_ids = candidate_items['item'].to_numpy()
        candidate_ndarr = candidate_items[list(emo_tags)].to_numpy()

        is_ascending = kwargs.get('is_ascending', True)
        emo_vector = np.array(emo_vals)
        scale_factor = candidate_ndarr.max(axis=0) - candidate_ndarr.min(axis=0)
        if np.any(scale_factor == 0):
            log.warning('One or more emotion dimensions have zero variance among candidates. Skipping scaling.')
        else:
            emo_vector = scale_factor * emo_vector
        metric = 'cityblock' if self.norm == 'L1' else 'euclidean'
        dist = distance.cdist(candidate_ndarr, emo_vector.reshape(1, -1), metric=metric).flatten()

        dist_to_input_df = pd.DataFrame(
            {'item': candidate_ids, 'distance': dist},
            columns=['item', 'distance'],  # type: ignore
        )

        return dist_to_input_df.sort_values(by='distance', ascending=is_ascending)

    def _rank_candidates_by_emotion(
        self,
        candidates: pd.DataFrame,
        candidate_emotions: pd.DataFrame,
        emotion_inputs: list[EmoTagInput],
        is_ascending: bool = False,
    ) -> pd.DataFrame:
        """Rank candidate items based on weighted emotion scores."""
        candidates['ori_rank'] = np.arange(len(candidates), 0, -1)
        recs_emotions_df = pd.merge(candidates, candidate_emotions, on='item')

        emo_tags, emo_vals = zip(*emotion_inputs)

        col_query = ['ori_rank'] + list(emo_tags)
        candidates_df_scaled = recs_emotions_df[col_query].copy()

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(candidates_df_scaled.to_numpy())
        candidates_df_scaled = pd.DataFrame(scaled_values, columns=col_query)  # type: ignore
        user_emotion_vals_np = np.array(emo_vals)
        scaled_emotions = candidates_df_scaled[list(emo_tags)].values
        scaled_rank = candidates_df_scaled['ori_rank'].values
        total_emotion_weight = np.sum(np.absolute(user_emotion_vals_np))

        if total_emotion_weight == 0:
            recs_emotions_df['new_rank_score'] = scaled_rank
        else:
            alpha = np.clip(np.max(np.absolute(user_emotion_vals_np)), 0.0, 1.0)
            normalized_user_vals = user_emotion_vals_np / total_emotion_weight
            emotion_score = alpha * np.sum(scaled_emotions * normalized_user_vals, axis=1)
            rank_score = (1.0 - alpha) * scaled_rank
            recs_emotions_df['new_rank_score'] = emotion_score + rank_score

        emotion_score = np.sum(scaled_emotions * user_emotion_vals_np, axis=1)
        rank_weight = 1 - np.sum(np.absolute(user_emotion_vals_np))
        rank_score = rank_weight * scaled_rank

        recs_emotions_df['new_rank_score'] = emotion_score + rank_score

        recs_emotions_df.sort_values(by='new_rank_score', ascending=is_ascending, inplace=True)
        return recs_emotions_df

    def _diversify_with_emotions(
        self, user_id: str, ratings: list[MovieLensRating], candidate_pool_size: int = 500, sampling_size: int = 50
    ) -> pd.DataFrame:
        """Diversify candidate items based on emotion features."""
        candidates, candidate_emotions = self._get_candidate_item(user_id, ratings, candidate_pool_size)
        item_ids = candidate_emotions.index.to_numpy()
        item_emotions_ndarray = candidate_emotions[self.emotion_tags].to_numpy()
        rec_diverse, _ = self._diversify_item_feature(
            candidates, item_emotions_ndarray, item_ids, sampling_size=sampling_size
        )
        return rec_diverse

    def _generate_tuned_recs(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        limit: int = 500,
        *,
        emotion_input: list[Union[dict, Any]],
        diversify: bool = False,
        candidate_pool_size: int = 500,
        sampling_size: int = 50,
        ranking_strategy: str = 'distance',
        emotion_discrete_cutoffs: tuple[float, float] = (0.3, 0.8),
        div_criteria: str = 'unspecified',
    ) -> pd.DataFrame:
        """Generate tuned recommendations based on emotion input."""
        emo_input, unspecified_tags = self._process_emotion_input(
            emotion_input, emotion_cutoffs=emotion_discrete_cutoffs
        )
        candidates, candidate_emotions = self._get_candidate_item(user_id, ratings, candidate_pool_size)

        diverse_recs: Optional[pd.DataFrame] = None

        if 'item' not in candidate_emotions.columns:
            candidate_emotions = candidate_emotions.reset_index()

        if not emo_input:
            if diversify:
                query_tags = self.emotion_tags if div_criteria != 'unspecified' else unspecified_tags
                candidate_ndarray = candidate_emotions[query_tags].to_numpy()
                rec_items, _ = self._diversify_item_feature(
                    candidates, candidate_ndarray, candidate_emotions['item'].to_numpy(), sampling_size=sampling_size
                )
                return rec_items.head(limit)
            return candidates.head(limit)
        ranked_candidates: pd.DataFrame
        if ranking_strategy == 'distance':
            ranked_candidates = self._get_distance_to_input(candidate_emotions, emo_input)
            ranked_candidates = pd.merge(
                ranked_candidates, candidates[['item', 'score', 'discounted_score']], on='item'
            )
        elif ranking_strategy == 'weighted':
            candidates_subset: pd.DataFrame = candidates[['item', 'discounted_score']].copy()  # type: ignore
            ranked_candidates = self._rank_candidates_by_emotion(candidates_subset, candidate_emotions, emo_input)
        else:
            raise NotImplementedError

        if diversify:
            aligned_pool_size = max(sampling_size * 2, 100)
            aligned_items = ranked_candidates.head(aligned_pool_size)
            aligned_emotions = candidate_emotions[candidate_emotions['item'].isin(aligned_items['item'])]
            aligned_emotions = aligned_emotions.set_index('item').loc[aligned_items['item']].reset_index()

            query_tags = self.emotion_tags if div_criteria != 'unspecified' else unspecified_tags
            aligned_ndarray = aligned_emotions[query_tags].to_numpy()
            diverse_recs, _ = self._diversify_item_feature(
                aligned_items, aligned_ndarray, aligned_items['item'].to_numpy(), sampling_size=sampling_size
            )
            return diverse_recs.head(limit)

        return ranked_candidates.head(limit)
