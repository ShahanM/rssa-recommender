"""Unified Service for Implicit MF Recommendations.

Combines logic from AlternateRS (alt_recs) and PreferenceCommunity (pref_comm).
"""

import logging
import os
import random
import time
from typing import Literal, Optional

import binpickle
import numpy as np
import pandas as pd
from lenskit.algorithms import als

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.common.mf_base import RSSABase
from rssa_recommender.common.schemas import MovieLensRating
from rssa_recommender.common.utils import get_and_unzip_resource

setup_logging()
log = logging.getLogger(__name__)

S3_BUCKET = os.environ.get('S3_BUCKET')


class ImplicitMFRecsService(RSSABase):
    """Service to provide Implicit MF recommendations, AlternateRS routes, and Advisor Profiles."""

    def __init__(self, asset_root: str, asset_bundle_key: str, resampled_asset_bundle_key: Optional[str] = None):
        """Initialize the unified service.

        Downloads and loads the main model (RSSABase) and specific assets for sub-services.
        """
        super().__init__(asset_root, asset_bundle_key)

        if resampled_asset_bundle_key:
            log.info('Initializing AlternateRS assets...')
            asset_bundle_path = f'{self.path}/{resampled_asset_bundle_key}'
            if S3_BUCKET:
                get_and_unzip_resource(S3_BUCKET, asset_bundle_path, self.path)

    # --------------------------------------------------------------------------
    # PreferenceCommunity Methods (from pref_comm/service.py)
    # --------------------------------------------------------------------------

    def predict_advisors_with_profile(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        strategy: Literal['top_n', 'diverse_n', 'compromised_diverse_n'],
        limit=10,
    ) -> list:
        """Identifies K nearest neighbors (advisors) and gets profile/rec for each.

        Community Advisors types:
        1. Low Diversity - Top N (Strategy: 'top_n')
        2. High Diversity no compromise - Diverse N (Strategy: 'diverse_n')
        3. High Diversity compromise - Midpoint Interpolation (Strategy: 'compromised_diverse_n')
        """
        advisors = []

        predicted_items = []
        if strategy == 'top_n':
            predicted_items = self.predict_top_n(user_id, ratings, limit=limit)
        elif strategy == 'diverse_n':
            predicted_items = self.predict_diverse(user_id, ratings, limit=limit)
        elif strategy == 'compromised_diverse_n':
            predicted_items = self._generate_compromise_recommendations(user_id, ratings, limit=limit)

        global_claimed_items: set[int] = set()
        claimed_anchors: set[int] = set()

        for target_rec in predicted_items:
            profile = self._generate_masquerade_profile(
                target_rec,
                ratings,
                global_claimed_items=global_claimed_items,
                claimed_anchors=claimed_anchors,
                profile_size=limit,
                anchor_pos_item=True,
            )
            advisors.append({'id': target_rec, 'recommendation': target_rec, 'profile_top_n': profile})

        return advisors

    def _generate_masquerade_profile(
        self,
        target_item_id: int,
        ratings: list[MovieLensRating],
        global_claimed_items: set[int],
        claimed_anchors: set[int],
        profile_size: int = 10,
        anchor_pos_item: bool = False,
    ) -> list[int]:
        """Reverse-engineers an advisor profile by finding items similar to the target recommendation."""
        exclude_items = {r.item_id for r in ratings}
        item_features = self.model.item_features_
        item_index = self.model.item_index_

        try:
            internal_idx = item_index.get_loc(target_item_id)
        except KeyError:
            return []

        target_vector = item_features[internal_idx, :]
        similarities = item_features @ target_vector
        most_similar_indices = np.argsort(similarities)[::-1]

        profile = []

        if anchor_pos_item:
            positive_rated = [
                r.item_id
                for r in ratings
                if r.rating >= 4.0 and r.item_id != target_item_id and r.item_id not in claimed_anchors
            ]
            if positive_rated:
                anchor_item = random.choice(positive_rated)
                profile.append(anchor_item)
                claimed_anchors.add(anchor_item)
                exclude_items.discard(anchor_item)

        for idx in most_similar_indices:
            candidate_id = int(item_index[idx])

            if candidate_id != target_item_id and candidate_id not in exclude_items:
                profile.append(candidate_id)
                if len(profile) == profile_size:
                    break

        return profile

    def _generate_compromise_recommendations(
        self, user_id: str, ratings: list[MovieLensRating], limit: int = 10
    ) -> list[int]:
        """Calculates compromise recommendations via midpoint interpolation."""
        item_features = self.model.item_features_
        item_index = self.model.item_index_
        rated_items = {r.item_id for r in ratings}

        v_pref = self.get_user_feature_vector(ratings)
        if v_pref is None:
            return []

        diverse_targets = self.predict_diverse(user_id, ratings, limit=limit)
        compromise_recs = []

        for div_item_id in diverse_targets:
            try:
                div_idx = item_index.get_loc(div_item_id)
            except KeyError:
                continue

            v_div = item_features[div_idx, :]
            v_mid = (v_pref + v_div) / 2.0
            similarities = item_features @ v_mid

            sorted_indices = np.argsort(similarities)[::-1]

            for idx in sorted_indices:
                candidate_id = int(item_index[idx])

                # The compromise item must not be:
                # 1. A movie they've already seen
                # 2. The exact diverse target we are stepping away from
                # 3. A compromise item already assigned to another advisor
                if (
                    candidate_id not in rated_items
                    and candidate_id != div_item_id
                    and candidate_id not in compromise_recs
                ):
                    compromise_recs.append(candidate_id)
                    break

        return compromise_recs

    # --------------------------------------------------------------------------
    # AlternateRS Methods (from alt_recs/service.py)
    # --------------------------------------------------------------------------
    def predict_top_n(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict standard top-N items for the user."""
        return self.predict(user_id, ratings, limit=limit)['item'].astype(int).to_list()

    def predict_discounted_top_n(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict top-N items with a popularity discount applied."""
        return self.predict_discounted(user_id, ratings, limit=limit)['item'].astype(int).to_list()

    def predict_hate_items(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict items the user is highly likely to dislike based on reverse margin."""
        preds = self.predict_discounted(user_id, ratings)
        preds = pd.merge(preds, self.ave_item_score, on='item')
        preds['margin_discounted'] = preds['ave_discounted_score'] - preds['score']
        preds = preds.sort_values(by='margin_discounted', ascending=False).head(limit)
        return preds['item'].astype(int).to_list()

    def predict_hip_items(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict items that are highly rated but obscure (long-tail)."""
        num_bs = 1000
        top_n = self.predict_discounted(user_id, ratings, limit=num_bs)
        hip_items = top_n.sort_values(by='count', ascending=True).head(limit)
        return hip_items['item'].astype(int).to_list()

    def predict_no_clue_items(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict items with the highest uncertainty/variance across resampled models."""
        resampled_df = self._high_std(user_id, ratings)
        rated_items = {r.item_id for r in ratings}
        resampled_df = resampled_df[~resampled_df['item'].isin(rated_items)]  # type: ignore
        resampled_df = resampled_df.sort_values(by='std', ascending=False).head(limit)  # type: ignore
        return resampled_df['item'].astype(int).to_list()

    def predict_controversial_items(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict items that polarize the user's nearest neighbor community."""
        search_space_k = 20
        user_features = self.get_user_feature_vector(ratings)
        if user_features is None:
            return []

        annoy_index, annoy_user_map = self._load_annoy_assets()
        rated_items = {r.item_id for r in ratings}
        neighborhood = annoy_index.get_nns_by_vector(user_features, search_space_k)
        variance = self._controversial(neighborhood, annoy_user_map)
        del annoy_index

        variance_wo_rated = variance[~variance['item'].isin(rated_items)]  # type: ignore
        del annoy_user_map

        controversial_items = variance_wo_rated.sort_values(by='variance', ascending=False).head(limit)  # type: ignore
        return controversial_items['item'].astype(int).to_list()

    def _high_std(self, user_id: str, ratings: list[MovieLensRating]):
        """Helper to calculate standard deviation across k resampled models."""
        all_resampled_df = pd.DataFrame(self.items, columns=['item'])  # type: ignore
        rated_items = np.array([rating.item_id for rating in ratings], dtype=np.int32)
        new_ratings = pd.Series([rating.rating for rating in ratings], index=rated_items, dtype=np.float64)
        is_dev = os.environ.get('ENV', 'production') == 'development'
        n_models = 5 if is_dev else 20  # FIXME: we should keep a model manifest but for now we hardcode
        for i in range(1, n_models + 1):
            filename = f'{self.path}/resampled_model_{i}.bpk'
            model: als.ImplicitMF = binpickle.load(filename)
            items_in_sample = model.item_index_.to_numpy()
            resampled_preds = model.predict_for_user(user_id, items_in_sample, new_ratings)

            resampled_df = resampled_preds.to_frame().reset_index()
            col = f'score{i}'
            resampled_df.columns = ['item', col]

            all_resampled_df = pd.merge(all_resampled_df, resampled_df, on='item')
            del model

        preds_only_df = all_resampled_df.drop(columns=['item']).apply(pd.to_numeric, errors='coerce')
        all_resampled_df['std'] = np.nanstd(preds_only_df, axis=1)
        all_items_std_df = all_resampled_df[['item', 'std']]
        all_items_std_df = pd.merge(all_items_std_df, self.item_popularity, on='item')
        return all_items_std_df

    def _controversial(self, neighborhood_annoy_ids: list[int], user_map_lookup: dict[int, int]):
        """Helper to calculate rating variance across a user's nearest neighbors."""
        start = time.time()
        log.info(f'Starting vectorized calculation of variance for {len(neighborhood_annoy_ids)} neighbors...')

        external_neighbor_ids = [user_map_lookup.get(aid) for aid in neighborhood_annoy_ids]
        external_neighbor_ids = [uid for uid in external_neighbor_ids if uid is not None]

        try:
            internal_indices = np.arange(len(self.model.user_index_))
            external_internal_map = pd.Series(data=internal_indices, index=self.model.user_index_)
            internal_index_series = external_internal_map.reindex(external_neighbor_ids)
            model_internal_indices = internal_index_series.dropna().astype(int).values.tolist()
        except KeyError as e:
            log.error(f'Failed to find user from neighbor index: {e}')
            return pd.DataFrame(columns=['item', 'variance'])  # type: ignore

        global_bias = getattr(self.model, 'global_bias', 0.0)
        user_features = self.model.user_features_
        item_features = self.model.item_features_
        item_index = self.model.item_index_

        if user_features is None:
            log.error('Model does not have user_features')
            return pd.DataFrame(columns=['item', 'variance'])  # type: ignore

        neighbor_features = user_features[model_internal_indices, :]
        prediction_matrix = neighbor_features @ item_features.T
        prediction_matrix += global_bias

        item_variance_vector = np.nanvar(prediction_matrix, axis=0)
        scores_df = pd.DataFrame({'item': item_index, 'variance': item_variance_vector})

        scores_df = pd.merge(scores_df, self.item_popularity, on='item')
        log.info(f'Time spent (vectorized controversial): {(time.time() - start):.4f}s.')
        return scores_df[['item', 'variance', 'count', 'rank_popular']]
