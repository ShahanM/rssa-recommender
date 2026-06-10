"""Base class for Matrix Factorization-based Recommenders.

File: mf_base.py
Project: RS:SA Recommender System (Clemson University)
Created Date: Friday, 1st September 2023
Author: Mehtab 'Shahan' Iqbal
Affiliation: Clemson University
----
Last Modified: Wednesday, 10th December 2025 3:15:57 am
Modified By: Mehtab 'Shahan' Iqbal (mehtabi@clemson.edu)
----
Copyright (c) 2025 Clemson University
License: MIT License (See LICENSE.md)
# SPDX-License-Identifier: MIT License
"""

import logging
import os
from typing import Optional, Union, cast

import binpickle
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from lenskit.algorithms import als
from lenskit.algorithms.mf_common import MFPredictor
from scipy.spatial import distance

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.common.schemas import MovieLensRating
from rssa_recommender.common.utils import get_and_unzip_resource

setup_logging()


log = logging.getLogger(__name__)
S3_BUCKET = os.environ['S3_BUCKET']

ITEM_POPULARITY_FILENAME = 'item_popularity.csv'
AVG_ITEM_SCORE_FILENAME = 'averaged_item_score.csv'
MODEL_FILENAME = 'model.bpk'
USER_HISTORY_FILENAME = 'user_history_lookup.parquet'
ANNOY_INDEX_FILENAME = 'annoy_index'
ANNOY_USERMAP_FILENAME = f'{ANNOY_INDEX_FILENAME}_map.csv'

MFModelType = Union[als.BiasedMF, als.ImplicitMF]
NumericType = Union[int, float]


class RSSABase:
    """Base class for Matrix Factorization-based Recommenders."""

    norm = 'L1'

    def __init__(self, asset_root: str, asset_bundle_key: str):
        """Initializes the RSSABase with model and data assets."""
        if S3_BUCKET:
            self.path = f'/tmp/{asset_root}'
        else:
            self.path = asset_root
        os.makedirs(self.path, exist_ok=True)
        log.info(f'Initializing RSSABase. Caching assets to {self.path}')

        asset_bundle_path = f'{asset_root}/{asset_bundle_key}'
        get_and_unzip_resource(S3_BUCKET, asset_bundle_path, self.path)

        self.item_popularity = pd.read_csv(f'{self.path}/{ITEM_POPULARITY_FILENAME}', dtype={'item': int})
        self.ave_item_score = pd.read_csv(f'{self.path}/{AVG_ITEM_SCORE_FILENAME}', dtype={'item': int})
        self.discounting_factor = self._init_discounting_factor(self.item_popularity)

        mf_model: MFPredictor = self._load_model_asset()
        model_instance: Optional[MFModelType] = self._get_typed_model_instance(mf_model)
        if model_instance is None:
            raise RuntimeError('Model was not loaded properly.')
        self.model: MFModelType = model_instance
        self.items = self.item_popularity.item.unique()
        log.info('RSSABase initialization complete.')

    def set_norm(self, norm: str) -> None:
        """Set the norm for distance calculations."""
        if norm.lower() not in ['l1', 'l2']:
            raise ValueError('The value of norm must be either L1, or L2.')
        self.norm = norm.upper()

    def _init_discounting_factor(self, item_popularity):
        max_count = item_popularity['count'].max()
        return 10 ** len(str(max_count))

    def _load_model_asset(self):
        return binpickle.load(f'{self.path}/{MODEL_FILENAME}')

    def _get_typed_model_instance(self, model: MFPredictor) -> Optional[Union[als.BiasedMF, als.ImplicitMF]]:
        if isinstance(model, als.BiasedMF):
            model = cast(als.BiasedMF, model)
        elif isinstance(model, als.ImplicitMF):
            model = cast(als.ImplicitMF, model)
        else:
            return None
        return model

    def _find_nearest_neighbors_annoy(self, new_user_vector: np.ndarray, num_neighbors: int) -> list[int]:
        """Finds K nearest neighbors using the pre-built Annoy index over the P matrix."""
        annoy_index, annoy_user_map = self._load_annoy_assets()
        internal_ids: list[int] = annoy_index.get_nns_by_vector(new_user_vector, num_neighbors, include_distances=False)
        del annoy_index

        external_ids: list[int] = [annoy_user_map[i] for i in internal_ids]
        del annoy_user_map

        return external_ids

    def _load_history_lookup_asset(self) -> pd.Series:
        """Loads the compact user history Parquet file and converts it to a dict/Series for quick lookup."""
        history_path = f'{self.path}/{USER_HISTORY_FILENAME}'
        history_df = pd.read_parquet(history_path)
        return history_df.set_index('user')['history_tuples']  # type: ignore

    def _load_annoy_assets(self):
        """Loads the pre-built Annoy index and the ID mapping table."""
        annoy_index_path = f'{self.path}/{ANNOY_INDEX_FILENAME}'
        user_map_path = f'{self.path}/{ANNOY_USERMAP_FILENAME}'

        user_feature_vector = self.model.user_features_
        if user_feature_vector is None:
            raise RuntimeError()

        dims = user_feature_vector.shape[1]
        index = AnnoyIndex(dims, 'angular')
        try:
            index.load(annoy_index_path)
        except Exception as e:
            raise FileNotFoundError(
                f'Annoy index file not found at {annoy_index_path}. Did you run training with --cluster_index?'
            ) from e

        user_map_df = pd.read_csv(user_map_path, index_col=0)
        return index, user_map_df.iloc[:, 0].to_dict()

    def _calculate_neighborhood_average(self, neighbor_ids: list[int], target_item: int, min_ratings: int = 1):
        """Calculates the average observed rating for a target item among the K nearest neighbors."""
        history_lookup_map = self._load_history_lookup_asset()
        ratings = []
        for user_id in neighbor_ids:
            history_tuples = history_lookup_map.get(user_id)
            if history_tuples:
                for item_id, rating in history_tuples:
                    if item_id == target_item:
                        ratings.append(rating)
                        break

        if len(ratings) < min_ratings:
            return None

        del history_lookup_map
        return np.mean(ratings)

    def _get_target_item_factors(self, item_ids: list[int]) -> tuple[np.ndarray, list[int]]:
        """Retrieves the Q (item factor) matrix subset corresponding to the list of item UUIDs."""
        item_vocab = self.model.item_index_
        item_codes_full = item_vocab.get_indexer(item_ids)

        valid_mask = np.greater_equal(item_codes_full, 0)
        target_item_codes = item_codes_full[valid_mask]
        Q_full_numpy = self.model.item_features_
        Q_target_slice = Q_full_numpy[target_item_codes, :]

        valid_item_ids = np.array(item_ids)[valid_mask].tolist()

        return Q_target_slice, valid_item_ids

    def predict(
        self,
        user_id: Union[str, int],
        ratings: Optional[list[MovieLensRating]] = None,
        limit: Optional[int] = None,
        include_rated: bool = False,
    ) -> pd.DataFrame:
        """Generates predictions for a new (out-of-sample) user using the trained LensKit Pipeline."""
        new_ratings = None
        rated_items = np.array([], dtype=np.int32)

        if ratings is not None:
            rated_items = np.array([rating.item_id for rating in ratings], dtype=np.int32)
            new_ratings = pd.Series([rating.rating for rating in ratings], index=rated_items, dtype=np.float64)

        if new_ratings is None:
            als_preds = self.model.predict_for_user(user_id, self.items)
        else:
            als_preds = self.model.predict_for_user(user_id, self.items, new_ratings)

        als_preds = als_preds.sort_values(ascending=False)

        als_preds_df: pd.DataFrame = als_preds.to_frame().reset_index()
        als_preds_df.columns = ['item', 'score']
        als_preds_df['item'] = als_preds_df['item'].astype(int)

        if not include_rated and len(rated_items) > 0:
            als_preds_df = als_preds_df[~als_preds_df['item'].isin(rated_items)]  # type: ignore

        if limit is not None:
            als_preds_df = als_preds_df.head(limit)

        return als_preds_df

    def predict_discounted(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        discount_factor: Optional[int] = None,
        coeff: float = 0.5,
        include_rated: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict the ratings for the new items for the live user."""
        als_preds = self.predict(user_id, ratings)

        factor = discount_factor if discount_factor is not None else self.discounting_factor

        als_preds: pd.DataFrame = pd.merge(als_preds, self.item_popularity, on='item')
        als_preds['discounted_score'] = als_preds['score'] - coeff * (als_preds['count'] / factor)

        als_preds.sort_values(by='discounted_score', ascending=False, inplace=True)

        if not include_rated:
            rated_ids = {rating.item_id for rating in ratings}
            als_preds = als_preds[~als_preds['item'].isin(rated_ids)]  # type: ignore

        if limit is not None:
            als_preds = als_preds.head(limit)

        return als_preds

    def predict_diverse(
        self, user_id: str, ratings: list[MovieLensRating], *, candidate_pool_size: int = 100, limit: int = 50
    ) -> list[int]:
        """Diversifies recommendations by maximizing distance in the model's latent feature space."""
        candidate_df = self.predict(user_id, ratings, limit=candidate_pool_size, include_rated=False)

        if candidate_df.empty:
            return []

        item_index = self.model.item_index_
        candidate_items = candidate_df['item'].to_numpy()

        internal_indices = []
        valid_candidate_items = []

        for iid in candidate_items:
            try:
                internal_indices.append(item_index.get_loc(iid))
                valid_candidate_items.append(iid)
            except KeyError:
                continue

        latent_vectors = self.model.item_features_[internal_indices, :]

        valid_candidates_df: pd.DataFrame = candidate_df[candidate_df['item'].isin(valid_candidate_items)]  # type: ignore

        diverse_recs_df, _ = self._diversify_item_feature(
            candidates=valid_candidates_df,
            vectors=latent_vectors,
            items=np.array(valid_candidate_items),
            sampling_size=limit,
        )

        return diverse_recs_df['item'].astype(int).to_list()

    def _diversify_item_feature(
        self,
        candidates: pd.DataFrame,
        vectors: np.ndarray,
        items: np.ndarray,
        sampling_size: int = 50,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Diversify items based on their feature vectors using a greedy algorithm."""
        if 'item' not in candidates.columns:
            candidates = candidates.reset_index()
        candidates = candidates.set_index('item')

        if len(vectors) == 0:
            return candidates.reset_index(), pd.DataFrame()

        vectors_df = pd.DataFrame(vectors, index=items)
        candidate_vectors_df = vectors_df[vectors_df.index.isin(candidates.index)].reindex(candidates.index)

        candidate_vectors = candidate_vectors_df.to_numpy()
        candidate_items = candidate_vectors_df.index.to_numpy()
        n_candidates = len(candidate_items)
        feature_dim = candidate_vectors.shape[1]

        diverse_item_indices = np.full(sampling_size, -1, dtype=int)
        diverse_vectors = np.zeros((sampling_size, feature_dim))

        selected_mask = np.zeros(n_candidates, dtype=bool)

        metric = 'cityblock' if self.norm == 'L1' else 'euclidean'
        centroid = np.mean(candidate_vectors, axis=0)
        dists_to_centroid = distance.cdist(candidate_vectors, centroid.reshape(1, -1), metric=metric).flatten()

        first_idx = np.argmin(dists_to_centroid)
        selected_mask[first_idx] = True
        diverse_item_indices[0] = first_idx
        diverse_vectors[0] = candidate_vectors[first_idx]

        min_distances = distance.cdist(candidate_vectors, diverse_vectors[0:1], metric=metric).flatten()

        for i in range(1, min(sampling_size, n_candidates)):
            remaining_indices = np.where(~selected_mask)[0]

            best_candidate_idx_in_remaining = np.argmax(min_distances[remaining_indices])
            next_idx = remaining_indices[best_candidate_idx_in_remaining]

            selected_mask[next_idx] = True
            diverse_item_indices[i] = next_idx
            diverse_vectors[i] = candidate_vectors[next_idx]

            new_distances = distance.cdist(candidate_vectors, diverse_vectors[i : i + 1], metric=metric).flatten()
            min_distances = np.maximum(min_distances, new_distances)

        valid_indices = diverse_item_indices[diverse_item_indices != -1]
        diverse_item_ids = candidate_items[valid_indices]
        final_diverse_vectors = diverse_vectors[: len(valid_indices)]

        diverse_vectors_df = pd.DataFrame(final_diverse_vectors, index=diverse_item_ids)

        candidates.index.name = 'item'
        recommendations = candidates.loc[diverse_item_ids].reset_index()

        if 'item' not in recommendations.columns and 'index' in recommendations.columns:
            recommendations.rename(columns={'index': 'item'}, inplace=True)

        return recommendations, diverse_vectors_df

    def get_user_feature_vector(self, ratings: list[MovieLensRating]) -> Optional[np.ndarray]:
        """Extracts the new user's latent feature vector (q_u)."""
        rated_items = np.array([rating.item_id for rating in ratings], dtype=np.int32)
        new_ratings = pd.Series([rating.rating for rating in ratings], index=rated_items, dtype=np.float64)

        ri_idxes = self.model.item_index_.get_indexer_for(new_ratings.index)
        ri_good = ri_idxes >= 0
        ri_it = ri_idxes[ri_good]
        ri_val = new_ratings.values[ri_good]

        if isinstance(self.model, als.ImplicitMF):
            self.model = cast(als.ImplicitMF, self.model)
            ri_val *= self.model.weight
            return als._train_implicit_row_lu(ri_it, ri_val, self.model.item_features_, self.model.OtOr_)
        elif isinstance(self.model, als.BiasedMF):
            self.model = cast(als.BiasedMF, self.model)
            ureg = self.model.regularization
            return als._train_bias_row_lu(ri_it, ri_val, self.model.item_features_, ureg)

        return None


def normalize(
    value, new_min: NumericType, new_max: NumericType, cur_min: NumericType, cur_max: NumericType
) -> NumericType:
    """Normalizes a value from current range to a new range."""
    new_range = new_max - new_min
    cur_range = cur_max - cur_min
    new_value = new_range * (value - cur_min) / cur_range + new_min
    return new_value
