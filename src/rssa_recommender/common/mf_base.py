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


class RSSABase:
    """Base class for Matrix Factorization-based Recommenders."""

    def __init__(self, asset_root: str, asset_bundle_key: str):
        """Initializes the RSSABase with model and data assets.

        Args:
            asset_root: The root directory for caching assets.
            asset_bundle_key: The S3 key for the asset bundle zip file.
        """
        self.path = f'/tmp/{asset_root}'
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
        """Finds K nearest neighbors using the pre-built Annoy index over the P matrix.

        Args:
            new_user_vector: The new user's latent feature vector (q_u).
            num_neighbors: The number of nearest neighbors to retrieve.

        Returns:
            list[int]: The list of external user IDs of the nearest neighbors.
        """
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

        # Convert the DataFrame back to a Series indexed by user ID for O(1) lookup speed
        # The Series values are the list of (item_id, rating) tuples
        return history_df.set_index('user')['history_tuples']

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

        # Load User Map (Annoy ID -> user ID)
        user_map_df = pd.read_csv(user_map_path, index_col=0)

        # Convert the Series/DataFrame to a fast dictionary lookup (internal ID -> external ID)
        return index, user_map_df.iloc[:, 0].to_dict()

    def _calculate_neighborhood_average(self, neighbor_ids: list[int], target_item: int, min_ratings: int = 1):
        """Calculates the average observed rating for a target item among the K nearest neighbors."""
        history_lookup_map = self._load_history_lookup_asset()
        ratings = []
        for user_id in neighbor_ids:
            # Lookup the neighbor's history (O(1) operation)
            history_tuples = history_lookup_map.get(user_id)

            if history_tuples:
                for item_id, rating in history_tuples:
                    if str(item_id) == str(target_item):
                        ratings.append(rating)
                        break

        if len(ratings) < min_ratings:
            return None

        del history_lookup_map
        return np.mean(ratings)

    def _get_target_item_factors(self, item_ids: list[int]) -> tuple[np.ndarray, list]:
        """Retrieves the Q (item factor) matrix subset corresponding to the list of item UUIDs.

        Args:
            item_ids: The list of external item IDs to retrieve.

        Returns:
            np.ndarray: The sliced Q matrix (N_target_items x F_features).
        """
        item_vocab = self.model.item_index_

        # This returns an array of integer indices, with -1 for Out-of-Vocabulary (OOV) items.
        item_codes_full = item_vocab.get_indexer(item_ids)

        # Filter out OOV items (where code is -1)
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
        """Generates predictions for a new (out-of-sample) user using the trained LensKit Pipeline.

        Args:
            user_id: The external user ID of the live user.
            ratings: New ratings of the live user.
            limit: If provided, limits the number of returned predictions to this number.
            include_rated: If False, excludes items that were rated by the user in the returned predictions.

        Returns:
            pd.DataFrame: DataFrame containing item and score columns.
        """
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

        als_preds_df = als_preds.to_frame().reset_index()
        als_preds_df.columns = ['item', 'score']
        als_preds_df['item'] = als_preds_df['item'].astype(int)

        if not include_rated and len(rated_items) > 0:
            als_preds_df = als_preds_df[~als_preds_df['item'].isin(rated_items)]

        if limit is not None:
            als_preds_df = als_preds_df.head(limit)

        return als_preds_df

    def predict_discounted(
        self,
        userid: str,
        ratings: list[MovieLensRating],
        discount_factor: Optional[int] = None,
        coeff: float = 0.5,
        include_rated: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Predict the ratings for the new items for the live user.

        Discount the score of the items based on their popularity and
        compute the RSSA score.

        Args:
            userid: The external user ID of the live user.
            ratings: New ratings of the live user.
            discount_factor: The discounting factor to use. If None, uses the class default.
            coeff: The coefficient to scale the popularity penalty.
            include_rated: If False, excludes items that were rated by the user in the returned predictions.
            limit: If provided, limits the number of returned predictions to this number.

        Returns:
            pd.DataFrame: ['item', 'score', 'count', 'rank', 'discounted_score']
                The dataframe is sorted by the discounted_score in descending order.
        """
        als_preds = self.predict(userid, ratings)

        factor = discount_factor if discount_factor is not None else self.discounting_factor

        als_preds = pd.merge(als_preds, self.item_popularity, on='item')
        als_preds['discounted_score'] = als_preds['score'] - coeff * (als_preds['count'] / factor)

        als_preds.sort_values(by='discounted_score', ascending=False, inplace=True)

        if not include_rated:
            rated_ids = {rating.item_id for rating in ratings}
            als_preds = als_preds[~als_preds['item'].isin(rated_ids)]

        if limit is not None:
            als_preds = als_preds.head(limit)

        return als_preds

    def get_user_feature_vector(self, ratings: list[MovieLensRating]) -> Optional[np.ndarray]:
        """Extracts the new user's latent feature vector (q_u).

        Args:
            ratings: The list of new ratings provided by the live user.

        Returns:
            np.ndarray: The projected user feature vector (q_u).
        """
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


def normalize(value, new_min, new_max, cur_min, cur_max):
    """Normalizes a value from current range to a new range.

    Args:
        value: The value to normalize.
        new_min: The minimum of the new range.
        new_max: The maximum of the new range.
        cur_min: The minimum of the current range.
        cur_max: The maximum of the current range.

    Returns:
        The normalized value.
    """
    new_range = new_max - new_min
    cur_range = cur_max - cur_min
    new_value = new_range * (value - cur_min) / cur_range + new_min
    return new_value
