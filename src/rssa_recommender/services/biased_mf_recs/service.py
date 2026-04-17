"""Biased MF Recommender Service (PrefViz).

Replicates functionality from rssa_api.services.recommenders.prev_viz_service.
"""

import logging
import os
from itertools import count, islice, product
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.common.mf_base import RSSABase, normalize
from rssa_recommender.common.schemas import MovieLensRating

setup_logging()
log = logging.getLogger(__name__)

S3_BUCKET = os.environ.get('S3_BUCKET')


class BiasedMFRecsService(RSSABase):
    """Service to provide biased MF recommendations."""

    def predict_top_n(self, user_id: str, ratings: list[MovieLensRating], limit: int = 10) -> list[int]:
        """Predict top N items for the user."""
        return self.predict(user_id, ratings, limit=limit)['item'].astype(int).to_list()

    def get_candidates(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        ave_score_type: Literal['global', 'nn_observed', 'nn_predicted'],
        min_rating_count: int = 50,
    ) -> pd.DataFrame:
        """Get candidate items with baseline scores for preference visualization.

        Args:
            user_id: User identifier.
            ratings: User ratings.
            ave_score_type: Type of average score to compute ('global', 'nn_observed', 'nn_predicted').
            min_rating_count: Minimum rating count for items. Defaults to 50.

        Returns:
            DataFrame containing candidate items with baseline scores.
        """
        preds_df = self.predict(user_id, ratings)
        candidates = pd.merge(preds_df, self.item_popularity, how='left', on='item')
        baseline_df = pd.DataFrame()

        if ave_score_type == 'global':
            # Global Observed Average is pre-calculated in self.ave_item_score.
            baseline_df = self.ave_item_score.copy()

        if ave_score_type == 'nn_observed':
            # Average Neighborhood Observed Ratings
            user_features = self.get_user_feature_vector(ratings)
            if user_features is None:
                raise RuntimeError('User featuers not trained in the model.')

            search_space_k = 200
            all_neighbors_ids = self._find_nearest_neighbors_annoy(
                new_user_vector=user_features,
                num_neighbors=search_space_k,
            )
            target_item_ids = set(preds_df['item'].to_list())
            observed_ratings_list = []
            for item_id in target_item_ids:
                neighborhood_avg = self._calculate_neighborhood_average(all_neighbors_ids, item_id, min_rating_count)
                observed_ratings_list.append({'item': item_id, 'ave_score': neighborhood_avg})
            average_neighbor_ratings = pd.DataFrame(observed_ratings_list)
            average_neighbor_ratings = average_neighbor_ratings.rename(columns={'rating': 'ave_score'})
            baseline_df = average_neighbor_ratings

        if ave_score_type == 'nn_predicted':
            # Average Neighborhood Predicted Ratings
            user_features = self.get_user_feature_vector(ratings)
            if user_features is None:
                raise RuntimeError('User featuers not trained in the model.')

            search_space_k = 500
            annoy_index, _ = self._load_annoy_assets()
            nn_ids: list[int] = annoy_index.get_nns_by_vector(user_features, search_space_k, include_distances=False)
            del annoy_index
            del _
            neighbor_internal_codes_np: np.ndarray = np.array(nn_ids, dtype=np.int32)
            target_item_ids = set(preds_df['item'].to_list())
            p_nn_ave_df = self.calculate_predicted_neighborhood_average(
                neighbor_internal_codes_np, preds_df['item'].to_list()
            )
            baseline_df = p_nn_ave_df.rename(columns={'p_nn_ave_score': 'ave_score'})
        if not baseline_df.empty:
            candidates = pd.merge(
                candidates,
                baseline_df[['item', 'ave_score']],
                how='left',
                on='item',
                suffixes=('_user', '_baseline'),
            )

        candidates = candidates[candidates['count'] >= min_rating_count].copy()
        candidates.dropna(inplace=True)
        candidates.index = pd.Index(candidates['item'].values)

        return candidates

    def calculate_predicted_neighborhood_average(
        self, neighbor_internal_codes: np.ndarray, target_item_ids: list[int]
    ) -> pd.DataFrame:
        """Calculate average predicted scores from neighboring users.

        Args:
            neighbor_internal_codes: Internal codes of neighboring users.
            target_item_ids: List of target item IDs.

        Returns:
            DataFrame containing average predicted scores for target items.
        """
        user_features = self.model.user_features_
        if user_features is None:
            return pd.DataFrame()

        Q_target_slice, valid_item_ids = self._get_target_item_factors(target_item_ids)
        P_neighbors_slice = user_features[neighbor_internal_codes, :]

        Pred_Matrix = P_neighbors_slice @ Q_target_slice.T

        # Add Biases (Global + User + Item)
        # Note: We use the term 'bias' as per LensKit terminology (intercepts)
        if hasattr(self.model, 'global_bias_'):
            Pred_Matrix += self.model.global_bias_  # type: ignore

            # Map internal codes -> external user IDs to look up biases
            if hasattr(self.model, 'user_index_') and hasattr(self.model, 'user_biases_'):
                neighbor_external_ids = self.model.user_index_[neighbor_internal_codes]
                neighbor_biases = self.model.user_biases_.reindex(neighbor_external_ids).values  # type: ignore

                # Broadcast: (n_neighbors, 1) to add to each row
                Pred_Matrix += neighbor_biases[:, np.newaxis]

            if hasattr(self.model, 'item_biases_'):
                target_item_biases = self.model.item_biases_.reindex(valid_item_ids).values  # type: ignore
                # Broadcast: (1, n_targets) to add to each column
                Pred_Matrix += target_item_biases[np.newaxis, :]

        P_NN_Ave_Vector = np.mean(Pred_Matrix, axis=0)
        p_nn_ave_df = pd.DataFrame({'p_nn_ave_score': P_NN_Ave_Vector}, index=valid_item_ids)
        p_nn_ave_df.index.name = 'item'
        return p_nn_ave_df.reset_index()

    def predict_with_community_scores(
        self,
        user_id: str,
        ratings: list[MovieLensRating],
        limit: int,
        **kwargs,
    ) -> list[dict]:
        """Generate community scored preference items."""
        log.info(f'Reference N limit = {limit}')
        ratedset = tuple([r.item_id for r in ratings])
        seed = hash(ratedset) % (2**32)
        np.random.seed(seed)
        ave_score_type = kwargs.pop('ave_score_type', 'nn_predicted')
        candidates = self.get_candidates(user_id, ratings, ave_score_type=ave_score_type)
        diverse_items: pd.DataFrame = candidates.copy()

        method = kwargs.pop('method', 'fishnet + single_linkage')
        diverse_items = self._compute_community_score(diverse_items, method, **kwargs)
        n_default = min(limit, len(diverse_items))
        diverse_items = diverse_items.head(n_default)
        scaled_items = self.scale_and_label(diverse_items)

        return scaled_items.to_dict(orient='records')

    def _compute_community_score(self, candidates: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
        """Compute community score for each item.

        Args:
            candidates: DataFrame containing items.
            method: diversification method ('fishnet', 'fishnet + single_linkage', 'single_linkage', 'convex_hull')
            **kwargs: other names parameters

        Returns:
            DataFrame with community scores.
        """
        sampling_size = kwargs.get('sampling_size', 500)

        if method == 'fishnet':
            diverse_items = self._fishingnet(candidates)

        elif method == 'single_linkage':
            # Stratified sampling before clustering to ensure coverage of score range
            candidates.sort_values(by='score', ascending=False, inplace=True)
            candlen = len(candidates)

            # Ensure sampling doesn't exceed available data
            sample_size = min(sampling_size, int(candlen / 3))

            # Calculate indices for Top, Middle, Bottom samples
            mid_start = int(candlen / 2) - int(sample_size / 2)

            topn_user = candidates.head(sample_size).copy()
            botn_user = candidates.tail(sample_size).copy()
            midn_user = candidates.iloc[mid_start : mid_start + sample_size].copy()

            sampled_candidates = pd.concat([topn_user, botn_user, midn_user]).drop_duplicates()

            # Cluster the sampled data into num_rec clusters
            diverse_items = self._single_linkage_clustering(sampled_candidates)

        elif method == 'random':
            # Simple random sampling (always reproducible due to np.random.seed(seed))
            n_sample = min(sampling_size, len(candidates))
            diverse_items = candidates.sample(n=n_sample)

        elif method == 'fishnet + single_linkage':
            # Two-stage process: Filter by grid coverage, then cluster the results
            initial_candidates = self._fishingnet(candidates)
            diverse_items = self._single_linkage_clustering(initial_candidates, **kwargs)
        elif method == 'convexhull':
            diverse_items = self._convexhull(candidates)
        else:
            # Default fallback (just top predicted items)
            diverse_items = candidates.sort_values(by='ave_score', ascending=False)

        return diverse_items

    def create_square_grid(self, n: int, interval_count: int) -> list[float]:
        """Create a square grid of values.

        Args:
            n: The maximum value.
            interval_count: Number of intervals.

        Returns:
            List of grid values.
        """
        return list(islice(count(n, (n - 1) / interval_count), interval_count + 1))

    def scale_grid(self, minval, maxval, num_divisions):
        """Scale grid points between minval and maxval.

        Args:
            minval: Minimum value.
            maxval: Maximum value.
            num_divisions: Number of divisions.

        Returns:
            Numpy array of scaled grid points.
        """
        ticks = [minval]
        step = (maxval - minval) / num_divisions
        for i in range(num_divisions):
            ticks.append(ticks[i] + step)

        grid = list(product(ticks, ticks))
        grid = np.asarray(grid, dtype=np.float64)

        return grid

    def _convexhull(self, candidates: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute the outer boundary (Convex Hull) in the 2D space.

        Identifies the set of items that define the outer boundary (Convex Hull)
        in the 2D score space (user_score vs. ave_score).

        These items represent the extremes of the model's predictions for the user.

        Args:
            candidates: DataFrame containing item candidates, must
                                    include 'user_score' and 'ave_score'.
            kwargs: additional named parameters

        Returns:
            pd.DataFrame: A DataFrame containing only the items that lie on the
                        convex hull boundary.
        """
        if candidates.shape[0] < 3:
            # A Convex Hull requires at least 3 points
            return candidates

        # Use the 2D score space: (User Score, Community Score)
        X = candidates[['score', 'ave_score']].values

        # This finds the indices of the points (rows in X) that form the convex hull.
        try:
            hull = ConvexHull(X)
        except Exception as e:
            # Handle the edge case where points are perfectly collinear (rare, but possible)
            log.warning(f'ConvexHull computation failed (collinearity): {e}')
            return candidates.head()

        hull_indices = hull.vertices
        extreme_items = candidates.iloc[hull_indices].copy()

        log.info(f'Convex Hull analysis found {len(extreme_items)} extreme items.')

        return extreme_items

    def _fishingnet(self, candidates: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
        """Sample a maximally diverse items.

        Performs grid-based sampling to select N items that are maximally diverse
        in the 2D score space, ensuring the selected items cover the grid widely.

        Args:
            candidates (pd.DataFrame): Input candidates, indexed by item ID.
            limit (int): The target number of items to select (should match the number of grid points used).

        Returns:
            pd.DataFrame: A DataFrame containing the N selected diverse items.
        """
        if candidates.empty:
            return candidates

        X = candidates[['score', 'ave_score']].values
        # Assume scale_grid returns a list of N coordinates: [(x1, y1), (x2, y2), ...]
        grid_points = self.scale_grid(minval=1, maxval=5, num_divisions=int(np.sqrt(limit)))
        is_selected = np.zeros(len(candidates), dtype=bool)

        selected_items_list = []
        for point in grid_points:
            dist_to_point = np.sum(np.abs(X - point), axis=1)
            current_distances = dist_to_point.copy()
            current_distances[is_selected] = np.inf  # Ignore already selected items

            if np.all(is_selected):
                break

            idx_closest_candidate = np.argmin(current_distances)

            selected_items_list.append(candidates.iloc[idx_closest_candidate])
            is_selected[idx_closest_candidate] = True
            if len(selected_items_list) >= limit:
                break

        return pd.DataFrame(selected_items_list)

    def _single_linkage_clustering(self, candidates: pd.DataFrame, num_clusters: int = 80, **kwargs) -> pd.DataFrame:
        """Cluster the items using the minimum spanning tree cut.

        Performs Single Linkage Clustering (equivalent to your MST cut) on the
        2D score space (user_score, ave_score) and assigns cluster IDs.

        Args:
            candidates: DataFrame containing item candidates.
            num_clusters: The target number of clusters (N) to form.
            kwargs: additional named parameters.

        Returns:
            pd.DataFrame: Candidates DataFrame with the 'cluster' ID assigned.
        """
        if candidates.shape[0] == 0:
            return candidates

        X = candidates[['score', 'ave_score']].values

        linkage_matrix = linkage(X, method='single', metric='cityblock')

        # To form N clusters, we cut the dendrogram at a distance such that N clusters remain.
        # fcluster cuts the tree and assigns cluster labels (1 to N).
        # Note: The cluster numbering depends on the structure of the tree.
        cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

        candidates['cluster'] = cluster_labels

        final_items = []

        for i in range(1, num_clusters + 1):
            cluster_df = candidates[candidates['cluster'] == i].copy()

            if not cluster_df.empty:
                mid_score = cluster_df['score'].mean()
                mid_ave = cluster_df['ave_score'].mean()
                cluster_df['dist_to_center'] = np.abs(cluster_df['score'] - mid_score) + np.abs(
                    cluster_df['ave_score'] - mid_ave
                )
                representative_item = cluster_df.sort_values(by='dist_to_center', ascending=True).iloc[0]
                final_items.append(representative_item)

        return pd.DataFrame(final_items)

    def scale_and_label(self, items: pd.DataFrame, new_min: int = 1, new_max: int = 5) -> pd.DataFrame:
        """Scale and label the items DataFrame.

        Args:
            items: _description_
            new_min: _description_. Defaults to 1.
            new_max: _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        scaled_items = items.copy()
        if not scaled_items.empty:
            score_min = scaled_items['score'].min()
            score_max = scaled_items['score'].max()

            if score_max > score_min:
                scaled_items['score'] = scaled_items['score'].apply(
                    lambda x: normalize(x, new_min, new_max, score_min, score_max)
                )
            else:
                scaled_items['score'] = (new_min + new_max) / 2

            ave_min = scaled_items['ave_score'].min()
            ave_max = scaled_items['ave_score'].max()

            if ave_max > ave_min:
                scaled_items['ave_score'] = scaled_items['ave_score'].apply(
                    lambda x: normalize(x, new_min, new_max, ave_min, ave_max)
                )
            else:
                scaled_items['ave_score'] = (new_min + new_max) / 2

        scaled_items.rename(columns={'ave_score': 'community_score'}, inplace=True)

        global_avg = np.mean([np.median(scaled_items['community_score']), np.median(scaled_items['score'])])

        def label(row_score):
            return 1 if row_score >= global_avg else 0

        scaled_items['community_label'] = np.where(scaled_items['community_score'] > global_avg, 1, 0)
        scaled_items['label'] = np.where(scaled_items['score'] > global_avg, 1, 0)
        scaled_items['cluster'] = scaled_items['cluster'].astype('int32') if 'cluster' in scaled_items else 0

        return scaled_items
