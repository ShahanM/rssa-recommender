"""RSSA Matrix Factorization Model Training Script (MF).

This script trains and serializes LensKit Matrix Factorization (ALS) models
for the Recommender System for Study Analysis (RS:SA) project.
It includes data pre-processing (discounting popular items) and generates
auxiliary baselines and artifacts (Annoy index, popularity stats).
----
File: train_mf.py
Project: RS:SA Recommender System (Clemson University)
Created Date: Saturday, 11th October 2025
Author: Mehtab 'Shahan' Iqbal
Affiliation: Clemson University
----
Last Modified: Wednesday, 10th December 2025 1:02:45 am
Modified By: Mehtab 'Shahan' Iqbal (mehtabi@clemson.edu)
----
Copyright (c) 2025 Clemson University
License: MIT License (See LICENSE.md)
# SPDX-License-Identifier: MIT License
"""

import argparse
import logging
import math
import os
import pickle
import time
from typing import Optional, Union, cast

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from joblib import Memory
from lenskit.algorithms import als
from lenskit.algorithms.mf_common import MFPredictor
from pydantic import BaseModel

log = logging.getLogger(__name__)

try:
    import binpickle
except ImportError:
    log.warning('Warning: binpickle not installed. Falling back to pickle.')
    binpickle = None


def setup_logging(output_dir: str) -> None:
    """Configures the logger to write to console and a date-stamped file."""
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = time.strftime('training_%Y%m%d_%H%M%S.log')
    log_path = os.path.join(output_dir, log_filename)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info('Logging configured successfully. Output directed to console and file.')


cachedir = '.cache'
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir, verbose=1)


class Config(BaseModel):
    """Pydantic model for holding the script's run configuration."""

    data_path: str
    model_path: str
    algo: str
    item_popularity: Optional[bool]
    ave_item_score: Optional[bool]
    cluster_index: Optional[bool]
    ratings_index: Optional[bool]
    resample_count: int
    filter_list: Optional[str]
    emotion_index_path: Optional[str]


def _get_model_path(model_path: str) -> str:
    """Gets the full model file path with the correct serialization extension."""
    base_file = os.path.join(model_path, 'model')
    return f'{base_file}.bpk' if binpickle else f'{base_file}.pkl'


@memory.cache
def load_training_data(data_path: str) -> pd.DataFrame:
    """Loads and standardizes training data from a CSV file.

    Renames common column variations (e.g., 'userId', 'movieId')
    to the script's standard ('user', 'item').
    """
    ratings_train = pd.read_csv(
        data_path, dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'timestamp': 'int32'}
    )
    col_rename_dict = {}
    if 'user_id' in ratings_train:
        col_rename_dict['user_id'] = 'user'
    elif 'userId' in ratings_train:
        col_rename_dict['userId'] = 'user'
    if 'movie_id' in ratings_train:
        col_rename_dict['movie_id'] = 'item'
    elif 'movieId' in ratings_train:
        col_rename_dict['movieId'] = 'item'

    ratings_train = ratings_train.rename(columns=col_rename_dict)

    return ratings_train


def load_training_data_npz(data_path: str) -> pd.DataFrame:
    """Loads training data from a pre-saved .npz file.

    Assumes the .npz file contains a 'dataset' array with columns
    ['user', 'item', 'rating', 'timestamp'].
    """
    model_loaded = np.load(data_path)
    data = model_loaded['dataset']
    trainset = pd.DataFrame({'user': data[:, 0], 'item': data[:, 1], 'rating': data[:, 2], 'timestamp': data[:, 3]})
    trainset = trainset.astype({'user': int, 'item': int, 'rating': float, 'timestamp': int})

    return trainset


def _train_mf_model(training_data: pd.DataFrame, algo: str) -> MFPredictor:
    """Trains a single LensKit matrix factorization model."""
    model = None
    if algo == 'implicit':
        model = als.ImplicitMF(20, iterations=10, method='lu', use_ratings=True, save_user_features=True)
    elif algo == 'biased':
        model = als.BiasedMF(20)

    if model is None:
        raise ValueError(f'Invalid algo: {algo}')
    model.fit(training_data)

    return model


def _train_resampled_models(
    training_data: pd.DataFrame, algo: str, resample_count: int, output_dir: str, alpha: float = 0.5
) -> None:
    """Trains and serializes multiple models on subsets of the data.

    This is used for bootstrapping or robust evaluation.
    """
    sample_size = int(training_data.shape[0] * alpha)
    log.info(f'Training {resample_count} resampled models')
    start = time.time()

    for i in range(resample_count):
        log.info(f'Training resampled model {i + 1}')

        sample = training_data.sample(n=sample_size, replace=False)
        model = _train_mf_model(sample, algo)
        log.info(f'Serializing the trained model {i + 1} of {resample_count} to disk.')
        output_filename = f'{output_dir}/resampled_model_{i + 1}'
        output_file = f'{output_filename}.bpk' if binpickle else f'{output_filename}.pkl'
        if binpickle:
            binpickle.dump(model, output_file)
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(model, f)

    end = time.time() - start
    log.info('Resampled models trained.')
    log.info(f'Time spent: {end:.2f}')


def _pre_aggregate_user_history(training_data: pd.DataFrame, output_path: str) -> None:
    """Aggregates user rating history into a compact lookup table.

    Creates a Parquet file mapping each user ID to a list of their
    (item, rating) history tuples for fast runtime lookup.

    Args:
        training_data: The training data.
        output_path: Path to save the output .parquet file.
    """
    log.info('Building history look up table')

    user_history_df = (
        training_data.groupby('user').agg(rated_items=('item', list), ratings=('rating', list)).reset_index()
    )

    # Vectorized Combination of Lists
    # We use numpy.vstack and then convert the array structure to a list of tuples
    # and assign it back to a column. This is much faster than itertuples.
    user_history_df['history_tuples'] = [
        list(zip(items, ratings, strict=False))
        for items, ratings in zip(user_history_df['rated_items'], user_history_df['ratings'], strict=False)
    ]

    final_history_df = user_history_df[['user', 'history_tuples']]
    final_history_df.to_parquet(output_path, compression='snappy')

    log.info(f'User history lookup table saved to: {output_path} (Compressed)')


def _create_item_emotion_lookup(emotion_data_path: str, output_path: str) -> None:
    """Loads raw item emotion data and saves it as a compact lookup file.

    Renames 'movie_id' to 'item', filters for known emotion tags,
    and saves the result as an item-indexed Parquet file.
    """
    log.info(f'Building item emotion lookup table from: {emotion_data_path}')
    try:
        emotion_data = pd.read_csv(emotion_data_path)

        if 'movie_id' in emotion_data.columns:
            emotion_data = emotion_data.rename({'movie_id': 'item'}, axis=1)
        elif 'item' not in emotion_data.columns:
            log.error(
                f'Emotion file {emotion_data_path} must have "item" or "movie_id" column. Aborting lookup creation.'
            )
            return

        emotion_tags = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

        available_tags = [tag for tag in emotion_tags if tag in emotion_data.columns]
        missing_tags = [tag for tag in emotion_tags if tag not in emotion_data.columns]
        if missing_tags:
            log.warning(f'Emotion data is missing expected columns: {missing_tags}. Proceeding with available columns.')

        if 'item' not in emotion_data.columns:
            log.error('Logic error: "item" column not found after rename. Aborting lookup creation.')
            return

        final_columns = ['item'] + available_tags
        final_lookup = emotion_data[final_columns].copy()
        final_lookup = final_lookup.set_index('item')

        final_lookup.to_parquet(output_path, compression='snappy')
        log.info(f'Item emotion lookup table saved to: {output_path} (Compressed)')

    except FileNotFoundError:
        log.error(f'Emotion data file not found: {emotion_data_path}. Cannot create lookup table.')
    except Exception as e:
        log.error(f'Failed to create item emotion lookup: {e}')


def _get_exact_mf_model(model: MFPredictor) -> Union[als.BiasedMF, als.ImplicitMF, None]:
    """Casts a generic MFPredictor to its specific ALS implementation.

    This is used to gain access to algorithm-specific attributes like
    'user_features_' or 'bias'.
    """
    if isinstance(model, als.BiasedMF):
        model = cast(als.BiasedMF, model)
    elif isinstance(model, als.ImplicitMF):
        model = cast(als.ImplicitMF, model)
    else:
        log.warning('Could not build Annoy index, model type unknown.')
        return None

    return model


def _compute_ave_item_scores(model: MFPredictor, item_popularity: pd.DataFrame, alpha: float = 0.2):
    """Computes model-predicted average scores and discounted scores for all items.

    This function calculates a baseline score for every item by using the
    mean user feature vector and mean bias terms. It then applies a
    popularity-based discount to this score.
    """
    start = time.time()
    log.info('Starting vectorized computation of average item scores...')

    model_instance = _get_exact_mf_model(model)
    if model_instance is None:
        log.error('Failed to get MF model instance for vectorization.')
        return None

    bias = getattr(model_instance, 'bias', None)

    user_features = model_instance.user_features_
    item_features = model_instance.item_features_
    item_index = model_instance.item_index_

    if user_features is None:
        return None

    # Mean user features shape (K, )
    mean_user_features = user_features.mean(axis=0)

    # Core projection means shape(I, )
    # (K, ) @ (K, I) -> (I, )
    core_projection_means = mean_user_features @ item_features.T

    # Average scores vector shape (I, ) indexed by item_index
    ave_scores_vector = pd.Series(core_projection_means, index=item_index)

    if bias is not None:
        ave_scores_vector += bias.mean_

        if bias.item_offsets_ is not None:
            ioff = bias.item_offsets_.reindex(item_index, fill_value=0)
            ave_scores_vector = ave_scores_vector + ioff

        if bias.user_offsets_ is not None:
            mean_user_offset = bias.user_offsets_.mean()
            ave_scores_vector = ave_scores_vector + mean_user_offset

    ave_scores_df = pd.DataFrame({'ave_score': ave_scores_vector})

    if 'item' not in item_popularity.columns:
        item_popularity_df = item_popularity.reset_index()
    else:
        item_popularity_df = item_popularity

    if 'item' not in ave_scores_df.columns:
        ave_scores_df = ave_scores_df.reset_index().rename(columns={'index': 'item'})

    max_count = cast(int, item_popularity_df['count'].max())
    discounting_factor = 10 ** (math.floor(math.log10(max_count)) + 1) if max_count > 0 else 1

    ave_scores_df = pd.merge(ave_scores_df, item_popularity, how='left', on='item')
    ave_scores_df['count'] = ave_scores_df['count'].fillna(0)
    ave_scores_df['ave_discounted_score'] = ave_scores_df['ave_score'] - alpha * (
        ave_scores_df['count'] / discounting_factor
    )

    log.info(f'Time spent (vectorized): {(time.time() - start):.4f}s. Calculated scores for {len(item_index)} items.')

    return ave_scores_df[['item', 'ave_score', 'ave_discounted_score']]


def _compute_observed_item_mean(training_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes observed item rating statistics from the training data.

    Calculates the mean rating, rating count, and popularity/quality
    ranks for every item.
    """
    item_stats = training_data.groupby('item')['rating'].agg(['mean', 'count'])
    item_stats = item_stats.reset_index()

    item_stats['rank_popular'] = item_stats['count'].rank(method='min', ascending=False).astype(int)
    item_stats['rank_quality'] = item_stats['mean'].rank(method='min', ascending=False).astype(int)

    item_popularity = item_stats[['item', 'count', 'rank_popular', 'rank_quality']]
    ave_scores_df = item_stats.rename(columns={'mean': 'ave_score'})[['item', 'ave_score']]
    ave_scores_df['ave_discounted_score'] = np.nan

    assert isinstance(ave_scores_df, pd.DataFrame)
    assert isinstance(item_popularity, pd.DataFrame)
    return ave_scores_df, item_popularity


def _create_annoy_index(user_factors: np.ndarray, user_index: pd.Index, output_path: str, n_trees: int = 50) -> None:
    """Creates and saves an Annoy index for user latent factors.

    This enables fast K-Nearest-Neighbor lookups on the user feature vectors.
    Also saves a CSV mapping internal Annoy IDs (0..N) to external user IDs.
    """
    dims = user_factors.shape[1]
    index = AnnoyIndex(dims, 'angular')

    for i, vector in enumerate(user_factors):
        index.add_item(i, vector)

    index.build(n_trees)
    index.save(output_path)
    user_map = pd.Series(user_index.values, index=np.arange(len(user_index)), name='user_id')

    user_map.to_csv(f'{output_path}_map.csv', header=True, index_label='internal_id')

    log.info(f'Annoy index built and saved to {output_path}')


def _discount_popular_item_ratings(
    input_data: pd.DataFrame, items_popularity: pd.DataFrame, bias_factor: float = 0.4
) -> pd.DataFrame:
    """Discounts ratings for popular items to reduce popularity bias.

    Applies a penalty to ratings based on the item's popularity rank,
    making very popular items' ratings slightly lower.
    """
    rank_map = items_popularity.set_index('item')['rank_popular']
    mapped_ranks = input_data['item'].map(rank_map)  # type: ignore
    rtrain = input_data[['user', 'item', 'timestamp']].copy()
    rtrain['rating'] = input_data['rating'] * (1 - bias_factor / (2 * mapped_ranks))
    rtrain = rtrain.assign(rating=input_data['rating'] * (1 - bias_factor / (2 * mapped_ranks)))

    return rtrain


def _get_model(config: Config, train_data: pd.DataFrame) -> MFPredictor:
    """Retrieves a pre-trained model or trains a new one.

    Loads a model from disk if it exists and resampling is not requested.
    Otherwise, it trains a new model, serializes it, and returns it.
    """
    model_file = _get_model_path(config.model_path)
    model = None

    if os.path.exists(model_file) and not config.resample_count > 0:
        log.info(f'Existing model found: {model_file}. Loading trained model...')
        try:
            if binpickle:
                model = binpickle.load(model_file)
            else:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            log.info('Model loaded successfully.')
        except Exception as e:
            log.warning(f'Error loading model ({e}). Retraining will proceed.')
            model = None

    if model is None:
        log.info('Model not found or load failed. Initiating full training...')
        log.info(f'Training {config.algo} MF models')
        start = time.time()

        model = _train_mf_model(train_data, config.algo)

        end = time.time() - start
        log.info('MF model trained.')
        log.info(f'Time spent: {end:.2f}s')

        log.info('Serializing the trained model to disk.')
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        if binpickle:
            binpickle.dump(model, model_file)
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

    return model


def _apply_item_filter(train_data: pd.DataFrame, filter_list_path: Optional[str]) -> pd.DataFrame:
    """Filters the training data based on an item inclusion list.

    If a filter path is provided, it reads the CSV, extracts the 'item'
    column, and filters the training data to only include interactions
    with items in that list.
    """
    if not filter_list_path:
        return train_data

    log.info(f'Attempting to filter training data with item list: {filter_list_path}')
    try:
        filter_items_df = pd.read_csv(filter_list_path)
        if 'movie_id' in filter_items_df.columns:
            filter_items_df = filter_items_df.rename({'movie_id': 'item'}, axis=1)
        elif 'item' not in filter_items_df.columns:
            log.error(
                f'Emotion file {filter_list_path} must have "item" or "movie_id" column. Aborting lookup creation.'
            )
            raise KeyError(
                f'Emotion file {filter_list_path} must have "item" or "movie_id" column. Aborting lookup creation.'
            )

        filter_item_set: set[int] = set(filter_items_df['item'])
        original_count = len(train_data)
        filtered_data = train_data[train_data['item'].isin(list(filter_item_set))].copy()
        filtered_count = len(filtered_data)
        log.info(f'Filtered training data from {original_count} to {filtered_count} interactions.')

        assert isinstance(filtered_data, pd.DataFrame)
        return filtered_data

    except FileNotFoundError:
        log.warning(f'Filter list file not found: {filter_list_path}. Proceeding with unfiltered data.')
    except Exception as e:
        log.warning(f'Error processing filter list {filter_list_path}: {e}. Proceeding with unfiltered data.')

    return train_data


def _run_post_processing_tasks(
    config: Config,
    model: MFPredictor,
    discounted_train_data: pd.DataFrame,
    obs_ave_scores_df: pd.DataFrame,
    items_popularity_df: pd.DataFrame,
):
    """Orchestrates all optional post-training analysis and artifact generation.

    This function checks the config flags and runs the corresponding
    helper functions to save popularity stats, train resampled models,
    compute average scores, build Annoy indexes, and create lookup tables.
    """
    if config.item_popularity:
        log.info('Saving the item popularity as a csv file')
        items_popularity_df.to_csv(f'{config.model_path}/item_popularity.csv', index=False)

    if config.resample_count > 0:
        _train_resampled_models(discounted_train_data, config.algo, config.resample_count, config.model_path)

    if config.ave_item_score:
        log.info('Computing the average model item scores')
        scores_df = _compute_ave_item_scores(model, items_popularity_df)
        if scores_df is not None:
            log.info('Saving the average model item scores as a csv file')
            scores_df.to_csv(f'{config.model_path}/averaged_item_score.csv', index=False)

        log.info('Saving the average observed item scores as a csv file')
        obs_ave_scores_df.to_csv(f'{config.model_path}/obs_ave_item_score.csv', index=False)

    if config.cluster_index:
        log.info('Building and saving the Annoy index')
        model_instance = _get_exact_mf_model(model)
        if model_instance is not None:
            user_mat = model_instance.user_features_
            user_index = model_instance.user_index_
            if user_mat is not None and user_index is not None:
                _create_annoy_index(user_mat, user_index, f'{config.model_path}/annoy_index')
            else:
                log.warning('Could not build Annoy index: user features or index not found.')
        else:
            log.warning('Could not build Annoy index: model instance not valid.')

    if config.ratings_index:
        _pre_aggregate_user_history(discounted_train_data, f'{config.model_path}/user_history_lookup.parquet')

    if config.emotion_index_path:
        _create_item_emotion_lookup(config.emotion_index_path, f'{config.model_path}/item_emotion_lookup.parquet')


def _main(config: Config):
    """Main execution function for the training script.

    Orchestrates the entire workflow:
    1. Load and filter data.
    2. Get or train the main model.
    3. Compute core statistics from the data.
    4. Run all optional post-processing tasks.
    """
    train_data = load_training_data(config.data_path)
    train_data = _apply_item_filter(train_data, config.filter_list)

    if train_data.empty:
        log.error('Filtering resulted in 0 interactions or data is empty. Aborting.')
        return

    obs_ave_scores_df, items_popularity_df = _compute_observed_item_mean(train_data)
    discounted_train_data = _discount_popular_item_ratings(train_data, items_popularity_df)

    model = _get_model(config, discounted_train_data)
    if model is None:
        log.error('ERROR: Could not load or train model. Skipping post-processing.')
        return

    _run_post_processing_tasks(config, model, discounted_train_data, obs_ave_scores_df, items_popularity_df)

    log.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Train Matrix Factorization (MF) models for the RSSA (Recommender System for Study Analysis) '
            'project. Generates primary and baseline models for evaluation.'
        )
    )
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        required=True,
        help=(
            'Path to the CSV file containing the raw user interaction data. '
            'Expected columns include: [user_id, movie_id, rating, timestamp].'
        ),
    )
    parser.add_argument(
        '-o',
        '--model_path',
        type=str,
        required=True,
        help=('Path to the output directory where the trained model, and analysis files will be saved.'),
    )

    parser.add_argument(
        '-a',
        '--algo',
        type=str,
        required=False,
        default='implicit',
        choices=['implicit', 'biased'],
        help=(
            'Specifies the Matrix Factorization algorithm type to train. '
            'Choose "implicit" for Implicit ALS (optimizing preference/confidence) or '
            '"biased" for Biased MF (optimizing explicit rating prediction).'
        ),
    )

    parser.add_argument(
        '--item_popularity',
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If set, calculates the observed popularity statistics (count and rank) of all items '
            'and saves the results as a CSV file.'
        ),
    )
    parser.add_argument(
        '--ave_item_score',
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If set, computes and saves two item score baselines: '
            '1) the global observed mean rating for each item, and '
            '2) the model-predicted average score for each item across all users.'
        ),
    )
    parser.add_argument(
        '--cluster_index',
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If set, builds and saves an Annoy index over the final user latent factor matrix (P matrix). '
            ' This enables extremely fast K-NN lookups for the warm-start prediction baseline.'
        ),
    )

    parser.add_argument(
        '--ratings_index',
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If set, aggregates the large training data into a compact, serializable DataFrame. '
            ' This enables a fast an efficient way to look up ratings from the training data at runtime.'
        ),
    )

    parser.add_argument(
        '-f',
        '--filter_list',
        type=str,
        required=False,
        default=None,
        help=(
            'Path to a CSV file containing a single "item" column. '
            'If provided, the training data will be filtered to only include '
            'interactions with items present in this list.'
        ),
    )

    parser.add_argument(
        '--emotion_index',
        type=str,
        required=False,
        default=None,
        help=(
            'Path to the item emotion data file (e.g., "iersg20.csv"). '
            'If provided, creates a compact item feature lookup table '
            '(e.g., "item_emotion_lookup.parquet") in the model output directory.'
        ),
    )

    parser.add_argument(
        '-r',
        '--resample_count',
        type=int,
        required=False,
        default=0,
        help=(
            'The number of additional resampled models to train for robust evaluation (e.g., bootstrapping). '
            'If set to N > 0, N models are trained on distinct subsets of the training data '
            'and serialized individually.'
        ),
    )
    args = parser.parse_args()

    setup_logging(args.model_path)

    run_config = Config(
        data_path=args.data_path,
        model_path=args.model_path,
        algo=args.algo,
        item_popularity=args.item_popularity,
        ave_item_score=args.ave_item_score,
        cluster_index=args.cluster_index,
        ratings_index=args.ratings_index,
        resample_count=args.resample_count,
        filter_list=args.filter_list,
        emotion_index_path=args.emotion_index,
    )
    log.info('Starting model training script.')
    _main(run_config)
    log.info('Script execution finished.')

    # Defaults for the current RSSA
    # ieRS
    """
    python scripts/train_mfs.py \
    -d ~/zugzug/data/movies/ml-32m/ratings.csv \
    -o assets/models/implicit_als_ers_ml32m/ \
    -a implicit \
    --item_popularity \
    --ave_item_score \
    --cluster_index \
    --emotion_index ~/zugzug/data/movies/ieRS_emotions_g20.csv \
    --filter_list ~/zugzug/data/movies/ieRS_emotions_g20.csv \
    --resample_count 20
    """

    # alt algo
    """
    python scripts/train_mfs.py \
    -d ~/zugzug/data/movies/ml-32m/ratings.csv \
    -o assets/models/implicit_als_ml32m/ \
    -a implicit \
    --item_popularity \
    --ave_item_score \
    --cluster_index \
    --resample_count 20
    """

    # pref viz
    # --algo 'biased'
    # --item_popularity
    # --ave_item_score
    """
    python scripts/train_mfs.py \
    -d ~/zugzug/data/movies/ml-32m/ratings.csv \
    -o assets/models/biased_als_ml32m/ \
    -a biased \
    --item_popularity \
    --ave_item_score \
    --cluster_index
    """
