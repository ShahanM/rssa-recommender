"""Lambad handler for the BiasedMF recommendations."""

import logging
import os

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.core.handler import BaseLambdaHandler
from rssa_recommender.core.interfaces import RecommenderServiceProtocol
from rssa_recommender.services.biased_mf_recs.service import BiasedMFRecsService

setup_logging()
log = logging.getLogger(__name__)

log.info('Cold start... initializing BiasedMFRecsService.')

ASSET_ROOT = os.environ.get('MODEL_FOLDER_PATH', 'ml32m')
MODEL_ASSET_BUNDLE_KEY = os.environ.get('BIASED_RS_ASSET_BUNDLE_KEY', 'biased_als_ml32m_bundle.zip')

recs_service = BiasedMFRecsService(asset_root=ASSET_ROOT, asset_bundle_key=MODEL_ASSET_BUNDLE_KEY)

log.info('Service initialized.')


def route_community_scores(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    raw = ctx['raw_payload']
    results = service.predict_with_community_scores(
        user_id=ctx['user_id'],
        ratings=ctx['ratings'],
        limit=ctx['limit'],
        ave_score_type=raw.get('ave_score_type', 'nn_predicted'),
        method=raw.get('method', 'fishnet + single_linkage'),
    )
    return {'response_type': 'community_comparison', 'items': results}


def route_top_n(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    results = service.predict_top_n(user_id=ctx['user_id'], ratings=ctx['ratings'], limit=ctx['limit'])

    return {'response_type': 'standard', 'items': results}


routes = {
    'community_scored_predictions': route_community_scores,
    'top_n': route_top_n,
}

handler = BaseLambdaHandler(recs_service, routes)
