"""Lambda handler for the RSSA Emotion recommender model."""

import os

import structlog

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.core.handler import BaseLambdaHandler
from rssa_recommender.core.interfaces import RecommenderServiceProtocol
from rssa_recommender.services.implicit_mf_ers_recs.service import ImplicitMFErsRecsService

setup_logging()
log = structlog.getLogger(__name__)

log.info('Initializing ImplicitMFErsRecsService.')

ASSET_ROOT = os.environ.get('MODEL_FOLDER_PATH', 'ml32m')
MODEL_ASSET_BUNDLE_KEY = os.environ.get('ERS_RS_ASSET_BUNDLE_KEY', 'implicit_als_ers_ml32m_bundle.zip')

recs_service = ImplicitMFErsRecsService(asset_root=ASSET_ROOT, asset_bundle_key=MODEL_ASSET_BUNDLE_KEY)

log.info('Implicit MF ERS Service initialized.')


def route_recommendations(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Endpoint for emotions recommendations."""
    raw = ctx['raw_payload']
    strategy = raw.get('strategy', 'diverse_n')
    emotion_input = raw.get('emotion_input', [])
    ranking_strategy = raw.get('ranking_strategy', 'distance')

    results = service.predict_with_emotions(
        user_id=ctx['user_id'],
        ratings=ctx['ratings'],
        limit=ctx['limit'],
        strategy=strategy,
        emotion_input=emotion_input,
        ranking_strategy=ranking_strategy,
    )

    return {'response_type': 'standard', 'items': results}


routes = {'get_recommentations': route_recommendations}

handler = BaseLambdaHandler(recs_service, routes)
