"""The lambda routing to the implicit ALS recommendation model for the RSSA algorithms."""

import logging
import os

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.core.handler import BaseLambdaHandler
from rssa_recommender.core.interfaces import RecommenderServiceProtocol
from rssa_recommender.services.implicit_mf_recs.service import ImplicitMFRecsService

setup_logging()
log = logging.getLogger(__name__)

log.info('Cold start... initializing ImplicitMFRecsService.')

ASSET_ROOT = os.environ.get('MODEL_FOLDER_PATH', 'ml32m')
RESAMPLED_ASSET_BUNDLE_KEY = os.environ.get('ALT_RS_RESAMPLED_ASSET_KEY', 'implicit_als_ml32m_resampled_bundle.zip')
MODEL_ASSET_BUNDLE_KEY = os.environ.get('ALT_RS_ASSET_BUNDLE_KEY', 'implicit_als_ml32m_bundle.zip')

recs_service = ImplicitMFRecsService(
    asset_root=ASSET_ROOT,
    asset_bundle_key=MODEL_ASSET_BUNDLE_KEY,
    resampled_asset_bundle_key=RESAMPLED_ASSET_BUNDLE_KEY,
)

log.info('Implicit MF Service initialized.')


def route_top_n(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Return the top n recommendation using the ALS model."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_discounted_top_n(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Returns the top n recommendation using the ALS model but discounted for popularity bias."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_controversial(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Returns n items that the system thinks are controversial to the user's preferences."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_hate(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Returns n items that the system thinks are contradictory to the user's preferences."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_hip(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Returns n items that the system think the user will be among the first to try."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_no_clue(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Return n items that the system is unsure regarding the user's preferences."""
    return {'response_type': 'Not Implemented', 'items': []}


def route_community_advisors(service: RecommenderServiceProtocol, ctx: dict) -> dict:
    """Returns n items masquerading as advisors packaged for the preference community study."""
    raw = ctx['raw_payload']
    results = service.predict_advisors_with_profile(
        user_id=ctx['user_id'],
        ratings=ctx['ratings'],
        limit=ctx['limit'],
        strategy=raw.get('strategy', 'top_n'),
    )
    return {'response_type': 'community_advisors', 'items': results}


routes = {
    'top_n': route_top_n,
    'discounted_top_n': route_discounted_top_n,
    'controversial': route_controversial,
    'hate': route_hate,
    'hip': route_hip,
    'no_clue': route_no_clue,
    # Community Advisors types:1
    # 1. Low Diversity - Top N
    # 2. High Diversity no compromise
    # 3. High Diversity compromise
    'community_advisors': route_community_advisors,
}

handler = BaseLambdaHandler(recs_service, routes)
