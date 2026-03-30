"""Lambad handler for the BiasedMF recommendations."""

import logging
import os

from rssa_recommender.common.logging_config import setup_logging
from rssa_recommender.core.handler import BaseLambdaHandler
from rssa_recommender.core.standard_adapters import (
    handle_diverse_community_score,
    handle_reference_community_score,
    handle_top_n,
)
from rssa_recommender.services.biased_mf_recs.service import BiasedMFRecsService

setup_logging()
log = logging.getLogger(__name__)

log.info('Cold start... initializing BiasedMFRecsService.')

ASSET_ROOT = os.environ.get('MODEL_FOLDER_PATH', 'ml32m')
MODEL_ASSET_BUNDLE_KEY = os.environ.get('BIASED_RS_ASSET_BUNDLE_KEY', 'biased_als_ml32m_bundle.zip')

recs_service = BiasedMFRecsService(asset_root=ASSET_ROOT, asset_bundle_key=MODEL_ASSET_BUNDLE_KEY)

log.info('Service initialized.')

routes = {
    'diverse_community_score': handle_diverse_community_score,
    'reference_community_score': handle_reference_community_score,
    'top_n': handle_top_n,
}

handler = BaseLambdaHandler(recs_service, routes)
