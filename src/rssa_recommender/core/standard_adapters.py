"""Adapters to facilitate service calls."""

import inspect
import logging

from rssa_recommender.common.schemas import EmotionContinuousInputSchema, EmotionDiscreteInputSchema

log = logging.getLogger(__name__)


def _parse_emotions(raw_payload):
    e_input = raw_payload.get('emotion_input')
    if not e_input:
        return None
    try:
        if 'switch' in e_input[0]:
            return [EmotionContinuousInputSchema(**e) for e in e_input]
        return [EmotionDiscreteInputSchema(**e) for e in e_input]
    except Exception:
        return e_input


def _get_kwargs(ctx):
    """Extract optional params from raw payload safely."""
    raw = ctx.get('raw_payload', {})
    return {
        'emotion_input': _parse_emotions(raw),
        'candidate_pool_size': raw.pop('candidate_pool_size', 500),
        'sampling_size': raw.pop('sampling_size', 50),
        'ranking_strategy': raw.pop('ranking_strategy', 'weighted'),  # 'weighted' or 'distance'
        'emotion_discrete_cutoffs': raw.pop('emotion_discrete_cutoff', (0.3, 0.8)),
        'min_rating_count': raw.pop('min_rating_count', 50),
    }


def _call_service_method(method, ctx):
    """Dynamically maps context arguments to the method signature."""
    sig = inspect.signature(method)
    kwargs = _get_kwargs(ctx)

    bound_args = {}

    if 'user_id' in sig.parameters:
        bound_args['user_id'] = ctx['user_id']

    if 'ratings' in sig.parameters:
        bound_args['ratings'] = ctx['ratings']

    limit = ctx['limit']
    if 'limit' in sig.parameters:
        bound_args['limit'] = limit

    for k, v in kwargs.items():
        if k in sig.parameters:
            bound_args[k] = v
        elif any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            bound_args[k] = v

    return method(**bound_args)


def handle_top_n(service, ctx):
    raw_results = _call_service_method(service.predict_top_n, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_discounted_top_n(service, ctx):
    raw_results = _call_service_method(service.predict_discounted_top_n, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_diverse_n(service, ctx):
    raw_results = _call_service_method(service.predict_diverse_n, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_diverse_community_score(service, ctx):
    raw_results = _call_service_method(service.predict_diverse_items, ctx)
    return {'response_type': 'community_comparison', 'items': raw_results}


def handle_reference_community_score(service, ctx):
    raw_results = _call_service_method(service.predict_reference_items, ctx)
    return {'response_type': 'community_comparison', 'items': raw_results}


def handle_controversial(service, ctx):
    raw_results = _call_service_method(service.predict_controversial_items, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_hate(service, ctx):
    raw_results = _call_service_method(service.predict_hate_items, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_hip(service, ctx):
    raw_results = _call_service_method(service.predict_hip_items, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_no_clue(service, ctx):
    raw_results = _call_service_method(service.predict_no_clue_items, ctx)
    return {'response_type': 'standard', 'items': raw_results}


def handle_community_advisors(service, ctx):
    raw_result = _call_service_method(service.get_advisors_with_profile, ctx)
    return {'response_type': 'community_advisors', 'items': raw_result}
