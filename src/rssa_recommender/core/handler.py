"""Generic Lambda Handler for Recommender Services."""

import json
import logging
from typing import Any, Callable, Union

from rssa_recommender.common.schemas import MovieLensRating
from rssa_recommender.core.interfaces import RecommenderServiceProtocol

log = logging.getLogger(__name__)

RouteHandler = Callable[[RecommenderServiceProtocol, dict[str, Any]], dict[str, Any]]


class BaseLambdaHandler:
    """Generic harness for Recommender Lambdas.

    Handles event parsing, error catching, and routing.
    """

    def __init__(self, service: Any, routes: dict[str, RouteHandler]):
        """Initializes the handler with a service and routing map.

        Args:
            service: The recommender service instance.
            routes: A mapping of path substrings to handler functions.
        """
        self.service = service
        self.routes = routes

    def _get_payload(self, event: dict) -> dict:
        """Safe payload extraction."""
        if 'body' in event:
            try:
                if isinstance(event['body'], str):
                    return json.loads(event['body'])
                return event['body']
            except Exception as e:
                log.error(f'Failed to parse body: {e}')
                return {}
        return event

    def _serialize_response(self, data: Any) -> Union[list, dict, Any]:
        """Helper to handle Pydantic/dict serialization."""
        if isinstance(data, dict):
            return {k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in data.items()}
        if isinstance(data, list):
            return [(i.model_dump() if hasattr(i, 'model_dump') else i) for i in data]
        return data

    def __call__(self, event, context):
        """The main entry point called by AWS."""
        try:
            path = event.get('rawPath') or event.get('path') or ''
            payload = self._get_payload(event)
            log.info(f'Received request for path: {path}')

            user_id = payload.get('user_id')

            if not user_id:
                return {'statusCode': 400, 'body': json.dumps({'error': 'Missing user_id'})}

            ctx = {
                'user_id': str(user_id),
                'ratings': [MovieLensRating(**r) for r in payload.get('ratings', [])],
                'limit': int(payload.get('limit') or 10),
                'raw_payload': payload,
            }

            for route_key, handler_func in self.routes.items():
                if route_key in path:
                    result = handler_func(self.service, ctx)
                    return {
                        'statusCode': 200,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps(self._serialize_response(result)),
                    }

            return {'statusCode': 404, 'body': json.dumps({'error': f"Path '{path}' not found."})}

        except Exception as e:
            log.error(f'Handler Error: {e}', exc_info=True)
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
