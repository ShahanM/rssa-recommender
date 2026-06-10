"""Local development server simulating AWS API Gateway routing to multiple Lambdas."""

import json
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rssa_recommender.services.biased_mf_recs.main import handler as biased_handler
from rssa_recommender.services.implicit_mf_ers_recs.main import handler as emotion_handler
from rssa_recommender.services.implicit_mf_recs.main import handler as implicit_handler

app = FastAPI(title='RSSA Recommender Multi-Lambda Simulator')


async def invoke_lambda_handler(request: Request, target_handler):
    """Generic wrapper to convert FastAPI requests to AWS Lambda events."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    # The BaseLambdaHandler routes based on `event.get('rawPath') or event.get('path')`.
    # Since LocalDevStrategy passes the routing key (e.g., 'top_n') inside the
    # JSON payload as 'path', we hoist it to the top level of the mock event.
    mock_event = {
        'body': json.dumps(body),
        'path': body.get('path', request.url.path),
        'requestContext': {'http': {'method': request.method, 'path': request.url.path}},
    }
    mock_context = {}

    try:
        response = target_handler(mock_event, mock_context)

        if isinstance(response, dict) and 'body' in response:
            if isinstance(response['body'], str):
                response['body'] = json.loads(response['body'])
        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/invoke/biased_mf')
async def invoke_biased(request: Request):
    return await invoke_lambda_handler(request, biased_handler)


@app.post('/invoke/implicit_mf')
async def invoke_implicit(request: Request):
    return await invoke_lambda_handler(request, implicit_handler)


@app.post('/invoke/emotion_mf')
async def invoke_emotion(request: Request):
    return await invoke_lambda_handler(request, emotion_handler)
