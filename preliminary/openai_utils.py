import json
import os
from openai import OpenAI

from config import OPENAI_API_KEY

CACHE_FILE = "openai_cache.json"

client = OpenAI(api_key=OPENAI_API_KEY)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

def fetch_openai_response(messages, use_cache=True):
    if use_cache:
        key = json.dumps(messages, sort_keys=True)

        if key in cache:
            return cache[key]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    if use_cache:
        response = serialize_completion(response)
        cache[key] = response

        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    
    return response

def serialize_completion(completion):
    return {
        "id": completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                    "function_call": {
                        "arguments": json.loads(
                            choice.message.function_call.arguments) if choice.message.function_call and choice.message.function_call.arguments else None,
                        "name": choice.message.function_call.name
                    } if choice.message and choice.message.function_call else None
                } if choice.message else None
            } for choice in completion.choices
        ],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "system_fingerprint": completion.system_fingerprint,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens
        }
    }
