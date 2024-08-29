import json
import os
from openai import OpenAI
from together import Together


from config import OPENAI_API_KEY, TOGETHER_API_KEY

CACHE_FILE = "openai_cache.json"

tclient = Together(api_key="cb24d34b78f7d504885d8c16bdf3425a32a5c640e0a451c918d2acf1d3ec617c")
client = OpenAI(api_key=OPENAI_API_KEY)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

def fetch_openai_response(messages, model="gpt-4o", use_cache=True):
    if use_cache:
        key = json.dumps(messages, sort_keys=True)

        if key in cache:
            return cache[key]
    
    if model in ["gpt-4o", "gpt-3.5-turbo-0125"]:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    else:
        response = tclient.chat.completions.create(
            model=model,
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
