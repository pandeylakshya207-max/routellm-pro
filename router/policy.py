import time
import json
import os
from dotenv import load_dotenv
from upstash_redis import Redis
from router.classifier import classify_query
from router.providers import call_small, call_mid, call_frontier

load_dotenv()

redis_client = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
)

# Rough relative cost weights (arbitrary units, small=cheapest)
COST_WEIGHTS = {
    "small": 1,
    "mid": 3,
    "frontier": 10,
}

TIER_TO_FUNCTION = {
    "simple": call_small,
    "medium": call_mid,
    "complex": call_frontier,
}

def log_to_redis(entry: dict):
    """Push a routing log entry into a Redis list, non-blocking on failure."""
    try:
        redis_client.lpush("routellm:logs", json.dumps(entry))
    except Exception as e:
        print(f"Redis logging failed: {e}")

def route_query(query: str) -> dict:
    """
    Classifies the query, routes it to the appropriate tier,
    logs the result to Redis, and returns the response with metadata.
    """
    start_time = time.time()

    complexity = classify_query(query)

    tier_map = {
        "simple": "small",
        "medium": "mid",
        "complex": "frontier",
    }
    chosen_tier = tier_map[complexity]

    call_fn = TIER_TO_FUNCTION[complexity]
    result = call_fn(query)

    latency = round(time.time() - start_time, 3)

    result["complexity_label"] = complexity
    result["latency_seconds"] = latency
    result["cost_weight"] = COST_WEIGHTS[chosen_tier]
    result["timestamp"] = time.time()
    result["query"] = query

    log_to_redis(result)

    return result
