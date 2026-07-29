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

# Fallback chain: if a tier fails, try the next one down
FALLBACK_CHAIN = {
    "complex": [call_frontier, call_mid, call_small],
    "medium": [call_mid, call_small],
    "simple": [call_small],
}

FALLBACK_TIER_NAMES = {
    call_small: "small",
    call_mid: "mid",
    call_frontier: "frontier",
}

def log_to_redis(entry: dict):
    try:
        redis_client.lpush("routellm:logs", json.dumps(entry))
    except Exception as e:
        print(f"Redis logging failed: {e}")

def route_query(query: str) -> dict:
    """
    Classifies the query, routes it to the appropriate tier with automatic
    fallback if the primary tier fails, logs the result, and returns the
    response with metadata.
    """
    start_time = time.time()

    complexity = classify_query(query)
    chain = FALLBACK_CHAIN[complexity]

    result = None
    fallback_occurred = False
    attempted_tiers = []

    for i, call_fn in enumerate(chain):
        tier_name = FALLBACK_TIER_NAMES[call_fn]
        attempted_tiers.append(tier_name)
        try:
            result = call_fn(query)
            if i > 0:
                fallback_occurred = True
            break
        except Exception as e:
            print(f"Tier '{tier_name}' failed: {e}")
            continue

    if result is None:
        # Every tier in the chain failed
        latency = round(time.time() - start_time, 3)
        error_result = {
            "text": "",
            "tier": "none",
            "model": "none",
            "provider": "none",
            "complexity_label": complexity,
            "latency_seconds": latency,
            "cost_weight": 0,
            "timestamp": time.time(),
            "query": query,
            "error": "all_tiers_failed",
            "attempted_tiers": attempted_tiers,
        }
        log_to_redis(error_result)
        return error_result

    chosen_tier = result["tier"]
    latency = round(time.time() - start_time, 3)

    result["complexity_label"] = complexity
    result["latency_seconds"] = latency
    result["cost_weight"] = COST_WEIGHTS[chosen_tier]
    result["timestamp"] = time.time()
    result["query"] = query
    result["fallback_occurred"] = fallback_occurred
    result["attempted_tiers"] = attempted_tiers

    log_to_redis(result)

    return result
