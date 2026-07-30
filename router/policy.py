import time
import json
import os
import random
from dotenv import load_dotenv
from upstash_redis import Redis
from router import classifier
from router import providers

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

FALLBACK_CHAIN = {
    "complex": ["frontier", "mid", "small"],
    "medium": ["mid", "small"],
    "simple": ["small"],
}

TIER_TO_CALL_FN_NAME = {
    "small": "call_small",
    "mid": "call_mid",
    "frontier": "call_frontier",
}

AB_VARIANT_FN_NAMES = {
    "A": "classify_query_a",
    "B": "classify_query_b",
}

def log_to_redis(entry: dict):
    try:
        redis_client.lpush("routellm:logs", json.dumps(entry))
    except Exception as e:
        print(f"Redis logging failed: {e}")

def route_query(query: str, ab_variant: str = None) -> dict:
    """
    Classifies the query (via a randomly assigned A/B variant unless one
    is explicitly passed), routes it with automatic fallback, logs the
    result, and returns the response with metadata.
    """
    start_time = time.time()

    if ab_variant is None:
        ab_variant = random.choice(["A", "B"])

    classify_fn = getattr(classifier, AB_VARIANT_FN_NAMES[ab_variant])
    complexity = classify_fn(query)
    chain = FALLBACK_CHAIN[complexity]

    result = None
    fallback_occurred = False
    attempted_tiers = []

    for i, tier_name in enumerate(chain):
        attempted_tiers.append(tier_name)
        call_fn = getattr(providers, TIER_TO_CALL_FN_NAME[tier_name])
        try:
            result = call_fn(query)
            if i > 0:
                fallback_occurred = True
            break
        except Exception as e:
            print(f"Tier '{tier_name}' failed: {e}")
            continue

    if result is None:
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
            "ab_variant": ab_variant,
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
    result["ab_variant"] = ab_variant

    log_to_redis(result)

    return result
