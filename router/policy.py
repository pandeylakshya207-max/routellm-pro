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

# Baseline expected latency per tier (seconds), used to detect degradation
BASELINE_LATENCY = {
    "small": 1.0,
    "mid": 2.0,
    "frontier": 5.0,
}

LATENCY_DEGRADED_MULTIPLIER = 3.0  # if rolling avg > baseline * this, treat as degraded
LATENCY_WINDOW_SIZE = 10  # how many recent latencies to track per tier

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

def record_latency(tier: str, latency: float):
    """Record a tier's latency into a rolling window in Redis."""
    try:
        key = f"routellm:latency:{tier}"
        redis_client.lpush(key, str(latency))
        redis_client.ltrim(key, 0, LATENCY_WINDOW_SIZE - 1)
    except Exception as e:
        print(f"Latency tracking failed for {tier}: {e}")

def get_rolling_avg_latency(tier: str) -> float:
    """Get the current rolling average latency for a tier, or its baseline if no data yet."""
    try:
        key = f"routellm:latency:{tier}"
        values = redis_client.lrange(key, 0, -1)
        if not values:
            return BASELINE_LATENCY[tier]
        latencies = [float(v) for v in values]
        return sum(latencies) / len(latencies)
    except Exception:
        return BASELINE_LATENCY[tier]

def is_tier_degraded(tier: str) -> bool:
    """Check if a tier's rolling average latency indicates degraded performance."""
    avg = get_rolling_avg_latency(tier)
    threshold = BASELINE_LATENCY[tier] * LATENCY_DEGRADED_MULTIPLIER
    return avg > threshold

def log_to_redis(entry: dict):
    try:
        redis_client.lpush("routellm:logs", json.dumps(entry))
    except Exception as e:
        print(f"Redis logging failed: {e}")

def route_query(query: str, ab_variant: str = None) -> dict:
    """
    Classifies the query, routes it with automatic fallback on hard failure
    AND proactive skipping of degraded (unusually slow) tiers, logs the
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
    latency_skip_occurred = False
    attempted_tiers = []
    skipped_degraded_tiers = []

    for i, tier_name in enumerate(chain):
        if is_tier_degraded(tier_name) and i < len(chain) - 1:
            # Skip this tier proactively, unless it's the last option in the chain
            skipped_degraded_tiers.append(tier_name)
            latency_skip_occurred = True
            print(f"Tier '{tier_name}' is degraded (slow), skipping proactively")
            continue

        attempted_tiers.append(tier_name)
        call_fn = getattr(providers, TIER_TO_CALL_FN_NAME[tier_name])
        tier_start = time.time()
        try:
            result = call_fn(query)
            tier_latency = time.time() - tier_start
            record_latency(tier_name, tier_latency)
            if i > 0 or latency_skip_occurred:
                fallback_occurred = True
            break
        except Exception as e:
            tier_latency = time.time() - tier_start
            record_latency(tier_name, tier_latency)
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
            "skipped_degraded_tiers": skipped_degraded_tiers,
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
    result["skipped_degraded_tiers"] = skipped_degraded_tiers

    log_to_redis(result)

    return result
