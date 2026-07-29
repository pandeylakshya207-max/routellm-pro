import os
import json
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()
r = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))

raw_logs = r.lrange("routellm:logs", 0, -1)
logs = [json.loads(entry) for entry in raw_logs]

frontier_logs = [l for l in logs if l.get("tier") == "frontier"]

print(f"Total logs: {len(logs)}")
print(f"Frontier logs found: {len(frontier_logs)}")

if frontier_logs:
    avg_latency = sum(l["latency_seconds"] for l in frontier_logs) / len(frontier_logs)
    print(f"Avg frontier latency (from real historical calls): {round(avg_latency, 3)}s")
    print(f"Frontier cost weight (fixed): 10")

    total_cost = sum(l.get("cost_weight", 0) for l in logs)
    total_latency = sum(l.get("latency_seconds", 0) for l in logs)
    n = len(logs)
    always_frontier_theoretical_cost = n * 10

    print(f"\nAcross all {n} logged queries:")
    print(f"  Router actual total cost: {total_cost}")
    print(f"  Always-frontier theoretical cost: {always_frontier_theoretical_cost}")
    print(f"  Real savings: {round((1 - total_cost/always_frontier_theoretical_cost)*100, 1)}%")

    fallback_logs = [l for l in logs if l.get("fallback_occurred")]
    print(f"  Total fallbacks triggered (all-time): {len(fallback_logs)}")
