import json
import time
from router.providers import call_small, call_mid, call_frontier
from router.policy import route_query

TEST_QUERIES = [
    "What is the capital of France?",
    "How many continents are there?",
    "What year did World War 2 end?",
    "Say hello to me",
    "What is the boiling point of water in Celsius?",
    "Convert 10 miles to kilometers",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "Explain the difference between TCP and UDP",
    "How does a hash table handle collisions?",
    "Write a function to check if a string is a palindrome",
    "What causes seasons to change on Earth?",
    "Explain how photosynthesis works",
    "Compare REST and GraphQL APIs",
    "How does garbage collection work in Java?",
    "Write a Python function to reverse a linked list",
    "Explain the mathematical proof behind why P vs NP is unsolved",
    "What are the philosophical implications of the simulation hypothesis?",
    "Derive the equations of general relativity from the equivalence principle",
    "Design a distributed consensus algorithm resilient to Byzantine faults",
    "What are the fundamental limits of computability according to the halting problem?",
    "Analyze the economic tradeoffs of universal basic income at a macro scale",
    "Prove that there are infinitely many prime numbers",
    "What ethical framework should guide autonomous weapons decision-making?",
]

COST_WEIGHTS = {"small": 1, "mid": 3, "frontier": 10}

def run_policy_always_small():
    total_cost = 0
    total_latency = 0
    for q in TEST_QUERIES:
        start = time.time()
        call_small(q)
        total_latency += time.time() - start
        total_cost += COST_WEIGHTS["small"]
    return total_cost, total_latency

def run_policy_always_frontier_partial():
    """Runs until quota is hit, returns partial results + how many succeeded."""
    total_cost = 0
    total_latency = 0
    successes = 0
    for i, q in enumerate(TEST_QUERIES):
        try:
            start = time.time()
            call_frontier(q)
            total_latency += time.time() - start
            total_cost += COST_WEIGHTS["frontier"]
            successes += 1
            print(f"  frontier call {i+1}/{len(TEST_QUERIES)} succeeded")
            time.sleep(13)
        except Exception as e:
            print(f"  frontier call {i+1} failed (quota likely exhausted): stopping here")
            break
    return total_cost, total_latency, successes

def run_policy_router():
    total_cost = 0
    total_latency = 0
    tier_counts = {"small": 0, "mid": 0, "frontier": 0}
    fallback_count = 0
    for q in TEST_QUERIES:
        result = route_query(q)
        total_cost += result["cost_weight"]
        total_latency += result["latency_seconds"]
        tier_counts[result["tier"]] += 1
        if result.get("fallback_occurred"):
            fallback_count += 1
    return total_cost, total_latency, tier_counts, fallback_count

print(f"Running Pareto analysis on {len(TEST_QUERIES)} queries...\n")

print("Policy: always-small...")
small_cost, small_latency = run_policy_always_small()
print(f"  done: cost={small_cost}, latency={round(small_latency,2)}s")

print("\nPolicy: always-frontier (will stop early if quota exhausted)...")
frontier_cost, frontier_latency, frontier_successes = run_policy_always_frontier_partial()
print(f"  completed {frontier_successes}/{len(TEST_QUERIES)} calls before stopping")

print("\nPolicy: RouteLLM-Pro (actual router, includes automatic fallback)...")
router_cost, router_latency, tier_counts, fallback_count = run_policy_router()
print(f"  done: cost={router_cost}, latency={round(router_latency,2)}s, fallbacks triggered={fallback_count}")

n = len(TEST_QUERIES)

# Extrapolate frontier cost/latency if we got a partial sample
if frontier_successes > 0:
    frontier_avg_cost = frontier_cost / frontier_successes
    frontier_avg_latency = frontier_latency / frontier_successes
    frontier_extrapolated_total_cost = frontier_avg_cost * n
else:
    frontier_avg_cost = None
    frontier_avg_latency = None
    frontier_extrapolated_total_cost = None

results = {
    "n_queries": n,
    "always_small": {
        "total_cost": small_cost,
        "avg_cost_per_query": round(small_cost / n, 3),
        "avg_latency": round(small_latency / n, 3),
    },
    "always_frontier": {
        "n_sampled": frontier_successes,
        "sampled_total_cost": frontier_cost,
        "avg_cost_per_query": round(frontier_avg_cost, 3) if frontier_avg_cost else None,
        "avg_latency": round(frontier_avg_latency, 3) if frontier_avg_latency else None,
        "extrapolated_total_cost_for_all_queries": round(frontier_extrapolated_total_cost, 1) if frontier_extrapolated_total_cost else None,
        "note": "Limited by Gemini free-tier daily quota (20 req/day); cost/latency per-query are measured directly, total is extrapolated" if frontier_successes < n else "Full sample",
    },
    "router": {
        "total_cost": router_cost,
        "avg_cost_per_query": round(router_cost / n, 3),
        "avg_latency": round(router_latency / n, 3),
        "tier_distribution": tier_counts,
        "fallback_count": fallback_count,
        "cost_savings_vs_frontier_pct": round((1 - (router_cost / n) / frontier_avg_cost) * 100, 1) if frontier_avg_cost else None,
    },
}

print("\n" + "=" * 70)
print("PARETO ANALYSIS RESULTS")
print("=" * 70)
print(json.dumps(results, indent=2))

with open("pareto_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to pareto_results.json")
