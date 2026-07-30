from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from router.policy import route_query, redis_client

app = FastAPI(title="RouteLLM-Pro", version="0.3.0")

class QueryRequest(BaseModel):
    query: str
    ab_variant: Optional[str] = None  # "A" or "B", or omit for random assignment

class QueryResponse(BaseModel):
    text: str
    tier: str
    model: str
    provider: str
    complexity_label: str
    latency_seconds: float
    cost_weight: int
    timestamp: float
    query: str
    fallback_occurred: bool = False
    attempted_tiers: List[str] = []
    ab_variant: str = "A"

@app.get("/")
def root():
    return {"status": "RouteLLM-Pro is running"}

@app.post("/route", response_model=QueryResponse)
def route(request: QueryRequest):
    result = route_query(request.query, ab_variant=request.ab_variant)
    return result

@app.get("/stats")
def stats():
    raw_logs = redis_client.lrange("routellm:logs", 0, -1)
    logs = [json.loads(entry) for entry in raw_logs]

    total = len(logs)
    if total == 0:
        return {
            "total_queries": 0,
            "tier_distribution": {},
            "total_cost_weight": 0,
            "avg_latency": 0,
            "cost_if_always_frontier": 0,
            "estimated_savings_percent": 0,
            "fallback_count": 0,
        }

    tier_counts = {}
    total_cost = 0
    total_latency = 0
    fallback_count = 0

    for log in logs:
        tier = log.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_cost += log.get("cost_weight", 0)
        total_latency += log.get("latency_seconds", 0)
        if log.get("fallback_occurred"):
            fallback_count += 1

    cost_if_always_frontier = total * 10
    savings_percent = round((1 - (total_cost / cost_if_always_frontier)) * 100, 1) if cost_if_always_frontier > 0 else 0

    return {
        "total_queries": total,
        "tier_distribution": tier_counts,
        "total_cost_weight": total_cost,
        "avg_latency": round(total_latency / total, 3),
        "cost_if_always_frontier": cost_if_always_frontier,
        "estimated_savings_percent": savings_percent,
        "fallback_count": fallback_count,
    }

@app.get("/ab-stats")
def ab_stats():
    raw_logs = redis_client.lrange("routellm:logs", 0, -1)
    logs = [json.loads(entry) for entry in raw_logs]

    variants = {"A": [], "B": []}
    for log in logs:
        variant = log.get("ab_variant")
        if variant in variants:
            variants[variant].append(log)

    result = {}
    for variant, entries in variants.items():
        n = len(entries)
        if n == 0:
            result[variant] = {"n": 0}
            continue
        total_cost = sum(e.get("cost_weight", 0) for e in entries)
        total_latency = sum(e.get("latency_seconds", 0) for e in entries)
        tier_counts = {}
        for e in entries:
            t = e.get("tier", "unknown")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        result[variant] = {
            "n": n,
            "avg_cost_per_query": round(total_cost / n, 3),
            "avg_latency": round(total_latency / n, 3),
            "tier_distribution": tier_counts,
        }

    return result

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()
