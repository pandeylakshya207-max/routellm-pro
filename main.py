from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
from router.policy import route_query, redis_client

app = FastAPI(title="RouteLLM-Pro", version="0.1.0")

class QueryRequest(BaseModel):
    query: str

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

@app.get("/")
def root():
    return {"status": "RouteLLM-Pro is running"}

@app.post("/route", response_model=QueryResponse)
def route(request: QueryRequest):
    result = route_query(request.query)
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
        }

    tier_counts = {}
    total_cost = 0
    total_latency = 0

    for log in logs:
        tier = log.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_cost += log.get("cost_weight", 0)
        total_latency += log.get("latency_seconds", 0)

    cost_if_always_frontier = total * 10  # frontier cost_weight is 10
    savings_percent = round((1 - (total_cost / cost_if_always_frontier)) * 100, 1) if cost_if_always_frontier > 0 else 0

    return {
        "total_queries": total,
        "tier_distribution": tier_counts,
        "total_cost_weight": total_cost,
        "avg_latency": round(total_latency / total, 3),
        "cost_if_always_frontier": cost_if_always_frontier,
        "estimated_savings_percent": savings_percent,
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()
