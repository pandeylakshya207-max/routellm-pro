# RouteLLM-Pro

A cost-aware, multi-tier LLM routing system that classifies incoming queries by complexity and routes each to the cheapest capable model.

Live demo: https://routellm-pro.onrender.com/docs
Dashboard: https://routellm-pro.onrender.com/dashboard

(Free tier hosting spins down after ~15 min idle; first request after that takes 30-60s.)

## Architecture

Query -> Classifier -> Router -> Model Tier -> Response -> Redis Log -> Dashboard

Tiers:
- Small: Groq llama-3.1-8b-instant (simple queries)
- Mid: Groq llama-3.3-70b-versatile (reasoning, bounded coding tasks)
- Frontier: Gemini flash-latest (open-ended theoretical reasoning)

## Results (n=9, initial dataset)

- 73.3% cost reduction vs always-frontier baseline
- Tier split: 56% small, 33% mid, 11% frontier

## Tech Stack

FastAPI, Groq, Google Gemini, Upstash Redis, Chart.js, Render

## Running Locally

git clone https://github.com/pandeylakshya207-max/routellm-pro.git
cd routellm-pro
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

## Roadmap

- [x] Multi-tier routing
- [x] Redis logging + dashboard
- [x] Public deployment
- [ ] Trained classifier (Phase 2)
- [ ] A/B testing framework

Author: Lakshya Pandey - github.com/pandeylakshya207-max

## Phase 2: Trained Classifier - Benchmark Results

Attempted to replace the prompt-based classifier with a fine-tuned DistilBERT model, trained on
221 synthetically-generated labeled examples (Groq-generated, stratified across simple/medium/complex).

**Head-to-head benchmark (n=24, hand-written held-out test set):**

| Classifier | Accuracy | F1 (macro) |
|---|---|---|
| Prompt-based (production) | 95.8% | 0.958 |
| Trained DistilBERT | 83.3% | 0.829 |

**Finding:** The trained classifier systematically under-predicted "complex" (recall 0.62 vs 1.00
for prompt-based), misclassifying genuinely complex queries as "medium" when phrased as direct
requests (e.g. "Prove that...", "Design a...") rather than the declarative research-toned phrasing
("Derive...", "Develop...") that dominated the synthetic training data for that class. This points
to a phrasing-style bias inherited from LLM-generated training data, not a failure of the
architecture itself.

**Decision:** Retained the prompt-based classifier in production. The trained model requires a
larger, more phrasing-diverse dataset (ideally with some human-labeled real queries, not purely
synthetic) before it can be considered for deployment.
