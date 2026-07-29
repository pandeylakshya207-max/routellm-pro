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


### Iteration: Fixing the Phrasing Bias

Hypothesis: the "complex" class underperformed because synthetic training examples were dominated
by declarative research-toned phrasing ("Derive...", "Develop...") rather than direct-request
phrasing ("Prove...", "Design..."). Generated 155 additional phrasing-diverse "complex" examples
explicitly varying command style, retrained (v2).

**Results (same n=24 held-out test set):**

| Classifier | Accuracy | F1 (macro) |
|---|---|---|
| Prompt-based (production) | 95.8% | 0.958 |
| Trained v1 | 83.3% | 0.829 |
| Trained v2 (phrasing-diverse data) | 91.7% | 0.915 |

**Finding:** The targeted fix worked as hypothesized - complex-class recall improved from 0.62 to
1.00, correctly resolving all four of v1's original failures (P vs NP, distributed consensus,
Byzantine fault tolerance, prime number proof). However, this introduced a new tradeoff: medium-class
recall dropped from 1.00 to 0.75, with the model now over-predicting "complex" on some medium
queries - likely due to class imbalance introduced by adding 155 new complex examples against
~73 each for simple/medium.

**Decision:** Prompt-based classifier remains in production. v2 is a meaningful improvement over v1
(83.3% -> 91.7%) but still trails the baseline, and its complex/medium confusion has a real cost
implication for a routing system (misrouting medium queries to the expensive frontier tier).
Next step would be rebalancing the dataset rather than simply adding more complex examples.


### Iteration 2: Testing the Class-Imbalance Hypothesis

Hypothesis: v2's medium-recall drop (1.00 -> 0.75) was caused by class imbalance (155 complex vs
~73 each for simple/medium). Rebalanced to ~74 examples per class (v3), retrained.

**Results (same n=24 held-out test set):**

| Classifier | Accuracy | F1 (macro) |
|---|---|---|
| Prompt-based (production) | 95.8% | 0.958 |
| Trained v2 (imbalanced) | 91.7% | 0.915 |
| Trained v3 (balanced) | 91.7% | 0.913 |

**Finding:** Rebalancing had no effect - v3 performs identically to v2, with the exact same two
failure cases (both science *explanation* queries - "What causes seasons to change on Earth?" and
"Explain how photosynthesis works"). This falsifies the class-imbalance hypothesis. The more likely
explanation: these two queries share surface-level structural and lexical similarity with the
"complex" training examples (explanation-style phrasing, science/physics vocabulary), suggesting
the model is partly relying on domain/vocabulary cues rather than fully disentangling genuine
open-endedness from well-defined explanatory questions - a harder distinction that likely needs
either more training data specifically contrasting these two patterns, or a fundamentally
different feature representation.

**Final decision:** Prompt-based classifier remains in production (95.8% vs 91.7% best trained
result). The three-iteration experiment (v1 -> v2 -> v3) demonstrates the value of hypothesis-driven
iteration: the first fix (phrasing diversity) produced a real, measurable improvement (+8.4pp);
the second fix (class rebalancing) was a reasonable hypothesis that the data disproved, which is
itself a useful negative result ruling out one candidate explanation for the remaining gap.


## Phase 3: Confidence-Based Cascading (Scoped Out)

Attempted to add confidence-based cascading fallback: escalate to a higher tier when the
classifier is uncertain, using token log-probabilities as the confidence signal.

**Finding:** Groq's inference API does not expose `logprobs` on any available model
(llama-3.1-8b-instant, llama-3.3-70b-versatile) - confirmed via direct API testing, both
return `400: logprobs is not supported with this model`. This is a real constraint of Groq's
LPU-based fast-inference stack, which trades this capability for speed, unlike standard
GPU-serving APIs (OpenAI, most vLLM deployments) that support token-level log-probabilities.

A fallback approach (asking the model to self-report a confidence label) was tested and
rejected: it returned "high" confidence on 8/8 test queries, including nonsense input
("asdf jkl random text"), demonstrating that LLM self-reported confidence is not a reliable
signal without actual log-probability access.

**Decision:** Scoped out confidence-based cascading for this iteration rather than building on
an unreliable signal. Revisit if switching to a provider that exposes logprobs, or explore a
verifier-model approach (a second model scores the first model's answer) as an alternative.


## Note on Free-Tier Rate Limits

During development of the cost-accuracy Pareto analysis (Phase 3), hit Gemini's free-tier daily
quota (20 requests/day/model for gemini-flash-latest) mid-analysis. This is a genuine operational
constraint of building on free-tier LLM APIs worth noting: production systems would need either a
paid tier, request batching/caching, or multi-key rotation to avoid this in real usage. Pareto
analysis resumed after quota reset.


## Cost-Accuracy Validation (Production Data)

Attempted a controlled Pareto analysis (always-small vs always-frontier vs router) but repeatedly
hit Gemini's free-tier daily quota (20 req/day) mid-run. Rather than force this through, pivoted to
validating cost savings against real production log data instead - arguably stronger evidence,
since it reflects actual usage rather than a synthetic benchmark run.

**Real production numbers (n=37 logged queries, live Redis data):**

- Router actual total cost: 102 units
- Always-frontier theoretical cost: 370 units
- **Real-world cost savings: 72.4%** (consistent with the earlier 73.3% figure from a smaller
  sample, confirming stability of the savings estimate)
- Fallback chain triggered 7 times across 37 queries (18.9% of requests), all recovering
  gracefully to a lower tier without failing the user's request

**Methodology note:** average frontier latency from historical logs (22.5s, n=3) is likely
inflated by rate-limit retry behavior during earlier testing and should not be treated as a
typical per-request latency figure; a controlled, quota-unconstrained benchmark would be needed
for a clean latency comparison.
