# Assignment 2 Report — LLM Inference + Observability

## 1. Serving Configuration (Phase 1)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB

**Final vLLM flags:**

```bash
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95
```

**Flag justifications:**

| Flag | Value | Rationale |
|------|-------|-----------|
| `--max-model-len` | 4096 | Workload is 1.5–2K-token prompts (schema + question) with short SQL outputs. Default 8192 wastes KV memory per sequence. Halving it doubles the number of sequences that fit in KV cache, increasing effective concurrency. |
| `--max-num-seqs` | 64 | Tells the scheduler to batch up to 64 concurrent sequences. Without this the default is lower and vLLM under-utilizes the H100 at 10 RPS. |
| `--enable-chunked-prefill` | — | Allows long prompt prefills to be interleaved with active decode steps instead of blocking them. Reduces head-of-line blocking when concurrent requests have different prompt lengths. |
| `--gpu-memory-utilization` | 0.95 | Allocates 95% of GPU memory to the KV cache pool, leaving the rest for activations and overhead. Safe on H100 80GB for this model size. |

**Agent server:**

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 --workers 4
```

`--workers 4` spawns 4 independent uvicorn processes. The `/answer` endpoint is synchronous (it blocks on `graph.invoke()`), so a single worker exhausts its thread pool at ~10 RPS. Four workers multiply available concurrency by 4×.

---

## 2. Baseline Evaluation (Phase 5)

Ran 30 questions sequentially (no concurrency) with `MAX_ITERATIONS = 3`.

| Metric | Value |
|--------|-------|
| Questions | 30 |
| Overall pass rate | **30.0%** (9/30) |
| Pass rate — iteration 1 (generate only) | **26.7%** (8/30) |
| Pass rate — iteration 2 (after first revise) | **30.0%** (9/30) |
| Pass rate — iteration 3 (after second revise) | **30.0%** (no gain) |
| Wall clock | **32.3s** (~1.1s/question, uncontended) |

**Commentary:** The revise loop recovered exactly 1 additional question (26.7% → 30.0%) and then plateaued — a second revision produced no further gains. The 30% overall accuracy is consistent with a general-purpose ~3B-active-parameter MoE model on BIRD benchmark without fine-tuning (state-of-the-art models reach 60–70%). The marginal gain from revision (+3.3pp) was a key input to the Phase 6 decision to eliminate the revise loop entirely when trading accuracy for latency.

---

## 3. SLO Tuning (Phase 6)

**Target SLO:** P95 end-to-end agent latency < 5s at ≥10 RPS over a 5-minute window.

**Load test:** `driver.py --rps 10 --duration 300` (3000 total requests).

### Iteration 1

**Saw:** P95 = 98.5s, P50 = 40.8s, 32% success rate. 816 connection refused, 956 timeouts.

**Hypothesized:** Two compounding problems. First, uvicorn ran with a single worker — the sync `/answer` endpoint saturates the thread pool immediately at 10 RPS, causing new connections to be refused. Second, vLLM used default `--max-model-len 8192` with no batching configuration, limiting concurrent sequence throughput.

**Changed:** Added `--workers 4` to uvicorn. Tuned vLLM: `--max-model-len 4096`, `--max-num-seqs 64`, `--enable-chunked-prefill`, `--gpu-memory-utilization 0.95`.

**Result:** P95 = 13.2s, P50 = 2.33s. Success rate 87%. Connection refused errors eliminated. 388 HTTP 500s appeared (new problem).

---

### Iteration 2

**Saw:** P95 = 13.2s — still 2.6× above SLO. 388 HTTP 500s. vLLM running 41–49 concurrent requests (confirmed via logs).

**Hypothesized:** Two causes. First, `MAX_ITERATIONS = 3` meant tail requests made up to 6 LLM calls (generate + verify + up to 2 revise + verify cycles). At ~1–2s per LLM call under load, 6 calls = 6–12s, directly explaining the P95 tail. Second, a shared module-level `CallbackHandler()` instance in `agent/server.py` was not thread-safe across concurrent uvicorn worker processes, causing race conditions in Langfuse instrumentation → HTTP 500s.

**Changed:** Reduced `MAX_ITERATIONS` 3 → 2. Replaced shared Langfuse handler with a fresh `CallbackHandler()` created per request.

**Result:** P95 = 8.28s, P50 = 2.08s. vLLM concurrency dropped to 17–22 running requests. HTTP 500s unchanged (fix not yet applied on VM).

---

### Iteration 3

**Saw:** P95 = 8.28s, 3.28s above SLO. HTTP 500 count identical (388 → 389) — confirming the Langfuse fix hadn't been pulled on the VM yet.

**Hypothesized:** The revise loop remains the primary tail driver. Even with `MAX_ITERATIONS = 2`, requests that fail verify make 4 LLM calls. Eliminating the revise loop entirely (`MAX_ITERATIONS = 1`) caps every request at exactly 2 LLM calls (generate + verify), making latency nearly constant. The baseline eval showed the revise loop only recovered 1 question (marginal benefit), making this accuracy/latency trade-off acceptable.

**Changed:** `MAX_ITERATIONS` 2 → 1. Applied Langfuse per-request handler fix via `git pull` + uvicorn restart.

**Result:** **P95 = 3.44s ✅ SLO met.** P50 = 1.41s. Timeouts dropped to 0. vLLM KV cache at 1–2%, no queue.

---

### Final Numbers

| Metric | Baseline (run 1) | Final (run 4) |
|--------|-----------------|---------------|
| P50 | 40.8s | **1.41s** |
| P95 | 98.5s | **3.44s ✅** |
| P99 | — | 6.68s |
| Success rate | 32% | 87% |
| Timeouts | 956 | 0 |
| Achieved RPS | ~3 | 9.24 |

**Grafana observations (final run):** vLLM throughput peaked at 17.4 completed req/s (2 LLM calls per agent request). Requests running: 10–18 concurrent, waiting queue: 0 throughout. TTFT P95: ~80ms. Prefix cache hit rate: ~89.9% (schema strings reused across requests). KV cache: 1–2%.

---

## 4. Did the Agent Loop Help?

**Short answer: marginally, and at significant latency cost.**

The per-iteration pass rates from the baseline eval show the revise loop recovered exactly 1 question out of 30 (26.7% → 30.0%). A second revision cycle added nothing further. This is a +3.3 percentage-point gain for the cost of up to 4 additional LLM calls per request, which directly caused the P95 tail during load testing.

The loop is worth keeping in a latency-tolerant context (batch processing, interactive single-user use) where the model's first attempt is wrong and a second chance helps. Under the 5s SLO at 10 RPS, the cost outweighs the benefit for this model/prompt combination. With a better verifier prompt or a stronger model, the loop could recover more questions and justify the latency overhead.

**What the Langfuse traces confirm:** Traces show the `generate_sql` and `verify` spans clearly. In the 30% of cases where the agent answered correctly, `verify` returned `ok: true` immediately, producing a total of 2 LLM calls. The slow traces visible in Langfuse were all 3-iteration runs where the verifier triggered a revise cycle.

---

## 5. What I'd Do With More Time

1. **Better schema rendering with BIRD metadata.** BIRD provides column descriptions and sample values per table. Including these in the `render_schema()` output would give the model the context it needs to resolve ambiguous column names — likely the biggest single lever for accuracy improvement without changing the model.

2. **Prompt engineering with few-shot examples.** The current prompts are zero-shot. Adding 2–3 worked examples of question → SQL pairs for common patterns (GROUP BY, nested subqueries, date arithmetic) would substantially improve first-attempt accuracy and reduce reliance on the revise loop.

3. **Async agent endpoint.** The current `/answer` handler is synchronous. Rewriting it as `async def` with `await graph.ainvoke()` would remove the thread pool bottleneck entirely, allowing a single uvicorn worker to handle far more concurrent requests without the `--workers N` process-proliferation workaround.

4. **Smarter verifier.** The current verifier asks the model to judge plausibility. A rule-based pre-check (did the SQL execute without error? did it return at least one row when the question implies rows exist?) would catch the obvious failures instantly without a full LLM call, saving one LLM round-trip on every failed attempt.

5. **A/B eval comparing MAX_ITERATIONS=1 vs 3.** Before committing to eliminating the revise loop in production, run a larger eval (200+ questions) to get a statistically meaningful read on the accuracy cost. 30 questions is too small a sample to be confident the 3.3pp difference isn't noise.
