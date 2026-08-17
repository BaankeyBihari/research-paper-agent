# research-paper-agent

[github.com/BaankeyBihari/research-paper-agent](https://github.com/BaankeyBihari/research-paper-agent)

A small Dockerized research assistant built on [NVIDIA NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
(Object-Oriented Agents), used to try the framework's "agent as a Python class" model and to compare
OpenRouter models on summarization quality/speed.

`ResearchAgent` scans `/data/papers` for PDFs, extracts each paper's Main Objective, Key Methodology,
and Resulting Metrics via an LLM call, and records the result in a local SQLite file so re-runs (even
under a different model) don't reprocess the same paper.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
docker compose up --build
```

## Viewing results

**The processed summaries** (title, objective, methodology, metrics) live in a SQLite table inside
the container and are the most reliable way to check what happened:

```bash
docker compose exec research-agent python -c "
import sqlite3, json
with sqlite3.connect('/data/papers/.agent_state.sqlite3') as conn:
    conn.row_factory = sqlite3.Row
    for r in conn.execute('SELECT * FROM processed_papers ORDER BY processed_at'):
        print(json.dumps(dict(r), indent=2))
"
```

**The trace viewer** (every LLM call, prompt, and method invocation NOOA makes) is trickier to reach
from a browser on Windows/Mac. It runs inside the container on port 5001, but its API refuses any
request that doesn't look like loopback traffic once bound to `0.0.0.0` — and Docker's port
publishing (`-p 5001:5001`) always makes host-side requests look like they came from a bridge/NAT
address, not `127.0.0.1`, so a plain `docker compose up` + `localhost:5001` in your browser gets a
403. Running `nooa start-dev` directly on Windows as a workaround doesn't work either — its storage
layer imports the Unix-only `fcntl` module.

The fix that actually works: open this folder in **VS Code with the Dev Containers extension**
("Reopen in Container"). VS Code's port forwarding tunnels from *inside* the container's own network
namespace rather than through Docker's external NAT, so the connection genuinely looks like loopback
traffic to the viewer and it just works — no token, no extra config. `.devcontainer/devcontainer.json`
is already set up to forward port 5001 and open it in your browser automatically. If you're not using
VS Code, the SQLite query above is the dependable fallback.

## Switching models

Set `ACTIVE_MODEL_SLUG` in `.env` (or inline) to any OpenRouter slug, then rebuild/rerun:

| Tier | Slug |
|---|---|
| Fast/Cheap | `nvidia/nemotron-3.5-lightning` |
| Efficient | `nvidia/nemotron-3-nano-30b-a3b` |
| High-Reasoning | `meta-llama/llama-3.1-70b-instruct` |

```bash
ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build
```

## Testing efficiency at home

1. **Nano run**: `ACTIVE_MODEL_SLUG=nvidia/nemotron-3-nano-30b-a3b docker compose up --build`.
   Watch time-to-first-token and summary quality in the trace viewer.
2. **Lightning run**: switch the env var and rerun. Compare summary quality, especially on
   math/code-heavy papers.
3. **Persistence check**: the `papers_data` Docker volume holds both the downloaded PDFs and
   `.agent_state.sqlite3`. Because that volume persists across runs regardless of which model is
   active, a paper processed under Nano is still recorded (title + extracted fields) when you
   switch to Lightning — inspect it directly:
   ```bash
   docker compose run --rm research-agent python -c \
     "from agent import ResearchAgent; import asyncio; a = ResearchAgent(); print(a.lookup_paper('<filename>.pdf'))"
   ```

## Comparing models against a reference

`compare_models.py` runs the same papers through a stronger "reference" model and one or more
cheaper "candidate" models via NOOA's real `eval_pipeline` package, then scores each candidate
against the reference's output with plain text similarity (`difflib`, stdlib only — no extra API
calls, no LLM-judge self-bias risk).

```bash
docker compose exec research-agent python compare_models.py
```

Defaults: 2 papers, reference = `openai/gpt-4o-mini`, candidates = both Nemotron tiers. Override via
env vars: `COMPARE_NUM_PAPERS`, `REFERENCE_MODEL_SLUG`, `CANDIDATE_MODEL_SLUGS` (comma-separated). A
summary prints to stdout; the full run is written as a `.noo-eval.jsonl` file under
`/data/papers/eval_results/`, which the trace viewer's **Evaluations** tab reads directly (see
"Viewing results" above for reaching the viewer).

`eval_pipeline` is not on PyPI, but it doesn't need a full monorepo clone either: it installs
straight from a git subdirectory URL (`pip install "eval_pipeline @
git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@<commit>#subdirectory=util/eval_pipeline"`,
pinned in `requirements.txt` to a specific commit), since its `pyproject.toml` is a self-contained
`hatchling` package with plain PyPI dependencies. `agent.py`'s `ResearchAgent` is handed to it
directly as `agent_class` — no throwaway per-model subclass needed, since NOOA's `Agent` supports a
per-instance `llm=` override.

**Cost note**: `openai/gpt-4o-mini` is paid (see OpenRouter's pricing page for current rates), unlike
the free-tier Nemotron candidate models. Each run costs 1 reference call per paper; use
`COMPARE_NUM_PAPERS` to bound it. `meta-llama/llama-3.1-70b-instruct` (an earlier default here) was
dropped after live verification: it reliably fails to return structured output under NOOA's
tool-calling strategy via OpenRouter — reproduces across multiple papers regardless of content, so
it's a model/provider incompatibility, not a prompt or extraction issue. Also note OpenRouter has
discontinued the free tier for that model entirely (`:free` now 404s), so that cost-saving trick no
longer applies to it either way.

**Reading the scores**: `difflib` similarity is sequence/character-based, not semantic — it
correctly catches gross divergence (and titles, which are supposed to match verbatim, are a good
sanity check that scores near 1.0), but it under-rewards accurate paraphrasing. A candidate scoring
~0.5 isn't necessarily half as good; read the actual summaries side by side, not just the number. Two
papers is also too small a sample for real conclusions — it's a cost-conscious default, bump
`COMPARE_NUM_PAPERS` if you want something you'd actually trust.

## Known deviations from the original spec, and why

- **Structured output is a Pydantic model, not a raw dict.** NOOA's documented convention for
  agentic methods with a structured return type is a `pydantic.BaseModel` subclass — that's what
  `summarize_paper` returns (`PaperSummary`). Functionally equivalent to a dict (same four fields:
  title, main_objective, key_methodology, resulting_metrics) but validated by the framework.
- **"NOOA's SQLite memory" is not the `nooa-memory` package.** `nooa-memory` (included in
  `requirements.txt` for you to explore) is a vector-search long-term recall subsystem meant for
  semantic recall, not exact-match dedup, and its API wasn't documented in enough depth to wire up
  with confidence. Instead, `ResearchAgent` tracks processed papers via a plain `sqlite3` table
  (`processed_papers`) in the same `/data/papers` volume — deterministic, and it's what actually
  makes the persistence test in step 3 above work.

## Verified

Confirmed end-to-end with a live run against `nvidia/nemotron-3-nano-30b-a3b` on OpenRouter:
- Image builds cleanly; `simulator.py` downloads real papers from the live arXiv API; `pypdf`
  extracts their text; `summarize_paper` produces valid `PaperSummary` objects from real model
  output; the SQLite `processed_papers` table persists correctly across `docker compose down`/`up`
  cycles (i.e. across the same volume you'd reuse when switching `ACTIVE_MODEL_SLUG`), confirming
  the cross-model persistence test in step 3 actually holds.
- `nooa start-dev` binds `0.0.0.0:5001` by default and `GET /` returns 200 (serves the SPA shell),
  **but** its data API 403s from outside the container over a published Docker port — the page loads
  and looks blank. The `.devcontainer` workaround (see "Viewing results" above) has been confirmed
  working from an actual VS Code Dev Containers session: traces show up correctly.
- The trace viewer's **Evaluations** tab is populated by real `eval_pipeline` runs — see
  `compare_models.py` above. The **Memory** tab will still look empty, and that's expected, not
  broken: it reads a `nooa_memory.MemoryStore` file that `ResearchAgent` never creates (see the
  SQLite-vs-`nooa-memory` deviation above).
- `compare_models.py` confirmed end-to-end against `eval_pipeline` (both a free-tier dev pass and a
  paid confirmation pass with `openai/gpt-4o-mini`): reference summarization, `Evaluator.add_test`,
  candidate scoring via a custom `PaperSimilarityScorer`, and a well-formed `.noo-eval.jsonl` output
  file (verified metadata/result/completion lines and per-field score breakdowns) written to
  `localhost:5001`'s OTLP endpoint, which the run log confirmed as reachable (`Viewer:
  http://localhost:5001/eval/experiment/...` printed, meaning `eval_pipeline`'s health probe
  succeeded). Two bugs surfaced and were fixed along the way: `docker-compose.yml` wasn't forwarding
  `REFERENCE_MODEL_SLUG`/`CANDIDATE_MODEL_SLUGS`/`COMPARE_NUM_PAPERS` into the container despite
  `.env.example` documenting them (now fixed); and `eval_pipeline`'s own `Evaluator.run()` (commit
  `8622fc4`, the version pinned in `requirements.txt`) raises a `pydantic.ValidationError` building
  its run-metadata line when using its documented plain-Python API (`Evaluator(models={...})`) rather
  than the YAML/`from_config` path — `_model_metadata` is only ever populated by the latter.
  `compare_models.py` works around this by pre-populating `evaluator._model_metadata` directly; worth
  removing if a future `eval_pipeline` release fixes it upstream.

One tuning note from the live run: on `nemotron-3-nano-30b-a3b`, the first version of the
`summarize_paper` docstring sometimes produced a `title` containing the full author list, and a
`main_objective` that was a verbatim sentence lift rather than a synthesis. The docstring now gives
explicit per-field instructions (title = paper title only, no authors; objective = synthesized, not
copied; metrics = "not reported" when absent) — this measurably cleaned up the output in a follow-up
run. Worth watching for the same failure mode if you try other budget/free-tier models.
