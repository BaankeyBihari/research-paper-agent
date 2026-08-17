# research-paper-agent

[github.com/BaankeyBihari/research-paper-agent](https://github.com/BaankeyBihari/research-paper-agent)

A small Dockerized research assistant built on [NVIDIA NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
(Object-Oriented Agents), used to try the framework's "agent as a Python class" model and to compare
OpenRouter models on summarization quality/speed.

`ResearchAgent` scans `/data/papers` for PDFs, extracts each paper's Main Objective, Key Methodology,
and Resulting Metrics via an LLM call, and records the result in a local SQLite file so re-runs (even
under a different model) don't reprocess the same paper.

## How this uses NOOA

NOOA isn't used as one monolithic thing — this project leans on five distinct surfaces of the
framework, each for a specific reason:

1. **Core agent pattern.** `ResearchAgent(Agent, llm=llm)` follows NOOA's fields-are-state,
   methods-are-capabilities convention. `summarize_paper`'s body is literally `...` — its docstring
   *is* the prompt, and NOOA's runtime drives the LLM call and validates the response against a
   Pydantic model (`PaperSummary`) automatically. Everything else (paper selection, dedup,
   persistence) is plain deterministic Python around that one agentic method.
2. **Model-agnostic LLM routing** (`nooa.unifiedllm.registry.get_llm_client`). Built once at module
   load from `ACTIVE_MODEL_SLUG` (any OpenRouter slug), so swapping models is a pure env-var change
   — no code touch. This is what makes "Switching models" and "Comparing models" below possible.
3. **Evaluation harness** (`eval_pipeline`, used by `compare_models.py`). NOOA's `Evaluator` runs the
   *same* agent method across multiple candidate models against a reference model's output, via a
   duck-typed `Scorer` interface implemented once (`PaperSimilarityScorer`). The parallel runs, trace
   correlation, and structured `.jsonl` output are NOOA doing the orchestration, not hand-rolled here.
   `PaperSimilarityScorer` itself is intentionally basic (plain `difflib` text diff, not semantic —
   see "Reading the scores" below) and will stay that way; this side-quest demonstrates `eval_pipeline`
   wiring, not scoring quality.
4. **Long-term memory** (`nooa-memory`, used by `find_similar_papers`). Deliberately narrow: not the
   full agentic surface (`MemoryManager`/`MemoryToolsMixin`, where the agent itself decides what to
   remember via tool calls, with reflection/decay on top) — that's built for agents reasoning across
   long horizons, which doesn't fit a one-shot "extract a summary" agent. Instead this uses
   `MemoryStore`/`Embedder` directly as building blocks: embed each summary, store it, `knn()` search
   it. See "Known deviations" below for the full reasoning.
5. **Observability** (auto-tracing + the trace viewer). Every method call and LLM call on a NOOA
   `Agent` is traced automatically, no code here does this. `nooa start-dev` serves that data through
   a viewer with three tabs, two of which this project actually populates: **Evaluations** (fed by
   `eval_pipeline`, #3 above) and **Memory** (fed by `MemoryStore`, #4 above).

The throughline: NOOA supplies the LLM-calling, evaluating, and memory infrastructure as composable
pieces; this project's own code is the deterministic scaffolding around a few narrow, deliberate uses
of each piece — not maximal use of the framework, but use of the parts that actually fit the problem.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
docker compose up --build
```

To discover live OpenRouter model slugs/pricing before editing `.env`, start the local selector UI:

```bash
python scripts/select_model.py
```

This starts a local-only server on `127.0.0.1` and opens (or prints) a URL with a searchable,
sortable table of all fetched models. Use the page controls to filter by family/tier, search by
slug, then choose which env key to update (`ACTIVE_MODEL_SLUG`, `REFERENCE_MODEL_SLUG`, or
`CANDIDATE_MODEL_SLUGS` with append/replace).

Additional options:

```bash
# Print the fetched model catalog as JSON (for scripting)
python scripts/select_model.py --json

# Do not auto-open a browser; only print the local URL
python scripts/select_model.py --no-browser
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
| High-Reasoning (paid) | `openai/gpt-4o-mini` |

(`meta-llama/llama-3.1-70b-instruct` previously held the High-Reasoning slot here, but live
verification found it reliably fails to return structured output under NOOA's tool-calling strategy
via OpenRouter — see the "Comparing models against a reference" cost note below for detail.)

```bash
ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build
```

## Choosing which papers to fetch/process

By default, `simulator.py` downloads 3 random recent papers and `agent.py` processes everything
pending in `PAPERS_DIR`. Two env vars narrow that down, with `NUM_PAPERS` taking precedence over
`ARXIV_PAPER_IDS` entirely when both are set:

- `ARXIV_PAPER_IDS=2608.11597,2608.11590` — fetch/process specifically these arXiv IDs (versioned
  or not) instead of random papers / everything pending.
- `NUM_PAPERS=5` — fetch/process up to this many (an upper bound, not a guarantee: the simulator
  skips feed entries that already exist on disk or have no PDF link, and the agent may simply have
  fewer than N pending PDFs), falling back to random selection (fetch) or first-N-pending (process),
  ignoring `ARXIV_PAPER_IDS` if it's also set.

```bash
ARXIV_PAPER_IDS=2608.11597,2608.11590 docker compose up --build
```

`compare_models.py` has its own scoped equivalents, `COMPARE_ARXIV_PAPER_IDS` /
`COMPARE_NUM_PAPERS`, with the same precedence — see "Comparing models against a reference" below.

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

## Finding similar papers

Beyond exact-match dedup (`lookup_paper`), `ResearchAgent.find_similar_papers(query, k=5)` does
semantic search over every processed paper's summary, backed by `nooa_memory.MemoryStore`
(deterministic, no LLM/network call — see "Known deviations" below for why this bypasses
`nooa-memory`'s agentic tool surface):

```bash
docker compose exec research-agent python -c \
  "from agent import ResearchAgent; print(ResearchAgent().find_similar_papers('transformer attention mechanism'))"
```

Returns a list of `{"filename", "title", "score"}` dicts, ranked by cosine similarity. Records are
also browsable in the trace viewer's **Memory** tab (see "Verified" below) — the store lives at
`.nooa/memory/memory.sqlite` (volumed via `./nooa-memory:/app/.nooa`), not under `/data/papers`, so
it lands where the viewer auto-discovers it.

## Comparing models against a reference

`compare_models.py` runs the same papers through a stronger "reference" model and one or more
cheaper "candidate" models via NOOA's real `eval_pipeline` package, then scores each candidate
against the reference's output with plain text similarity (`difflib`, stdlib only — no extra API
calls, no LLM-judge self-bias risk).

```bash
docker compose exec research-agent python compare_models.py
```

Defaults: 2 papers, reference = `openai/gpt-4o-mini`, candidates = both Nemotron tiers. Override via
env vars: `COMPARE_NUM_PAPERS`, `COMPARE_ARXIV_PAPER_IDS` (comma-separated arXiv IDs, precedence as
in "Choosing which papers to fetch/process" above), `REFERENCE_MODEL_SLUG`, `CANDIDATE_MODEL_SLUGS`
(comma-separated). A summary prints to stdout; the full run is written as a `.noo-eval.jsonl` file
under `/data/papers/eval_results/`. That file isn't what populates the trace viewer's
**Evaluations** tab, though: the tab is backed by the viewer's OTLP store, and `eval_pipeline`
posts eval spans to it live while `Evaluator.run()` executes -- so the tab only fills in if the
viewer was already reachable *during* the run (see "Viewing results" above for reaching it before
running `compare_models.py`). The `.jsonl` file is a separate, offline copy of the same results,
useful even without the viewer running.

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

**`PaperSimilarityScorer` is intentionally basic, and will stay that way.** It's a plain `difflib`
character-sequence diff, nothing more — no semantic understanding, so a candidate that correctly
paraphrases a field in different words scores poorly even when the content is right (this is why
"0/N passed" from `compare_models.py` doesn't mean the candidates are broken — see "Reading the
scores" above and, if you've hit this, the note in "How this uses NOOA"). Replacing it with something
semantic (embedding cosine similarity, an LLM judge) is a real, known option, but `compare_models.py`
is a side demonstrator for trying `eval_pipeline`'s wiring, not the focus of this project — improving
the scorer itself isn't planned work for the foreseeable future.

## Known deviations from the original spec, and why

- **Structured output is a Pydantic model, not a raw dict.** NOOA's documented convention for
  agentic methods with a structured return type is a `pydantic.BaseModel` subclass — that's what
  `summarize_paper` returns (`PaperSummary`). Functionally equivalent to a dict (same four fields:
  title, main_objective, key_methodology, resulting_metrics) but validated by the framework.
- **Exact-match dedup doesn't use `nooa-memory`'s agentic tool surface.** `ResearchAgent` still tracks
  processed papers via a plain `sqlite3` table (`processed_papers`) in the same `/data/papers` volume
  — deterministic, and it's what actually makes the persistence test in step 3 above work.
  `nooa-memory` *is* used, but only for semantic recall ("Finding similar papers" above), via its
  lower-level `MemoryStore`/`Embedder` primitives directly rather than `MemoryManager`/
  `MemoryToolsMixin` — that agentic surface (LLM-authored `remember`/`recall` tools, reflection,
  decay/forgetting) is built for an agent that autonomously curates its own memory during reasoning,
  which doesn't fit `process_pending_papers`'s fully deterministic orchestration loop.

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
  `compare_models.py` above. The **Memory** tab is populated by `nooa_memory.MemoryStore`
  (see "Finding similar papers" / "Known deviations" above) — confirmed live via the viewer's own
  `/api/memory/dbs` and `/api/memory/records` endpoints, no manual `?db=` needed. This required
  moving the store off `/data/papers`: the viewer's own source (`nooa/viewer/memory_routes.py`) only
  opens (and only auto-discovers) a store under its own process cwd (`/app` in this container), so
  the memory store now lives at `.nooa/memory/memory.sqlite` instead — matching
  `nooa_memory.MemoryManager`'s own default-path convention — with its own `docker-compose.yml`
  volume (`./nooa-memory:/app/.nooa`) for persistence.
- `compare_models.py` confirmed end-to-end against `eval_pipeline` (both a free-tier dev pass and a
  paid confirmation pass with `openai/gpt-4o-mini`): reference summarization, `Evaluator.add_test`,
  candidate scoring via a custom `PaperSimilarityScorer`, and a well-formed `.noo-eval.jsonl` output
  file (verified metadata/result/completion lines and per-field score breakdowns) written locally
  under `/data/papers/eval_results/`. Separately, the run log confirmed `eval_pipeline` also reached
  `localhost:5001`'s OTLP endpoint live during the run (`Viewer:
  http://localhost:5001/eval/experiment/...` printed, meaning `eval_pipeline`'s health probe
  succeeded) -- these are two distinct outputs of the same run, not one file being read by the
  viewer. Two bugs surfaced and were fixed along the way: `docker-compose.yml` wasn't forwarding
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
