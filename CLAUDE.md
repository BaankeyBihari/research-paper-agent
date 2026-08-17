# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Dockerized research assistant built on NVIDIA's [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
(Object-Oriented Agents) framework. `ResearchAgent` scans `/data/papers` for PDFs, extracts each
paper's Main Objective / Key Methodology / Resulting Metrics via an LLM call routed through
OpenRouter, and records results in a local SQLite file so re-runs (even under a different model)
don't reprocess the same paper.

## Commands

There is no test suite, linter, or build step beyond the Docker image itself.

```bash
# Sanity-check Python syntax (no venv/deps needed)
python -m py_compile agent.py simulator.py

# Build the image
docker compose build          # or: docker build -t research-paper-agent .

# Run (needs OPENROUTER_API_KEY in .env)
cp .env.example .env          # then edit .env
docker compose up --build

# Trace viewer (every LLM call / method invocation is auto-traced): its API
# 403s over a published Docker port (loopback-only auth check, defeated by
# Docker's NAT), and nooa can't run natively on Windows (imports Unix-only
# fcntl). Open the repo in VS Code's Dev Containers instead -- its port
# forwarding tunnels from inside the container's netns, so it reads as
# genuine loopback traffic. .devcontainer/devcontainer.json forwards :5001.
# Fallback that always works regardless of viewer access: query the SQLite
# table directly (see "Query processed papers" below).

# Exercise the agent's deterministic paths without a live API key/network,
# e.g. to check SQLite dedup logic after code changes:
docker run --rm --entrypoint python research-paper-agent -c "
import asyncio
from agent import ResearchAgent
asyncio.run(...)
"

# Query processed papers (always works, unlike the trace viewer over a
# published port -- see the note above):
docker compose exec research-agent python -c "
import sqlite3, json
with sqlite3.connect('/data/papers/.agent_state.sqlite3') as conn:
    conn.row_factory = sqlite3.Row
    for r in conn.execute('SELECT * FROM processed_papers ORDER BY processed_at'):
        print(json.dumps(dict(r), indent=2))
"

# Semantic search over processed papers (deterministic, no LLM/network call --
# see "State/memory" below):
docker compose exec research-agent python -c "
from agent import ResearchAgent
print(ResearchAgent().find_similar_papers('transformer attention mechanism'))
"

# Compare cheaper models against a reference model on the same papers
# (costs real money -- openai/gpt-4o-mini is paid; see README):
docker compose exec research-agent python compare_models.py
```

Switch models via `ACTIVE_MODEL_SLUG` (any OpenRouter slug), e.g.
`ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build`.

## Remote

https://github.com/BaankeyBihari/research-paper-agent — `origin` is HTTPS, not SSH. The SSH key on
this machine isn't registered with GitHub (`git@github.com: Permission denied (publickey)`), so
`git push` over SSH fails here; HTTPS works via the stored `gh`/credential-manager auth.

## Workflow

Track remaining/planned work as GitHub issues on this repo (`gh issue create` / `gh issue list`)
rather than as TODOs in code or docs — that's the source of truth for what's left to do and its
status. Make changes via a feature branch + PR (`gh pr create`), not direct commits to `main`:

```bash
git checkout -b <short-descriptive-branch-name>
# ...make changes, commit...
git push -u origin <branch-name>
gh pr create --fill
```

Reference the relevant issue number in the PR description (`Closes #N`) so it closes automatically
on merge.

## Architecture

**`agent.py`** is the entire application. `ResearchAgent(Agent, llm=llm)` follows NOOA's core
pattern: fields are state, methods are capabilities, and a method whose body is literally `...` is
an *agentic* method — the docstring becomes the prompt and NOOA's runtime (not this code) drives the
LLM call. Everything else is a normal deterministic Python method:

- `summarize_paper(self, text: str) -> PaperSummary` — the one agentic method. `PaperSummary` is a
  Pydantic model (`title`, `main_objective`, `key_methodology`, `resulting_metrics`); NOOA validates
  the LLM's output against it directly, so there is no manual JSON parsing here.
- `get_local_papers`, `lookup_paper`, `_record_summary`, `_record_memory`, `find_similar_papers`,
  `_init_state_db`, `process_pending_papers` — deterministic. `_init_state_db` is called lazily
  (idempotent `CREATE TABLE IF NOT EXISTS`) from the methods that need it rather than from `__init__`,
  to avoid interfering with `Agent`'s own Pydantic-style field construction.
- The LLM client is built once at module load: `get_llm_client(f"openrouter/{MODEL_SLUG}")` from
  `nooa.unifiedllm.registry`, where `MODEL_SLUG` comes from `ACTIVE_MODEL_SLUG` (env var, defaults to
  `nvidia/nemotron-3-nano-30b-a3b`). This is what makes model-switching a pure env-var change.

**State/memory**: two separate SQLite files, both written directly to the `/data/papers` volume.
Exact-match dedup and history use a plain `sqlite3` table (`processed_papers`, `.agent_state.sqlite3`)
— deliberately not `nooa-memory`'s own `MemoryManager`/`MemoryToolsMixin`, since that surface is built
for an LLM agent that autonomously authors/curates its own memory via tools during reasoning
(reflection, decay/forgetting, spontaneous per-turn context injection) — none of which fits
`process_pending_papers`'s fully deterministic orchestration loop. Semantic recall ("find papers
similar to X") instead uses `nooa_memory`'s lower-level primitives directly: `_record_memory` embeds
each `PaperSummary` with `get_embedder(EmbeddingConfig())` (`backend="hashing"` — deterministic,
offline, no LLM/network call) and writes it to a `nooa_memory.MemoryStore` (`.agent_memory.sqlite3`),
keyed by filename so re-processing a paper overwrites its memory the same way it overwrites its
`processed_papers` row; `find_similar_papers` embeds a query the same way and calls `store.knn(...)`.
Because both SQLite files live in the same Docker volume as the PDFs, history survives across
container restarts and model switches — this is what makes it possible to ask "did model A already
read this paper" while model B is active.

**`simulator.py`** is standalone and has no dependency on `agent.py`. It hits the live arXiv Atom API
(`http://export.arxiv.org/api/query`) directly with `requests` + stdlib `xml.etree.ElementTree`
(no arXiv client library), picks a random category/offset, and downloads PDFs into `PAPERS_DIR`,
skipping ones that already exist on disk.

**Runtime flow** (`entrypoint.sh`, single container): starts `nooa start-dev` (trace viewer) in the
background, waits 2s, runs `simulator.py` to seed papers, runs `agent.py` to process them, then
`wait`s on the trace-viewer process so the container stays up and trace ingestion keeps working
across reruns. `nooa start-dev` binds `0.0.0.0:5001` by default (verified).

**`compare_models.py`** is a separate, standalone script (not run by `entrypoint.sh`) for comparing
model quality/latency against a reference model. It runs on NOOA's real `eval_pipeline` package —
that package isn't on PyPI, but contrary to an earlier assumption here, it does *not* need a full
monorepo clone + `uv sync`: `util/eval_pipeline/pyproject.toml` is a self-contained `hatchling`
package with plain PyPI dependencies (its `[tool.uv.sources] nooa = {workspace = true}` line is
`uv`-only metadata that `pip` ignores), so it installs straight from a git-subdirectory URL, pinned
to a commit SHA in `requirements.txt` (`eval_pipeline @
git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@<commit>#subdirectory=util/eval_pipeline`) —
this is also why the `Dockerfile` installs `git`. `ResearchAgent` (from `agent.py`, unmodified) is
handed to `eval_pipeline.Evaluator` directly as `agent_class`, since `nooa.agent.Agent.__init__`
supports a per-instance `llm=` override (instance → class → parent resolution) — `eval_pipeline`'s
own `agent_from_spec` relies on exactly that (`cls(llm=client)`). The script still runs the reference
model synchronously per paper first (to supply `expected` values `Evaluator.add_test` needs up
front), then scores candidates with a custom `PaperSimilarityScorer` (`eval_pipeline`'s duck-typed
`Scorer.score(ctx) -> ScoreResult` interface) porting the same per-field `difflib` text-similarity
logic the script used before this was wired up to `eval_pipeline` — stdlib only, no extra API calls,
no LLM-judge self-bias risk. Results land in a `.noo-eval.jsonl` file under
`/data/papers/eval_results/` — a separate, offline copy, *not* what the trace viewer's Evaluations
tab actually reads. That tab is backed by the viewer's OTLP store, and `eval_pipeline` posts eval
spans to it live while `Evaluator.run()` executes (same `localhost:5001/v1/traces` endpoint
`entrypoint.sh` already runs `nooa start-dev` on), so the tab only populates if the viewer was
already reachable during the run — a `.jsonl` file produced with no viewer connected won't show up
there later. `REFERENCE_MODEL_SLUG`/`CANDIDATE_MODEL_SLUGS`/`COMPARE_NUM_PAPERS` are forwarded
into the container via `docker-compose.yml`'s `environment:` block, alongside `ACTIVE_MODEL_SLUG` —
without that, `.env` overrides for `compare_models.py` are silently ignored (a real bug found during
live verification; see "Verified live" below). The script also works around a genuine `eval_pipeline`
bug: `Evaluator.run()` (pinned commit `8622fc4`) raises a `pydantic.ValidationError` when using the
plain-Python `Evaluator(models={...})` API documented in its own README, because `_model_metadata`
(needed to build the run's metadata line) is only ever populated by the YAML/`from_config` path —
`compare_models.py` pre-populates `evaluator._model_metadata` directly before calling `run()` as a
workaround; filed upstream as
[NVIDIA-NeMo/labs-OO-Agents#152](https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/152) —
drop the workaround once a fix lands.

**Config surface**: `PAPERS_DIR` (default `/data/papers`), `ACTIVE_MODEL_SLUG`, `OPENROUTER_API_KEY`
— all read from the environment, set via `docker-compose.yml` from `.env`.

## Verified live

The full pipeline (build → arXiv download → `pypdf` extract → live OpenRouter call →
SQLite persist → trace viewer via VS Code Dev Containers) has been run end-to-end, including
`compare_models.py` against `eval_pipeline` with a paid `openai/gpt-4o-mini` reference. Note:
`nooa[cli]` alone does **not** include the trace viewer — `requirements.txt` needs
`nooa[cli,viewer]`, otherwise `nooa start-dev` exits immediately with "viewer dependencies are not
installed" and, because `entrypoint.sh` does `wait "$TRACE_VIEWER_PID"`, the whole container exits
once `agent.py` finishes. See `README.md`'s "Verified" section for more, including a real
prompt-quality issue seen on `nemotron-3-nano-30b-a3b` (titles picking up author lists, objectives
copied verbatim) that the `summarize_paper` docstring now explicitly guards against. The Evaluations
tab is populated by `compare_models.py`'s real `eval_pipeline` runs. The Memory tab stays empty even
though `nooa_memory.MemoryStore` is now wired up (see "State/memory" above) — confirmed from the
viewer's own source (`nooa/viewer/memory_routes.py`): it can only open a store under its own process
cwd (`/app` in this container, since `entrypoint.sh` never `cd`s from the Dockerfile's `WORKDIR`), and
auto-discovery narrows that further to `/app/.nooa/memory/*.sqlite`. `.agent_memory.sqlite3`
deliberately lives under `/data/papers` instead, matching `processed_papers`'s persistence story
(one Docker volume, survives restarts, no new `docker-compose.yml` volume needed) — a real,
considered trade-off, not an oversight. `find_similar_papers()` itself works fully regardless of the
viewer; only the visual tab is affected.

`_record_memory`/`find_similar_papers` confirmed live (`docker compose build` + `docker compose run`):
`nooa-memory` installs and imports cleanly, `.agent_memory.sqlite3` is created under the mounted
`/data/papers` volume and survives across separate container runs (same volume, new container), and a
query correctly ranks a topically-related fake paper above an unrelated one — no `OPENROUTER_API_KEY`
or network call needed, since the default `hashing` embedder is fully offline.

`meta-llama/llama-3.1-70b-instruct` was `compare_models.py`'s original reference-model default, but
live verification found it reliably fails to return structured output under NOOA's tool-calling
strategy via OpenRouter, regardless of paper content (reproduces on both severely corrupted and
clean PDF text) — a real model/provider incompatibility, not a bug in this repo's code. Default
switched to `openai/gpt-4o-mini`. Separately, OpenRouter has discontinued that model's free tier
entirely (`:free` now 404s), so the old cost-saving suffix no longer applies to it either way.
