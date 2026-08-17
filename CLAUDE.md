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

# Compare cheaper models against a reference model on the same papers
# (costs real money -- meta-llama/llama-3.1-70b-instruct is paid; see README):
docker compose exec research-agent python compare_models.py
```

Switch models via `ACTIVE_MODEL_SLUG` (any OpenRouter slug), e.g.
`ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build`.

## Remote

https://github.com/BaankeyBihari/research-paper-agent — `origin` is HTTPS, not SSH. The SSH key on
this machine isn't registered with GitHub (`git@github.com: Permission denied (publickey)`), so
`git push` over SSH fails here; HTTPS works via the stored `gh`/credential-manager auth.

## Architecture

**`agent.py`** is the entire application. `ResearchAgent(Agent, llm=llm)` follows NOOA's core
pattern: fields are state, methods are capabilities, and a method whose body is literally `...` is
an *agentic* method — the docstring becomes the prompt and NOOA's runtime (not this code) drives the
LLM call. Everything else is a normal deterministic Python method:

- `summarize_paper(self, text: str) -> PaperSummary` — the one agentic method. `PaperSummary` is a
  Pydantic model (`title`, `main_objective`, `key_methodology`, `resulting_metrics`); NOOA validates
  the LLM's output against it directly, so there is no manual JSON parsing here.
- `get_local_papers`, `lookup_paper`, `_record_summary`, `_init_state_db`, `process_pending_papers`
  — deterministic. `_init_state_db` is called lazily (idempotent `CREATE TABLE IF NOT EXISTS`) from
  the methods that need it rather than from `__init__`, to avoid interfering with `Agent`'s own
  Pydantic-style field construction.
- The LLM client is built once at module load: `get_llm_client(f"openrouter/{MODEL_SLUG}")` from
  `nooa.unifiedllm.registry`, where `MODEL_SLUG` comes from `ACTIVE_MODEL_SLUG` (env var, defaults to
  `nvidia/nemotron-3-nano-30b-a3b`). This is what makes model-switching a pure env-var change.

**State/memory**: dedup and history use a plain `sqlite3` table (`processed_papers`) written directly
to the `/data/papers` volume — *not* the `nooa-memory` package (which is present in
`requirements.txt` but unused; it's a vector-search semantic-recall subsystem, not an exact-match
dedup store, and wasn't documented well enough to wire up with confidence). Because the SQLite file
lives in the same Docker volume as the PDFs, history survives across container restarts and model
switches — this is what makes it possible to ask "did model A already read this paper" while model B
is active.

**`simulator.py`** is standalone and has no dependency on `agent.py`. It hits the live arXiv Atom API
(`http://export.arxiv.org/api/query`) directly with `requests` + stdlib `xml.etree.ElementTree`
(no arXiv client library), picks a random category/offset, and downloads PDFs into `PAPERS_DIR`,
skipping ones that already exist on disk.

**Runtime flow** (`entrypoint.sh`, single container): starts `nooa start-dev` (trace viewer) in the
background, waits 2s, runs `simulator.py` to seed papers, runs `agent.py` to process them, then
`wait`s on the trace-viewer process so the container stays up and trace ingestion keeps working
across reruns. `nooa start-dev` binds `0.0.0.0:5001` by default (verified).

**`compare_models.py`** is a separate, standalone script (not run by `entrypoint.sh`) for comparing
model quality/latency against a reference model, using plain `difflib` text similarity instead of
NOOA's real `nooa eval` (which needs the full monorepo + `uv sync`, not just `pip install nooa`, and
isn't wired up here). It defines its own throwaway `Agent` subclasses per model slug via
`make_agent()`, one per model, since `Agent`'s `llm=` binding happens at class-definition time and
can't be swapped per-instance — `ResearchAgent`'s single class-level `llm` is why `agent.py` only
ever runs one model per process. Its `summarize_paper` docstring is a deliberate literal copy of `agent.py`'s, not a shared constant —
simpler to keep in sync by hand for one short prompt than to add an indirection for it. Update both
together if the prompt changes.

**Config surface**: `PAPERS_DIR` (default `/data/papers`), `ACTIVE_MODEL_SLUG`, `OPENROUTER_API_KEY`
— all read from the environment, set via `docker-compose.yml` from `.env`.

## Verified live

The full pipeline (build → arXiv download → `pypdf` extract → live OpenRouter call →
SQLite persist → trace viewer via VS Code Dev Containers) has been run end-to-end, including
`compare_models.py` against the paid `meta-llama/llama-3.1-70b-instruct` reference. Note: `nooa[cli]`
alone does **not** include the trace viewer — `requirements.txt` needs `nooa[cli,viewer]`, otherwise
`nooa start-dev` exits immediately with "viewer dependencies are not installed" and, because
`entrypoint.sh` does `wait "$TRACE_VIEWER_PID"`, the whole container exits once `agent.py` finishes.
See `README.md`'s "Verified" section for more, including a real prompt-quality issue seen on
`nemotron-3-nano-30b-a3b` (titles picking up author lists, objectives copied verbatim) that the
`summarize_paper` docstring now explicitly guards against, and the Evaluations/Memory tabs being
empty by design (not wired up — see `compare_models.py` and the SQLite-vs-`nooa-memory` note above).
