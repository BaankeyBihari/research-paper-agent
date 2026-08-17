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

# Trace viewer (every LLM call / method invocation is auto-traced)
# http://localhost:5001 once the container is running

# Exercise the agent's deterministic paths without a live API key/network,
# e.g. to check SQLite dedup logic after code changes:
docker run --rm --entrypoint python research-paper-agent -c "
import asyncio
from agent import ResearchAgent
asyncio.run(...)
"
```

Switch models via `ACTIVE_MODEL_SLUG` (any OpenRouter slug), e.g.
`ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build`.

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
`wait`s on the trace-viewer process so the container stays up and the viewer stays reachable after
the run finishes. `nooa start-dev` already binds `0.0.0.0:5001` by default (verified) — the explicit
`--host`/`--port` flags in `entrypoint.sh` are redundant but harmless.

**Config surface**: `PAPERS_DIR` (default `/data/papers`), `ACTIVE_MODEL_SLUG`, `OPENROUTER_API_KEY`
— all read from the environment, set via `docker-compose.yml` from `.env`.

## Verified live

The full pipeline (build → arXiv download → `pypdf` extract → live OpenRouter call via
`nvidia/nemotron-3-nano-30b-a3b` → SQLite persist → trace viewer at `localhost:5001`) has been run
end-to-end. Note: `nooa[cli]` alone does **not** include the trace viewer — `requirements.txt` needs
`nooa[cli,viewer]`, otherwise `nooa start-dev` exits immediately with "viewer dependencies are not
installed" and, because `entrypoint.sh` does `wait "$TRACE_VIEWER_PID"`, the whole container exits
once `agent.py` finishes. See `README.md`'s "Verified" section for details, including a real
prompt-quality issue seen on `nemotron-3-nano-30b-a3b` (titles picking up author lists, objectives
copied verbatim) that the `summarize_paper` docstring now explicitly guards against.
