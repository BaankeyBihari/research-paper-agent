#!/bin/sh
set -e

# Start the NOOA trace viewer in the background, bound to all interfaces so the
# published Docker port can reach it. If --host/--port aren't supported by the
# installed nooa-cli version, this will fail loudly in the container logs --
# check `nooa start-dev --help` inside the container as a fallback.
nooa start-dev --host 0.0.0.0 --port 5001 &
TRACE_VIEWER_PID=$!

sleep 2

echo "Seeding /data/papers with sample arXiv PDFs..."
python simulator.py

echo "Running ResearchAgent (model: ${ACTIVE_MODEL_SLUG:-nvidia/nemotron-3-nano-30b-a3b})..."
python agent.py

# Keep the container alive so the trace viewer stays reachable after the run.
wait "$TRACE_VIEWER_PID"
