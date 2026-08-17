#!/bin/sh
set -e

# Start the NOOA trace viewer in the background. This is NOT meant to be
# reached from outside the container -- it 403s anything that doesn't look
# like loopback traffic once bound to 0.0.0.0, which Docker's NAT always
# defeats from the host side. It only needs to be reachable by agent.py over
# the container's own localhost, which this satisfies. To actually view
# traces, run `nooa start-dev --db ./traces/traces.db` natively on the host
# against the bind-mounted trace DB (see README.md).
nooa start-dev &
TRACE_VIEWER_PID=$!

sleep 2

echo "Seeding /data/papers with sample arXiv PDFs..."
python simulator.py

echo "Running ResearchAgent (model: ${ACTIVE_MODEL_SLUG:-nvidia/nemotron-3-nano-30b-a3b})..."
python agent.py

# Keep the container alive so trace ingestion keeps working across reruns.
wait "$TRACE_VIEWER_PID"
