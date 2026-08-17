#!/bin/sh
set -e

# Start the NOOA trace viewer in the background. This is NOT meant to be
# reached from outside the container -- it 403s anything that doesn't look
# like loopback traffic once bound to 0.0.0.0, which Docker's NAT always
# defeats from the host side, and nooa can't run natively on Windows either
# (imports Unix-only fcntl). It only needs to be reachable by agent.py over
# the container's own localhost, which this satisfies. To actually view
# traces, open this repo in VS Code's Dev Containers (see README.md) --
# its port forwarding tunnels from inside the container's own netns, so
# the connection reads as genuine loopback traffic and the viewer just works.
nooa start-dev &
TRACE_VIEWER_PID=$!

sleep 2

echo "Seeding /data/papers with sample arXiv PDFs..."
python simulator.py

echo "Running ResearchAgent (model: ${ACTIVE_MODEL_SLUG:-nvidia/nemotron-3-nano-30b-a3b})..."
python agent.py

# Keep the container alive so trace ingestion keeps working across reruns.
wait "$TRACE_VIEWER_PID"
