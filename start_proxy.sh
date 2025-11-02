#!/bin/bash
# Start MiniMax-M2 Proxy

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start proxy
echo "Starting MiniMax-M2 Proxy on port 8001..."
echo "Backend: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn proxy.main:app --host 0.0.0.0 --port 8001
