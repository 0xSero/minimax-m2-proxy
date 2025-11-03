#!/bin/bash
pkill -9 -f "uvicorn proxy.main"
sleep 2
cd /home/ser/minimax-m2-proxy
source venv/bin/activate
nohup python -m uvicorn proxy.main:app --host 0.0.0.0 --port 8001 --reload > proxy_fresh.log 2>&1 &
sleep 5
echo "Proxy restarted"
curl -s http://localhost:8001/ | python3 -m json.tool
