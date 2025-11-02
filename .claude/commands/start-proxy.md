Start the MiniMax-M2 proxy on port 8001 with streaming debug logging enabled.

Configuration:
- Port: 8001
- Backend: TabbyAPI on localhost:8000
- Streaming debug: /tmp/stream.log
- Config: .env in project root

Steps:
1. Kill any existing proxy process on port 8001
2. Ensure .env has ENABLE_STREAMING_DEBUG=true and STREAMING_DEBUG_PATH=/tmp/stream.log
3. Start proxy: `cd /home/ser/minimax-m2-proxy && source venv/bin/activate && nohup uvicorn proxy.main:app --host 0.0.0.0 --port 8001 > proxy.log 2>&1 &`
4. Verify proxy is running (show PID)
5. Check health endpoint: curl http://localhost:8001/health
6. Tail proxy.log to show startup messages
