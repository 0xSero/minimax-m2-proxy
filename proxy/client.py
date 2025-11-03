"""TabbyAPI client for backend communication"""

import httpx
import json
from typing import Dict, Any, AsyncIterator, List


class TabbyClient:
    """Async HTTP client for TabbyAPI backend"""

    def __init__(self, base_url: str, timeout: int = 300):
        """
        Initialize TabbyAPI client.

        Args:
            base_url: TabbyAPI base URL (e.g., http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "minimax-m2",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call TabbyAPI /v1/chat/completions endpoint (non-streaming).

        Args:
            messages: List of message dicts with role and content
            model: Model name
            **kwargs: Additional OpenAI-compatible parameters

        Returns:
            Complete response dict
        """
        # Filter out None and empty values to avoid TabbyAPI bugs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != []}

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **filtered_kwargs
        }

        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str = "minimax-m2",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Call TabbyAPI /v1/chat/completions endpoint (streaming).

        Args:
            messages: List of message dicts
            model: Model name
            **kwargs: Additional parameters

        Yields:
            SSE data lines (already prefixed with "data: ")
        """
        # Filter out None and empty values to avoid TabbyAPI bugs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != []}

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **filtered_kwargs
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line

    async def extract_streaming_content(
        self,
        messages: List[Dict[str, Any]],
        model: str = "minimax-m2",
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completions and extract content deltas.

        Yields parsed chunk dicts.
        """
        async for line in self.chat_completion_stream(messages, model, **kwargs):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix

                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        yield chunk
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        """Check if TabbyAPI is healthy"""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
