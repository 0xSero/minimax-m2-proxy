"""MiniMax-M2 Proxy - FastAPI application

Dual-API proxy supporting both OpenAI and Anthropic formats
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from .config import settings
from .client import TabbyClient
from .models import (
    OpenAIChatRequest,
    AnthropicChatRequest,
    anthropic_tools_to_openai,
    anthropic_messages_to_openai
)
from parsers.tools import parse_tool_calls
from parsers.streaming import StreamingParser
from parsers.reasoning import ensure_think_wrapped
from formatters.openai import OpenAIFormatter
from formatters.anthropic import AnthropicFormatter


# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
stream_logger = logging.getLogger("minimax.streaming")

if settings.enable_streaming_debug:
    stream_logger.setLevel(logging.DEBUG)
    if settings.streaming_debug_path:
        # Avoid duplicate handlers when reload=True
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == settings.streaming_debug_path for h in stream_logger.handlers):
            handler = logging.FileHandler(settings.streaming_debug_path)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            stream_logger.addHandler(handler)
else:
    stream_logger.setLevel(logging.INFO)

# Global instances
tabby_client: TabbyClient = None
openai_formatter = OpenAIFormatter()
anthropic_formatter = AnthropicFormatter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global tabby_client

    # Startup
    logger.info(f"Starting MiniMax-M2 Proxy on {settings.host}:{settings.port}")
    logger.info(f"Backend TabbyAPI: {settings.tabby_url}")

    tabby_client = TabbyClient(settings.tabby_url, settings.tabby_timeout)

    # Check backend health
    if await tabby_client.health_check():
        logger.info("TabbyAPI backend is healthy")
    else:
        logger.warning("TabbyAPI backend health check failed")

    yield

    # Shutdown
    logger.info("Shutting down proxy")
    await tabby_client.close()


app = FastAPI(
    title="MiniMax-M2 Proxy",
    description="Dual-API proxy for MiniMax-M2 with OpenAI and Anthropic compatibility",
    version="0.1.0",
    lifespan=lifespan
)


# ============================================================================
# OpenAI Endpoints
# ============================================================================

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint"""

    try:
        if request.stream:
            return StreamingResponse(
                stream_openai_response(request),
                media_type="text/event-stream"
            )
        else:
            return await complete_openai_response(request)

    except Exception as e:
        logger.error(f"Error in OpenAI endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=openai_formatter.format_error(str(e))
        )


async def complete_openai_response(request: OpenAIChatRequest) -> dict:
    """Handle non-streaming OpenAI request"""

    # Convert messages to dict
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    # Convert tools if present
    tools = None
    if request.tools:
        tools = [tool.model_dump(exclude_none=True) for tool in request.tools]

    # Call TabbyAPI
    response = await tabby_client.chat_completion(
        messages=messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        tools=tools,
        tool_choice=request.tool_choice,
        add_generation_prompt=True  # Required for <think> tags
    )

    if settings.log_raw_responses:
        logger.debug(f"Raw TabbyAPI response: {response}")

    # Extract raw content (contains XML and <think> blocks)
    raw_content = response["choices"][0]["message"].get("content", "")

    # Ensure think tags are wrapped (TabbyAPI includes closing but not opening tag)
    raw_content = ensure_think_wrapped(raw_content)

    # Parse tool calls
    result = parse_tool_calls(raw_content, tools)

    # Use parsed content if tool calls were found, otherwise use raw content
    content_payload = result["content"] if result["tools_called"] else raw_content

    # Format response
    return openai_formatter.format_complete_response(
        content=content_payload,
        tool_calls=result["tool_calls"] if result["tools_called"] else None,
        model=request.model
    )


async def stream_openai_response(request: OpenAIChatRequest) -> AsyncIterator[str]:
    """Handle streaming OpenAI request"""

    # Convert messages to dict
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    # Convert tools if present
    tools = None
    if request.tools:
        tools = [tool.model_dump(exclude_none=True) for tool in request.tools]

    # Initialize streaming parser
    streaming_parser = StreamingParser()
    streaming_parser.set_tools(tools)

    try:
        # Stream from TabbyAPI
        async for chunk in tabby_client.extract_streaming_content(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            tools=tools,
            tool_choice=request.tool_choice,
            add_generation_prompt=True  # Required for <think> tags
        ):
            # Extract delta
            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content_delta = delta.get("content", "")

                if content_delta:
                    # Process with streaming parser to extract tool calls and preserve <think> blocks
                    # Note: Parser will handle prepending <think> tag when it detects </think>
                    parsed = streaming_parser.process_chunk(content_delta)

                    if parsed:
                        if parsed["type"] == "content":
                            # Send content delta (includes <think> blocks)
                            yield openai_formatter.format_streaming_chunk(delta=parsed["delta"], model=request.model)

                        elif parsed["type"] == "tool_calls":
                            # Send tool calls as compliant streaming deltas
                            for idx, tool_call in enumerate(parsed["tool_calls"]):
                                for tool_chunk in openai_formatter.format_tool_call_stream(tool_call, idx, model=request.model):
                                    yield tool_chunk

                # Check for finish
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    pending_tail = streaming_parser.flush_pending()
                    if pending_tail:
                        yield openai_formatter.format_streaming_chunk(delta=pending_tail, model=request.model)
                    # Override finish_reason if tool calls were detected
                    if finish_reason == "stop" and streaming_parser.has_tool_calls():
                        finish_reason = "tool_calls"
                    yield openai_formatter.format_streaming_chunk(finish_reason=finish_reason, model=request.model)

        # Send done
        pending_tail = streaming_parser.flush_pending()
        if pending_tail:
            yield openai_formatter.format_streaming_chunk(delta=pending_tail, model=request.model)
        yield openai_formatter.format_streaming_done()

    except Exception as e:
        logger.error(f"Error in OpenAI streaming: {e}", exc_info=True)
        error_chunk = openai_formatter.format_error(str(e))
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ============================================================================
# Anthropic Endpoints
# ============================================================================

@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicChatRequest):
    """Anthropic-compatible messages endpoint"""

    try:
        if request.stream:
            return StreamingResponse(
                stream_anthropic_response(request),
                media_type="text/event-stream"
            )
        else:
            return await complete_anthropic_response(request)

    except Exception as e:
        logger.error(f"Error in Anthropic endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=anthropic_formatter.format_error(str(e))
        )


async def complete_anthropic_response(request: AnthropicChatRequest) -> dict:
    """Handle non-streaming Anthropic request"""

    # Convert Anthropic format to OpenAI format
    openai_messages = anthropic_messages_to_openai(request.messages)

    # Add system message if present
    if request.system:
        system_content = request.system if isinstance(request.system, str) else str(request.system)
        openai_messages.insert(0, {"role": "system", "content": system_content})

    # Convert tools
    tools = anthropic_tools_to_openai(request.tools)

    # Debug logging
    import pprint
    logger.debug(f"Converted {len(openai_messages)} messages to OpenAI format:")
    for i, msg in enumerate(openai_messages):
        logger.debug(f"  Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, has_tool_calls={bool(msg.get('tool_calls'))}")

    # Call TabbyAPI
    response = await tabby_client.chat_completion(
        messages=openai_messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=request.tool_choice,
        add_generation_prompt=True  # Required for <think> tags
    )

    if settings.log_raw_responses:
        logger.debug(f"Raw TabbyAPI response: {response}")

    # Extract raw content (contains XML and <think> blocks)
    raw_content = response["choices"][0]["message"].get("content", "")
    raw_content = ensure_think_wrapped(raw_content)

    # Parse tool calls
    result = parse_tool_calls(raw_content, tools)

    # Format as Anthropic response
    return anthropic_formatter.format_complete_response(
        content=result["content"] if result["tools_called"] else raw_content,
        tool_calls=result["tool_calls"] if result["tools_called"] else None,
        model=request.model
    )


async def stream_anthropic_response(request: AnthropicChatRequest) -> AsyncIterator[str]:
    """Handle streaming Anthropic request"""

    # Convert Anthropic format to OpenAI format
    openai_messages = anthropic_messages_to_openai(request.messages)

    # Add system message if present
    if request.system:
        system_content = request.system if isinstance(request.system, str) else str(request.system)
        openai_messages.insert(0, {"role": "system", "content": system_content})

    # Convert tools
    tools = anthropic_tools_to_openai(request.tools)

    # Initialize streaming parser
    streaming_parser = StreamingParser()
    streaming_parser.set_tools(tools)

    try:
        # Send message_start
        yield anthropic_formatter.format_message_start(request.model)

        # Start first content block
        content_block_index = 0
        content_block_started = False

        # Stream from TabbyAPI
        async for chunk in tabby_client.extract_streaming_content(
            messages=openai_messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop_sequences,
            tools=tools,
            tool_choice=request.tool_choice,
            add_generation_prompt=True  # Required for <think> tags
        ):
            # Extract delta
            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content_delta = delta.get("content", "")

                if content_delta:
                    # Process with streaming parser to preserve <think> blocks
                    # Note: Parser will handle prepending <think> tag when it detects </think>
                    parsed = streaming_parser.process_chunk(content_delta)

                    if parsed:
                        if parsed["type"] == "content":
                            # Start text block if not started
                            if not content_block_started:
                                yield anthropic_formatter.format_content_block_start(content_block_index, "text")
                                content_block_started = True

                            # Send content delta (includes <think> blocks)
                            yield anthropic_formatter.format_content_block_delta(
                                content_block_index,
                                parsed["delta"]
                            )

                        elif parsed["type"] == "tool_calls":
                            # Close text block if open
                            if content_block_started:
                                yield anthropic_formatter.format_content_block_stop(content_block_index)
                                content_block_index += 1
                                content_block_started = False

                            # Send tool use blocks
                            for tool_call in parsed["tool_calls"]:
                                yield anthropic_formatter.format_tool_use_delta(content_block_index, tool_call)
                                yield anthropic_formatter.format_content_block_stop(content_block_index)
                                content_block_index += 1

                # Check for finish
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    # Close last content block if open
                    if content_block_started:
                        yield anthropic_formatter.format_content_block_stop(content_block_index)

                    # Override finish_reason if tool calls were detected
                    if finish_reason == "stop" and streaming_parser.has_tool_calls():
                        finish_reason = "tool_calls"

                    # Map finish reason
                    stop_reason = "end_turn" if finish_reason == "stop" else finish_reason
                    if finish_reason == "tool_calls":
                        stop_reason = "tool_use"

                    yield anthropic_formatter.format_message_delta(stop_reason)

        # Send message_stop
        yield anthropic_formatter.format_message_stop()

    except Exception as e:
        logger.error(f"Error in Anthropic streaming: {e}", exc_info=True)
        error_response = anthropic_formatter.format_error(str(e))
        yield f"data: {json.dumps(error_response)}\n\n"


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    backend_healthy = await tabby_client.health_check()

    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend": settings.tabby_url,
        "backend_healthy": backend_healthy
    }


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "MiniMax-M2 Proxy",
        "version": "0.2.0-simplified",
        "code_version": "think_tags_preserved",
        "endpoints": {
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "models": "/v1/models",
            "health": "/health"
        },
        "backend": settings.tabby_url
    }


# ============================================================================
# Pass-through Endpoints (no parsing needed)
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """Pass through to TabbyAPI /v1/models endpoint"""
    try:
        response = await tabby_client.client.get(f"{settings.tabby_url}/v1/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error in /v1/models: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "api_error"}}
        )


@app.get("/v1/model")
async def get_model():
    """Pass through to TabbyAPI /v1/model endpoint (single model info)"""
    try:
        response = await tabby_client.client.get(f"{settings.tabby_url}/v1/model")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error in /v1/model: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "api_error"}}
        )
