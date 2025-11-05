"""MiniMax-M2 Proxy - FastAPI application

Dual-API proxy supporting both OpenAI and Anthropic formats
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import TabbyClient
from .config import settings
from .models import (
    AnthropicChatRequest,
    OpenAIChatRequest,
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
)
from formatters.anthropic import AnthropicFormatter
from formatters.openai import OpenAIFormatter
from parsers.reasoning import ensure_think_wrapped, split_think
from parsers.streaming import StreamingParser
from parsers.tools import parse_tool_calls
from .session_store import RepairResult, session_store


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


def require_auth(raw_request: Request) -> None:
    """Enforce bearer auth when configured."""
    if not settings.auth_api_key:
        return

    auth_header = raw_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = auth_header.split(" ", 1)[1].strip()
    if token != settings.auth_api_key:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def extract_session_id(raw_request: Request, extra_body: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract session identifier from headers, query params, or request body."""
    header_session = raw_request.headers.get("X-Session-Id")
    if header_session:
        return header_session.strip()

    query_session = raw_request.query_params.get("conversation_id")
    if query_session:
        return query_session.strip()

    if extra_body and isinstance(extra_body, dict):
        body_session = extra_body.get("conversation_id")
        if isinstance(body_session, str) and body_session.strip():
            return body_session.strip()

    return None


def normalize_openai_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize assistant messages so Tabby receives inline <think> content."""
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        msg_copy = dict(message)
        if msg_copy.get("role") == "assistant":
            reasoning_details = msg_copy.pop("reasoning_details", None)
            reasoning_text = ""
            if isinstance(reasoning_details, list):
                for detail in reasoning_details:
                    if isinstance(detail, dict):
                        reasoning_text += str(detail.get("text", ""))

            content = msg_copy.get("content") or ""
            if reasoning_text:
                reason_block = f"<think>{reasoning_text}</think>"
                if content and not content.startswith("\n"):
                    reason_block = f"{reason_block}\n"
                msg_copy["content"] = reason_block + content
            elif content and "</think>" in content:
                msg_copy["content"] = ensure_think_wrapped(content)

        normalized.append(msg_copy)
    return normalized


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
async def openai_chat_completions(chat_request: OpenAIChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint"""

    try:
        require_auth(raw_request)
        session_id = extract_session_id(raw_request, chat_request.extra_body)

        if chat_request.n is not None and chat_request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported")

        if chat_request.stream:
            return StreamingResponse(
                stream_openai_response(chat_request, session_id),
                media_type="text/event-stream"
            )
        else:
            return await complete_openai_response(chat_request, session_id)

    except Exception as e:
        logger.error(f"Error in OpenAI endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=openai_formatter.format_error(str(e))
        )


async def complete_openai_response(chat_request: OpenAIChatRequest, session_id: Optional[str]) -> dict:
    """Handle non-streaming OpenAI request"""

    # Convert messages to dict
    messages = [msg.model_dump(exclude_none=True) for msg in chat_request.messages]

    repair_result: RepairResult = session_store.inject_or_repair(
        messages,
        session_id,
        require_session=settings.require_session_for_repair,
    )
    if repair_result.repaired:
        logger.info(
            "Session history repaired (OpenAI)",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )
    elif repair_result.skip_reason:
        logger.debug(
            "Session repair skipped",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )

    messages = repair_result.messages
    normalized_messages = normalize_openai_history(messages)

    # Convert tools if present
    tools = None
    if chat_request.tools:
        tools = [tool.model_dump(exclude_none=True) for tool in chat_request.tools]

    # Call TabbyAPI
    banned_strings = settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None
    logger.info(f"Calling TabbyAPI with banned_strings enabled: {settings.enable_chinese_char_blocking}, count: {len(banned_strings) if banned_strings else 0}")

    response = await tabby_client.chat_completion(
        messages=normalized_messages,
        model=chat_request.model,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        top_k=chat_request.top_k,
        stop=chat_request.stop,
        tools=tools,
        tool_choice=chat_request.tool_choice,
        add_generation_prompt=True,  # Required for <think> tags
        banned_strings=banned_strings
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
    reasoning_split = (
        settings.enable_reasoning_split
        and bool(chat_request.extra_body and chat_request.extra_body.get("reasoning_split"))
    )

    thinking_text = ""
    visible_content = content_payload

    if reasoning_split:
        wrapped_content = ensure_think_wrapped(raw_content)
        thinking_text, visible_content = split_think(wrapped_content)

    client_content = visible_content if reasoning_split else content_payload

    formatted = openai_formatter.format_complete_response(
        content=client_content,
        tool_calls=result["tool_calls"] if result["tools_called"] else None,
        model=chat_request.model,
        reasoning_text=thinking_text if reasoning_split else None,
    )

    if session_id:
        assistant_message = {
            "role": "assistant",
            "content": ensure_think_wrapped(raw_content),
        }
        if result["tools_called"]:
            assistant_message["tool_calls"] = result["tool_calls"]
        if reasoning_split and thinking_text:
            assistant_message["reasoning_details"] = [
                {"type": "chain_of_thought", "text": thinking_text}
            ]
        session_store.append_message(session_id, assistant_message)

    return formatted


async def stream_openai_response(chat_request: OpenAIChatRequest, session_id: Optional[str]) -> AsyncIterator[str]:
    """Handle streaming OpenAI request"""

    # Convert messages to dict
    messages = [msg.model_dump(exclude_none=True) for msg in chat_request.messages]

    repair_result: RepairResult = session_store.inject_or_repair(
        messages,
        session_id,
        require_session=settings.require_session_for_repair,
    )
    if repair_result.repaired:
        logger.info(
            "Session history repaired (OpenAI/stream)",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )
    elif repair_result.skip_reason:
        logger.debug(
            "Session repair skipped",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )

    messages = repair_result.messages
    normalized_messages = normalize_openai_history(messages)

    # Convert tools if present
    tools = None
    if chat_request.tools:
        tools = [tool.model_dump(exclude_none=True) for tool in chat_request.tools]

    # Initialize streaming parser
    streaming_parser = StreamingParser()
    streaming_parser.set_tools(tools)
    raw_segments: List[str] = []
    visible_segments: List[str] = []
    reasoning_segments: List[str] = []
    captured_tool_calls: Optional[List[Dict[str, Any]]] = None

    reasoning_split = (
        settings.enable_reasoning_split
        and bool(chat_request.extra_body and chat_request.extra_body.get("reasoning_split"))
    )

    try:
        # Stream from TabbyAPI
        async for chunk in tabby_client.extract_streaming_content(
            messages=normalized_messages,
            model=chat_request.model,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            stop=chat_request.stop,
            tools=tools,
            tool_choice=chat_request.tool_choice,
            add_generation_prompt=True,  # Required for <think> tags
            banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None
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
                        reasoning_delta = parsed.get("reasoning_delta")
                        if reasoning_delta:
                            reasoning_segments.append(reasoning_delta)

                        if parsed["type"] == "content":
                            raw_delta = parsed.get("raw_delta") or ""
                            visible_delta = parsed.get("content_delta") or ""
                            if raw_delta:
                                raw_segments.append(raw_delta)
                            if visible_delta:
                                visible_segments.append(visible_delta)

                            delta_for_client = raw_delta
                            if reasoning_split:
                                delta_for_client = visible_delta or ""

                            if delta_for_client or (reasoning_split and reasoning_delta):
                                yield openai_formatter.format_streaming_chunk(
                                    delta=delta_for_client or None,
                                    reasoning_delta=reasoning_delta if reasoning_split else None,
                                    model=chat_request.model,
                                )

                            if parsed.get("tool_calls"):
                                captured_tool_calls = parsed["tool_calls"]
                                for idx, tool_call in enumerate(parsed["tool_calls"]):
                                    for tool_chunk in openai_formatter.format_tool_call_stream(
                                        tool_call, idx, model=chat_request.model
                                    ):
                                        yield tool_chunk

                        elif parsed["type"] == "tool_calls":
                            captured_tool_calls = parsed["tool_calls"]
                            if reasoning_split and reasoning_delta:
                                yield openai_formatter.format_streaming_chunk(
                                    reasoning_delta=reasoning_delta,
                                    model=chat_request.model,
                                )
                            for idx, tool_call in enumerate(parsed["tool_calls"]):
                                for tool_chunk in openai_formatter.format_tool_call_stream(
                                    tool_call, idx, model=chat_request.model
                                ):
                                    yield tool_chunk

                        elif parsed["type"] == "reasoning":
                            if reasoning_split and reasoning_delta:
                                yield openai_formatter.format_streaming_chunk(
                                    reasoning_delta=reasoning_delta,
                                    model=chat_request.model,
                                )

                # Check for finish
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    pending_tail = streaming_parser.flush_pending()
                    if pending_tail:
                        raw_delta = pending_tail.get("raw_delta") or ""
                        visible_delta = pending_tail.get("content_delta") or ""
                        reasoning_delta = pending_tail.get("reasoning_delta")

                        if raw_delta:
                            raw_segments.append(raw_delta)
                        if visible_delta:
                            visible_segments.append(visible_delta)
                        if reasoning_delta:
                            reasoning_segments.append(reasoning_delta)

                        delta_for_client = raw_delta
                        if reasoning_split:
                            delta_for_client = visible_delta or ""

                        if delta_for_client or (reasoning_split and reasoning_delta):
                            yield openai_formatter.format_streaming_chunk(
                                delta=delta_for_client or None,
                                reasoning_delta=reasoning_delta if reasoning_split else None,
                                model=chat_request.model,
                            )
                    # Override finish_reason if tool calls were detected
                    if finish_reason == "stop" and streaming_parser.has_tool_calls():
                        finish_reason = "tool_calls"
                    yield openai_formatter.format_streaming_chunk(finish_reason=finish_reason, model=chat_request.model)

        # Send done
        pending_tail = streaming_parser.flush_pending()
        if pending_tail:
            raw_delta = pending_tail.get("raw_delta") or ""
            visible_delta = pending_tail.get("content_delta") or ""
            reasoning_delta = pending_tail.get("reasoning_delta")

            if raw_delta:
                raw_segments.append(raw_delta)
            if visible_delta:
                visible_segments.append(visible_delta)
            if reasoning_delta:
                reasoning_segments.append(reasoning_delta)

            delta_for_client = raw_delta
            if reasoning_split:
                delta_for_client = visible_delta or ""

            if delta_for_client or (reasoning_split and reasoning_delta):
                yield openai_formatter.format_streaming_chunk(
                    delta=delta_for_client or None,
                    reasoning_delta=reasoning_delta if reasoning_split else None,
                    model=chat_request.model,
                )

        final_raw_content = "".join(raw_segments) if raw_segments else streaming_parser.get_final_content()
        final_reasoning = "".join(reasoning_segments)
        captured_tool_calls = captured_tool_calls or streaming_parser.get_last_tool_calls()
        if session_id:
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": ensure_think_wrapped(final_raw_content),
            }
            if captured_tool_calls:
                assistant_message["tool_calls"] = captured_tool_calls
            if reasoning_split and final_reasoning:
                assistant_message["reasoning_details"] = [
                    {"type": "chain_of_thought", "text": final_reasoning}
                ]
            session_store.append_message(session_id, assistant_message)

        yield openai_formatter.format_streaming_done()

    except Exception as e:
        logger.error(f"Error in OpenAI streaming: {e}", exc_info=True)
        error_chunk = openai_formatter.format_error(str(e))
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ============================================================================
# Anthropic Endpoints
# ============================================================================

@app.post("/v1/messages")
async def anthropic_messages(anthropic_request: AnthropicChatRequest, raw_request: Request):
    """Anthropic-compatible messages endpoint"""

    try:
        require_auth(raw_request)
        session_id = extract_session_id(raw_request)

        if anthropic_request.stream:
            return StreamingResponse(
                stream_anthropic_response(anthropic_request, session_id),
                media_type="text/event-stream"
            )
        else:
            return await complete_anthropic_response(anthropic_request, session_id)

    except Exception as e:
        logger.error(f"Error in Anthropic endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=anthropic_formatter.format_error(str(e))
        )


async def complete_anthropic_response(anthropic_request: AnthropicChatRequest, session_id: Optional[str]) -> dict:
    """Handle non-streaming Anthropic request"""

    # Convert Anthropic format to OpenAI format
    openai_messages = anthropic_messages_to_openai(anthropic_request.messages)

    # Add system message if present
    if anthropic_request.system:
        system_content = anthropic_request.system if isinstance(anthropic_request.system, str) else str(anthropic_request.system)
        openai_messages.insert(0, {"role": "system", "content": system_content})

    repair_result: RepairResult = session_store.inject_or_repair(
        openai_messages,
        session_id,
        require_session=settings.require_session_for_repair,
    )
    if repair_result.repaired:
        logger.info(
            "Session history repaired (Anthropic)",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )
    elif repair_result.skip_reason:
        logger.debug(
            "Session repair skipped",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )

    openai_messages = repair_result.messages
    normalized_messages = normalize_openai_history(openai_messages)

    # Convert tools
    tools = anthropic_tools_to_openai(anthropic_request.tools)

    # Debug logging
    import pprint
    logger.debug(f"Converted {len(openai_messages)} messages to OpenAI format:")
    for i, msg in enumerate(openai_messages):
        logger.debug(f"  Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, has_tool_calls={bool(msg.get('tool_calls'))}")

    # Call TabbyAPI
    response = await tabby_client.chat_completion(
        messages=normalized_messages,
        model=anthropic_request.model,
        max_tokens=anthropic_request.max_tokens,
        temperature=anthropic_request.temperature,
        top_p=anthropic_request.top_p,
        top_k=anthropic_request.top_k,
        stop=anthropic_request.stop_sequences,
        tools=tools,
        tool_choice=anthropic_request.tool_choice,
        add_generation_prompt=True,  # Required for <think> tags
        banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None
    )

    if settings.log_raw_responses:
        logger.debug(f"Raw TabbyAPI response: {response}")

    raw_content = response["choices"][0]["message"].get("content", "")
    wrapped_raw_content = ensure_think_wrapped(raw_content)

    # Parse tool calls
    result = parse_tool_calls(wrapped_raw_content, tools)

    content_source = result["content"] if result["tools_called"] else wrapped_raw_content
    content_source = ensure_think_wrapped(content_source) if content_source else ""

    thinking_text = ""
    visible_text = content_source
    if settings.enable_anthropic_thinking_blocks:
        thinking_text, visible_text = split_think(content_source)
    else:
        visible_text = content_source

    # Format as Anthropic response
    formatted = anthropic_formatter.format_complete_response(
        content=visible_text,
        tool_calls=result["tool_calls"] if result["tools_called"] else None,
        model=anthropic_request.model,
        thinking_text=thinking_text if settings.enable_anthropic_thinking_blocks else None,
    )

    if session_id:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": wrapped_raw_content,
        }
        if result["tools_called"]:
            assistant_message["tool_calls"] = result["tool_calls"]
        session_store.append_message(session_id, assistant_message)

    return formatted


async def stream_anthropic_response(anthropic_request: AnthropicChatRequest, session_id: Optional[str]) -> AsyncIterator[str]:
    """Handle streaming Anthropic request"""

    # Convert Anthropic format to OpenAI format
    openai_messages = anthropic_messages_to_openai(anthropic_request.messages)

    # Add system message if present
    if anthropic_request.system:
        system_content = anthropic_request.system if isinstance(anthropic_request.system, str) else str(anthropic_request.system)
        openai_messages.insert(0, {"role": "system", "content": system_content})

    repair_result: RepairResult = session_store.inject_or_repair(
        openai_messages,
        session_id,
        require_session=settings.require_session_for_repair,
    )
    if repair_result.repaired:
        logger.info(
            "Session history repaired (Anthropic/stream)",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )
    elif repair_result.skip_reason:
        logger.debug(
            "Session repair skipped",
            extra={"session_id": session_id, **repair_result.to_log_dict()},
        )

    openai_messages = repair_result.messages
    normalized_messages = normalize_openai_history(openai_messages)

    # Convert tools
    tools = anthropic_tools_to_openai(anthropic_request.tools)

    # Initialize streaming parser
    streaming_parser = StreamingParser()
    streaming_parser.set_tools(tools)
    captured_tool_calls: Optional[list[Dict[str, Any]]] = None
    thinking_block_started = False

    try:
        # Send message_start
        yield anthropic_formatter.format_message_start(anthropic_request.model)

        # Start first content block
        content_block_index = 0
        content_block_started = False

        # Stream from TabbyAPI
        async for chunk in tabby_client.extract_streaming_content(
            messages=normalized_messages,
            model=anthropic_request.model,
            max_tokens=anthropic_request.max_tokens,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            stop=anthropic_request.stop_sequences,
            tools=tools,
            tool_choice=anthropic_request.tool_choice,
            add_generation_prompt=True,  # Required for <think> tags
            banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None
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
                        reasoning_delta = parsed.get("reasoning_delta")
                        if reasoning_delta and settings.enable_anthropic_thinking_blocks:
                            if not thinking_block_started:
                                yield anthropic_formatter.format_content_block_start(content_block_index, "thinking")
                                thinking_block_started = True
                            yield anthropic_formatter.format_content_block_delta(
                                content_block_index,
                                reasoning_delta,
                                delta_type="thinking_delta",
                            )

                        if parsed["type"] == "content":
                            # Start text block if not started
                            if not content_block_started:
                                if thinking_block_started and settings.enable_anthropic_thinking_blocks:
                                    yield anthropic_formatter.format_content_block_stop(content_block_index)
                                    content_block_index += 1
                                    thinking_block_started = False
                                yield anthropic_formatter.format_content_block_start(content_block_index, "text")
                                content_block_started = True

                            delta_payload = parsed.get("raw_delta") if not settings.enable_anthropic_thinking_blocks else parsed.get("content_delta")
                            if delta_payload:
                                yield anthropic_formatter.format_content_block_delta(
                                    content_block_index,
                                    delta_payload,
                                    delta_type="text_delta"
                                )

                            if parsed.get("tool_calls"):
                                captured_tool_calls = parsed["tool_calls"]
                                if content_block_started:
                                    yield anthropic_formatter.format_content_block_stop(content_block_index)
                                    content_block_index += 1
                                    content_block_started = False
                                if thinking_block_started:
                                    yield anthropic_formatter.format_content_block_stop(content_block_index)
                                    content_block_index += 1
                                    thinking_block_started = False
                                for tool_call in parsed["tool_calls"]:
                                    yield anthropic_formatter.format_tool_use_delta(content_block_index, tool_call)
                                    yield anthropic_formatter.format_content_block_stop(content_block_index)
                                    content_block_index += 1

                        elif parsed["type"] == "tool_calls":
                            # Close text block if open
                            if content_block_started:
                                yield anthropic_formatter.format_content_block_stop(content_block_index)
                                content_block_index += 1
                                content_block_started = False
                            if thinking_block_started:
                                yield anthropic_formatter.format_content_block_stop(content_block_index)
                                content_block_index += 1
                                thinking_block_started = False

                            # Send tool use blocks
                            captured_tool_calls = parsed["tool_calls"]
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
        if thinking_block_started:
            yield anthropic_formatter.format_content_block_stop(content_block_index)
        yield anthropic_formatter.format_message_stop()

        if session_id:
            final_content = ensure_think_wrapped(streaming_parser.get_final_content())
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": final_content}
            if captured_tool_calls:
                assistant_message["tool_calls"] = captured_tool_calls
            session_store.append_message(session_id, assistant_message)

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
