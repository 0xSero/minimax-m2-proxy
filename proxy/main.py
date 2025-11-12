"""MiniMax-M2 Proxy - FastAPI application

Dual-API proxy supporting both OpenAI and Anthropic formats
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .client import TabbyClient
from .config import settings
from .models import (
    AnthropicChatRequest,
    AnthropicMessage,
    OpenAIChatRequest,
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    anthropic_tool_choice_to_openai,
)
from formatters.anthropic import AnthropicFormatter
from formatters.openai import OpenAIFormatter
from parsers.reasoning import ensure_think_wrapped, split_think
from parsers.streaming import StreamingParser
from parsers.tools import parse_tool_calls, tool_calls_to_minimax_xml
from .session_store import RepairResult, session_store


# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
stream_logger = logging.getLogger("minimax.streaming")


def is_minimax_model(model_name: str) -> bool:
    """Check if model uses MiniMax XML format"""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in settings.minimax_model_patterns)


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

            tool_calls = msg_copy.get("tool_calls")
            if tool_calls:
                xml_block = tool_calls_to_minimax_xml(tool_calls)
                if xml_block:
                    updated_content = msg_copy.get("content") or ""
                    if "</think>" in updated_content and "<think>" not in updated_content:
                        updated_content = ensure_think_wrapped(updated_content)

                    if xml_block not in updated_content:
                        stripped = updated_content.rstrip()
                        if stripped:
                            msg_copy["content"] = f"{stripped}\n\n{xml_block}"
                        else:
                            msg_copy["content"] = xml_block
                    else:
                        msg_copy["content"] = updated_content

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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

    # Check if this is a MiniMax model that needs XML parsing
    use_minimax_parsing = is_minimax_model(chat_request.model)

    if not use_minimax_parsing:
        logger.info(f"Non-MiniMax model detected: {chat_request.model}, passing through without XML parsing")
        # Pass through directly to backend without any normalization or parsing
        tools = None
        if chat_request.tools:
            tools = [tool.model_dump(exclude_none=True) for tool in chat_request.tools]

        response = await tabby_client.chat_completion(
            messages=messages,
            model=chat_request.model,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            stop=chat_request.stop,
            tools=tools,
            tool_choice=chat_request.tool_choice,
        )
        return response

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

    message_payload = response["choices"][0]["message"]
    backend_content = message_payload.get("content", "") or ""
    reasoning_text = message_payload.get("reasoning_content") or ""
    backend_tool_calls = message_payload.get("tool_calls")

    sections: List[str] = []

    if reasoning_text:
        trimmed_reasoning = reasoning_text.rstrip()
        sections.append(f"<think>{trimmed_reasoning}</think>")

    if backend_content.strip():
        sections.append(backend_content)

    if backend_tool_calls:
        xml_block = tool_calls_to_minimax_xml(backend_tool_calls)
        if xml_block:
            sections.append(xml_block)

    raw_content = "\n\n".join(section for section in sections if section).strip()
    if not raw_content:
        raw_content = backend_content

    raw_content = ensure_think_wrapped(raw_content)

    # Parse tool calls from content as fallback when backend omits structured payload
    result = parse_tool_calls(raw_content, tools)

    tool_calls = backend_tool_calls
    if not tool_calls and result["tools_called"]:
        tool_calls = result["tool_calls"]

    content_without_tool_blocks = raw_content
    if result["tools_called"] and result["content"]:
        content_without_tool_blocks = ensure_think_wrapped(result["content"])
    elif result["tools_called"] and not result["content"]:
        content_without_tool_blocks = ""

    content_payload = content_without_tool_blocks
    reasoning_split = (
        settings.enable_reasoning_split
        and bool(chat_request.extra_body and chat_request.extra_body.get("reasoning_split"))
    )

    thinking_text = ""

    if reasoning_split:
        wrapped_for_split = ensure_think_wrapped(content_without_tool_blocks)
        thinking_text, visible_content = split_think(wrapped_for_split)
        content_payload = visible_content
    else:
        thinking_text = reasoning_text.strip() if reasoning_text else ""

    client_content = content_payload

    formatted = openai_formatter.format_complete_response(
        content=client_content,
        tool_calls=tool_calls,
        model=chat_request.model,
        reasoning_text=thinking_text if reasoning_split else None,
    )

    if session_id:
        assistant_message = {
            "role": "assistant",
            "content": ensure_think_wrapped(raw_content),
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
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

    # Check if this is a MiniMax model that needs XML parsing
    use_minimax_parsing = is_minimax_model(chat_request.model)

    if not use_minimax_parsing:
        logger.info(f"Non-MiniMax model detected (streaming): {chat_request.model}, passing through without XML parsing")
        # Pass through directly to backend without any normalization or parsing
        tools = None
        if chat_request.tools:
            tools = [tool.model_dump(exclude_none=True) for tool in chat_request.tools]

        try:
            async for line in tabby_client.chat_completion_stream(
                messages=messages,
                model=chat_request.model,
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
                top_p=chat_request.top_p,
                top_k=chat_request.top_k,
                stop=chat_request.stop,
                tools=tools,
                tool_choice=chat_request.tool_choice,
            ):
                yield line + "\n"
        except Exception as e:
            logger.error(f"Error in OpenAI streaming (pass-through): {e}", exc_info=True)
            error_chunk = openai_formatter.format_error(str(e))
            yield f"data: {json.dumps(error_chunk)}\n\n"
        return

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

    reasoning_split = (
        settings.enable_reasoning_split
        and bool(chat_request.extra_body and chat_request.extra_body.get("reasoning_split"))
    )

    final_raw_content = ""
    final_reasoning_text = ""
    final_tool_calls: Optional[List[Dict[str, Any]]] = None

    async def prepend_chunk(first_chunk: Dict[str, Any], stream_gen: AsyncIterator[Dict[str, Any]]):
        yield first_chunk
        async for item in stream_gen:
            yield item

    async def structured_stream(chunk_iter: AsyncIterator[Dict[str, Any]]):
        nonlocal final_raw_content, final_reasoning_text, final_tool_calls

        raw_segments: List[str] = []
        reasoning_segments: List[str] = []
        tool_buffers: Dict[int, Dict[str, Any]] = {}
        think_started = False
        think_closed = False
        tool_xml_emitted = False
        finished = False

        def merge_tool_call_delta(delta_list: List[Dict[str, Any]]) -> None:
            for call in delta_list:
                idx = call.get("index", 0)
                entry = tool_buffers.setdefault(
                    idx,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if "id" in call and call["id"]:
                    entry["id"] = call["id"]
                if "type" in call and call["type"]:
                    entry["type"] = call["type"]
                function_payload = call.get("function", {})
                if function_payload:
                    if function_payload.get("name"):
                        entry["function"]["name"] = function_payload["name"]
                    if "arguments" in function_payload:
                        argument_text = function_payload.get("arguments") or ""
                        entry_args = entry["function"].get("arguments", "")
                        entry["function"]["arguments"] = entry_args + argument_text

        def finalize_tool_calls() -> List[Dict[str, Any]]:
            if not tool_buffers:
                return []
            ordered: List[Dict[str, Any]] = []
            for idx in sorted(tool_buffers.keys()):
                entry = tool_buffers[idx]
                function_payload = entry.get("function", {})
                ordered.append(
                    {
                        "id": entry.get("id"),
                        "type": entry.get("type") or "function",
                        "function": {
                            "name": function_payload.get("name", ""),
                            "arguments": function_payload.get("arguments", ""),
                        },
                    }
                )
            return ordered

        async for chunk in chunk_iter:
            if not chunk.get("choices"):
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            reasoning_delta = delta.get("reasoning_content")
            content_delta = delta.get("content")
            tool_delta = delta.get("tool_calls")
            finish_reason = choice.get("finish_reason")

            if reasoning_delta:
                addition = reasoning_delta
                if not think_started:
                    think_started = True
                    think_closed = False
                    addition = f"<think>{addition}"
                raw_segments.append(addition)
                reasoning_segments.append(reasoning_delta)
                if reasoning_split:
                    yield openai_formatter.format_streaming_chunk(
                        reasoning_delta=reasoning_delta,
                        model=chat_request.model,
                    )
                else:
                    yield openai_formatter.format_streaming_chunk(
                        delta=addition,
                        model=chat_request.model,
                    )

            if (not reasoning_delta) and think_started and not think_closed and (
                content_delta or tool_delta or finish_reason
            ):
                close_text = "</think>\n"
                raw_segments.append(close_text)
                if not reasoning_split:
                    yield openai_formatter.format_streaming_chunk(
                        delta=close_text,
                        model=chat_request.model,
                    )
                think_closed = True

            if content_delta:
                raw_segments.append(content_delta)
                if content_delta:
                    yield openai_formatter.format_streaming_chunk(
                        delta=content_delta,
                        model=chat_request.model,
                    )

            if tool_delta:
                merge_tool_call_delta(tool_delta)
                yield openai_formatter.format_streaming_chunk(
                    tool_calls=tool_delta,
                    model=chat_request.model,
                )

            if finish_reason:
                if think_started and not think_closed:
                    close_text = "</think>\n"
                    raw_segments.append(close_text)
                    if not reasoning_split:
                        yield openai_formatter.format_streaming_chunk(
                            delta=close_text,
                            model=chat_request.model,
                        )
                    think_closed = True

                final_tool_list = finalize_tool_calls()
                if final_tool_list and not tool_xml_emitted:
                    xml_block = tool_calls_to_minimax_xml(final_tool_list)
                    if xml_block:
                        if raw_segments and not raw_segments[-1].endswith("\n"):
                            raw_segments.append("\n")
                        raw_segments.append(xml_block)
                        tool_xml_emitted = True

                final_raw_content = "".join(raw_segments)
                final_reasoning_text = "".join(reasoning_segments)
                final_tool_calls = final_tool_list or None

                final_finish_reason = finish_reason
                if final_finish_reason == "stop" and final_tool_calls:
                    final_finish_reason = "tool_calls"

                yield openai_formatter.format_streaming_chunk(
                    finish_reason=final_finish_reason,
                    model=chat_request.model,
                )
                finished = True
                break

        if not finished:
            # No explicit finish_reason received; finalize with current buffers
            if think_started and not think_closed:
                raw_segments.append("</think>\n")
            final_raw_content = "".join(raw_segments)
            final_reasoning_text = "".join(reasoning_segments)
            tool_list = finalize_tool_calls()
            final_tool_calls = tool_list or None

        async for _ in chunk_iter:
            pass

    async def legacy_stream(chunk_iter: AsyncIterator[Dict[str, Any]]):
        nonlocal final_raw_content, final_reasoning_text, final_tool_calls

        streaming_parser = StreamingParser()
        streaming_parser.set_tools(tools)
        raw_segments: List[str] = []
        reasoning_segments: List[str] = []
        captured_tool_calls: Optional[List[Dict[str, Any]]] = None

        async for chunk in chunk_iter:
            if not chunk.get("choices"):
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            content_delta = delta.get("content", "")

            if content_delta:
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
                        delta_for_client = raw_delta if not reasoning_split else visible_delta
                        if delta_for_client or (reasoning_split and reasoning_delta):
                            yield openai_formatter.format_streaming_chunk(
                                delta=delta_for_client or None,
                                reasoning_delta=reasoning_delta if reasoning_split else None,
                                model=chat_request.model,
                            )

                    elif parsed["type"] == "tool_calls":
                        captured_tool_calls = parsed["tool_calls"]
                        raw_delta = parsed.get("raw_delta") or ""
                        if raw_delta:
                            raw_segments.append(raw_delta)
                            reasoning_delta = None
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

                    elif parsed["type"] == "reasoning" and reasoning_split and reasoning_delta:
                        yield openai_formatter.format_streaming_chunk(
                            reasoning_delta=reasoning_delta,
                            model=chat_request.model,
                        )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                pending_tail = streaming_parser.flush_pending()
                if pending_tail:
                    raw_delta = pending_tail.get("raw_delta") or ""
                    reasoning_delta = pending_tail.get("reasoning_delta")

                    if raw_delta:
                        raw_segments.append(raw_delta)
                    if reasoning_delta:
                        reasoning_segments.append(reasoning_delta)

                    sendable_raw_delta = raw_delta if raw_delta and "<minimax:tool_call>" not in raw_delta else None
                    if sendable_raw_delta or (reasoning_split and reasoning_delta):
                        yield openai_formatter.format_streaming_chunk(
                            delta=sendable_raw_delta,
                            reasoning_delta=reasoning_delta if reasoning_split else None,
                            model=chat_request.model,
                        )
                if finish_reason == "stop" and streaming_parser.has_tool_calls():
                    finish_reason = "tool_calls"
                yield openai_formatter.format_streaming_chunk(
                    finish_reason=finish_reason,
                    model=chat_request.model,
                )
                break

        pending_tail = streaming_parser.flush_pending()
        if pending_tail:
            raw_delta = pending_tail.get("raw_delta") or ""
            reasoning_delta = pending_tail.get("reasoning_delta")

            if raw_delta:
                raw_segments.append(raw_delta)
            if reasoning_delta:
                reasoning_segments.append(reasoning_delta)

            sendable_raw_delta = raw_delta if raw_delta and "<minimax:tool_call>" not in raw_delta else None
            if sendable_raw_delta or (reasoning_split and reasoning_delta):
                yield openai_formatter.format_streaming_chunk(
                    delta=sendable_raw_delta,
                    reasoning_delta=reasoning_delta if reasoning_split else None,
                    model=chat_request.model,
                )

        final_raw_content = "".join(raw_segments) if raw_segments else streaming_parser.get_final_content()
        final_reasoning_text = "".join(reasoning_segments)
        captured_tool_calls = captured_tool_calls or streaming_parser.get_last_tool_calls()
        final_tool_calls = captured_tool_calls

        async for _ in chunk_iter:
            pass

    try:
        stream_gen = tabby_client.extract_streaming_content(
            messages=normalized_messages,
            model=chat_request.model,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            stop=chat_request.stop,
            tools=tools,
            tool_choice=chat_request.tool_choice,
            add_generation_prompt=True,
            banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None,
        )

        try:
            first_chunk = await stream_gen.__anext__()
        except StopAsyncIteration:
            yield openai_formatter.format_streaming_done()
            return

        first_delta = first_chunk.get("choices", [{}])[0].get("delta", {})
        structured_mode = bool(first_delta.get("reasoning_content"))

        chunk_iter = prepend_chunk(first_chunk, stream_gen)

        if structured_mode:
            async for event in structured_stream(chunk_iter):
                yield event
        else:
            async for event in legacy_stream(chunk_iter):
                yield event

        if session_id:
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": ensure_think_wrapped(final_raw_content),
            }
            if final_tool_calls:
                assistant_message["tool_calls"] = final_tool_calls
            if reasoning_split and final_reasoning_text:
                assistant_message["reasoning_details"] = [
                    {"type": "chain_of_thought", "text": final_reasoning_text}
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

    # Check if this is a MiniMax model that needs XML parsing
    use_minimax_parsing = is_minimax_model(anthropic_request.model)

    # For MiniMax models, ensure max_tokens is high enough to avoid thinking-only responses
    effective_max_tokens = anthropic_request.max_tokens
    logger.info(f"Anthropic request - model: {anthropic_request.model}, max_tokens: {effective_max_tokens}, stream: {anthropic_request.stream}, use_minimax_parsing: {use_minimax_parsing}")
    if use_minimax_parsing and effective_max_tokens <= 8192:
        logger.info(f"Increasing max_tokens from {effective_max_tokens} to 32768 for MiniMax model")
        effective_max_tokens = 32768

    if not use_minimax_parsing:
        logger.info(f"Non-MiniMax model detected (Anthropic): {anthropic_request.model}, passing through without XML parsing")
        # Pass through to backend and format as Anthropic response
        tools = anthropic_tools_to_openai(anthropic_request.tools)
        tool_choice = anthropic_tool_choice_to_openai(anthropic_request.tool_choice)

        response = await tabby_client.chat_completion(
            messages=openai_messages,
            model=anthropic_request.model,
            max_tokens=effective_max_tokens,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            stop=anthropic_request.stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Convert OpenAI response to Anthropic format
        message_payload = response["choices"][0]["message"]
        content = message_payload.get("content", "") or ""
        tool_calls = message_payload.get("tool_calls")

        usage = response.get("usage", {"input_tokens": 0, "output_tokens": 0})
        if "prompt_tokens" in usage:
            usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }

        return anthropic_formatter.format_complete_response(
            content=content,
            tool_calls=tool_calls,
            model=anthropic_request.model,
            thinking_text=None,
            usage=usage,
        )

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
    tool_choice = anthropic_tool_choice_to_openai(anthropic_request.tool_choice)

    # Debug logging
    import pprint
    logger.debug(f"Converted {len(openai_messages)} messages to OpenAI format:")
    for i, msg in enumerate(openai_messages):
        logger.debug(f"  Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, has_tool_calls={bool(msg.get('tool_calls'))}")

    # Configure thinking tokens - always send for MiniMax to unlock full generation
    if anthropic_request.thinking:
        thinking_payload = anthropic_request.thinking
    elif use_minimax_parsing:
        # For MiniMax, send a very high thinking limit - model uses thinking extensively
        # Use half of max_tokens for thinking to leave room for content
        thinking_payload = {"max_thinking_tokens": min(effective_max_tokens // 2, 32768)}
    else:
        thinking_payload = None

    # Call TabbyAPI
    response = await tabby_client.chat_completion(
        messages=normalized_messages,
        model=anthropic_request.model,
        max_tokens=effective_max_tokens,
        temperature=anthropic_request.temperature,
        top_p=anthropic_request.top_p,
        top_k=anthropic_request.top_k,
        stop=anthropic_request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        add_generation_prompt=True,  # Required for <think> tags
        banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None,
        thinking=thinking_payload,
    )

    if settings.log_raw_responses:
        logger.debug(f"Raw TabbyAPI response: {response}")

    choice_payload = response["choices"][0]
    message_payload = choice_payload["message"]
    backend_content = message_payload.get("content", "") or ""
    reasoning_text = message_payload.get("reasoning_content") or ""
    backend_tool_calls = message_payload.get("tool_calls")

    # Build raw content with reasoning and content
    sections: List[str] = []

    if reasoning_text:
        trimmed_reasoning = reasoning_text.rstrip()
        sections.append(f"<think>{trimmed_reasoning}</think>")

    if backend_content.strip():
        sections.append(backend_content)

    if backend_tool_calls:
        xml_block = tool_calls_to_minimax_xml(backend_tool_calls)
        if xml_block:
            sections.append(xml_block)

    raw_content = "\n\n".join(section for section in sections if section).strip()
    if not raw_content:
        raw_content = backend_content

    wrapped_raw_content = ensure_think_wrapped(raw_content)

    # Parse tool calls from content as fallback when backend omits structured payload
    result = parse_tool_calls(wrapped_raw_content, tools)

    # Prefer backend-provided tool_calls over parsed ones
    tool_calls = backend_tool_calls
    if not tool_calls and result["tools_called"]:
        tool_calls = result["tool_calls"]

    # Extract content without tool blocks
    content_without_tool_blocks = wrapped_raw_content
    if result["tools_called"] and result["content"]:
        content_without_tool_blocks = ensure_think_wrapped(result["content"])
    elif result["tools_called"] and not result["content"]:
        content_without_tool_blocks = ""

    content_source = content_without_tool_blocks

    thinking_text = ""
    visible_text = content_source
    if settings.enable_anthropic_thinking_blocks:
        wrapped_for_split = ensure_think_wrapped(content_without_tool_blocks)
        thinking_text, visible_text = split_think(wrapped_for_split)
    else:
        thinking_text = reasoning_text.strip() if reasoning_text else ""
        visible_text = content_source

    if not (visible_text and visible_text.strip()) and choice_payload.get("finish_reason") == "length":
        visible_text = "(MiniMax stopped before it could produce a visible reply. Try increasing `max_tokens`.)"

    # Extract usage statistics from backend response
    usage = response.get("usage", {
        "input_tokens": 0,
        "output_tokens": 0
    })
    # Convert OpenAI field names to Anthropic format if needed
    if "prompt_tokens" in usage:
        usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }

    # Format as Anthropic response
    formatted = anthropic_formatter.format_complete_response(
        content=visible_text,
        tool_calls=tool_calls,
        model=anthropic_request.model,
        thinking_text=thinking_text if settings.enable_anthropic_thinking_blocks else None,
        usage=usage,
    )

    if session_id:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": ensure_think_wrapped(raw_content),
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if thinking_text and settings.enable_anthropic_thinking_blocks:
            assistant_message["reasoning_details"] = [
                {"type": "chain_of_thought", "text": thinking_text}
            ]
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

    # Check if this is a MiniMax model that needs XML parsing
    use_minimax_parsing = is_minimax_model(anthropic_request.model)

    # For MiniMax models, ensure max_tokens is high enough to avoid thinking-only responses
    effective_max_tokens = anthropic_request.max_tokens
    logger.info(f"Anthropic streaming request - model: {anthropic_request.model}, max_tokens: {effective_max_tokens}, use_minimax_parsing: {use_minimax_parsing}")
    if use_minimax_parsing and effective_max_tokens <= 8192:
        logger.info(f"Increasing max_tokens from {effective_max_tokens} to 32768 for MiniMax model (streaming)")
        effective_max_tokens = 32768

    if not use_minimax_parsing:
        logger.info(f"Non-MiniMax model detected (Anthropic/streaming): {anthropic_request.model}, passing through without XML parsing")
        # Pass through and convert OpenAI stream to Anthropic format
        tools = anthropic_tools_to_openai(anthropic_request.tools)
        tool_choice = anthropic_tool_choice_to_openai(anthropic_request.tool_choice)

        try:
            # Send message_start
            yield anthropic_formatter.format_message_start(anthropic_request.model)

            # Start first content block
            content_block_index = 0
            content_block_started = False
            tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

            async for chunk in tabby_client.extract_streaming_content(
                messages=openai_messages,
                model=anthropic_request.model,
                max_tokens=effective_max_tokens,
                temperature=anthropic_request.temperature,
                top_p=anthropic_request.top_p,
                top_k=anthropic_request.top_k,
                stop=anthropic_request.stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
            ):
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    content_delta = delta.get("content", "")
                    tool_calls_delta = delta.get("tool_calls")

                    if content_delta:
                        if not content_block_started:
                            yield anthropic_formatter.format_content_block_start(content_block_index, "text")
                            content_block_started = True
                        yield anthropic_formatter.format_content_block_delta(
                            content_block_index,
                            content_delta,
                            delta_type="text_delta"
                        )

                    if tool_calls_delta:
                        # Buffer tool calls
                        for tc_delta in tool_calls_delta:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            if "id" in tc_delta and tc_delta["id"]:
                                tool_calls_buffer[idx]["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                fn = tc_delta["function"]
                                if "name" in fn:
                                    tool_calls_buffer[idx]["function"]["name"] = fn["name"]
                                if "arguments" in fn:
                                    tool_calls_buffer[idx]["function"]["arguments"] += fn["arguments"]

                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        # Close text block if open
                        if content_block_started:
                            yield anthropic_formatter.format_content_block_stop(content_block_index)
                            content_block_index += 1

                        # Send tool calls
                        for idx in sorted(tool_calls_buffer.keys()):
                            tool_call = tool_calls_buffer[idx]
                            yield anthropic_formatter.format_tool_use_delta(content_block_index, tool_call)
                            yield anthropic_formatter.format_content_block_stop(content_block_index)
                            content_block_index += 1

                        # Map finish reason
                        stop_reason = "end_turn" if finish_reason == "stop" else finish_reason
                        if finish_reason == "tool_calls":
                            stop_reason = "tool_use"

                        yield anthropic_formatter.format_message_delta(stop_reason)
                        break

            yield anthropic_formatter.format_message_stop()

        except Exception as e:
            logger.error(f"Error in Anthropic streaming (pass-through): {e}", exc_info=True)
            error_response = anthropic_formatter.format_error(str(e))
            yield f"data: {json.dumps(error_response)}\n\n"
        return

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
    tool_choice = anthropic_tool_choice_to_openai(anthropic_request.tool_choice)

    # Initialize streaming parser
    streaming_parser = StreamingParser()
    streaming_parser.set_tools(tools)
    captured_tool_calls: Optional[Dict[int, Dict[str, Any]]] = None
    thinking_block_started = False
    text_emitted = False
    # Configure thinking tokens - always send for MiniMax to unlock full generation
    if anthropic_request.thinking:
        thinking_payload = anthropic_request.thinking
    elif use_minimax_parsing:
        # For MiniMax, send a very high thinking limit - model uses thinking extensively
        # Use half of max_tokens for thinking to leave room for content
        thinking_payload = {"max_thinking_tokens": min(effective_max_tokens // 2, 32768)}
    else:
        thinking_payload = None

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
            max_tokens=effective_max_tokens,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            stop=anthropic_request.stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=True,  # Required for <think> tags
            banned_strings=settings.banned_chinese_strings if settings.enable_chinese_char_blocking else None,
            thinking=thinking_payload,
        ):
            # Extract delta
            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                reasoning_delta = delta.get("reasoning_content", "")
                content_delta = delta.get("content", "")
                tool_calls_delta = delta.get("tool_calls")

                # Handle structured reasoning_content field from TabbyAPI
                if reasoning_delta and settings.enable_anthropic_thinking_blocks:
                    if not thinking_block_started:
                        yield anthropic_formatter.format_content_block_start(content_block_index, "thinking")
                        thinking_block_started = True
                    yield anthropic_formatter.format_content_block_delta(
                        content_block_index,
                        reasoning_delta,
                        delta_type="thinking_delta",
                    )

                if content_delta:
                    # Close thinking block if we're starting content
                    if thinking_block_started and settings.enable_anthropic_thinking_blocks:
                        yield anthropic_formatter.format_content_block_stop(content_block_index)
                        content_block_index += 1
                        thinking_block_started = False

                    # Start text block if not started
                    if not content_block_started:
                        yield anthropic_formatter.format_content_block_start(content_block_index, "text")
                        content_block_started = True

                    # Send content delta
                    yield anthropic_formatter.format_content_block_delta(
                        content_block_index,
                        content_delta,
                        delta_type="text_delta"
                    )
                    text_emitted = True

                # Handle tool calls
                if tool_calls_delta:
                    logger.debug(f"Received tool_calls_delta: {tool_calls_delta}")

                    # Close any open blocks
                    if content_block_started:
                        yield anthropic_formatter.format_content_block_stop(content_block_index)
                        content_block_index += 1
                        content_block_started = False
                    if thinking_block_started:
                        yield anthropic_formatter.format_content_block_stop(content_block_index)
                        content_block_index += 1
                        thinking_block_started = False

                    # Initialize captured_tool_calls if needed
                    if captured_tool_calls is None:
                        captured_tool_calls = {}

                    # Buffer tool calls until complete
                    for tc_delta in tool_calls_delta:
                        idx = tc_delta.get("index", 0)
                        if idx not in captured_tool_calls:
                            captured_tool_calls[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        if "id" in tc_delta:
                            captured_tool_calls[idx]["id"] = tc_delta["id"]
                        if "function" in tc_delta:
                            fn = tc_delta["function"]
                            if "name" in fn:
                                captured_tool_calls[idx]["function"]["name"] = fn["name"]
                            if "arguments" in fn:
                                logger.debug(f"Adding arguments: '{fn['arguments']}' to tool {captured_tool_calls[idx]['function']['name']}")
                                captured_tool_calls[idx]["function"]["arguments"] += fn["arguments"]

                # OLD PARSER-BASED LOGIC (keeping for fallback)
                if content_delta and False:  # Disabled
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
                                text_emitted = True

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
                        content_block_started = False
                    if thinking_block_started:
                        yield anthropic_formatter.format_content_block_stop(content_block_index)
                        thinking_block_started = False

                    # Emit buffered tool calls if any
                    if captured_tool_calls:
                        for idx in sorted(captured_tool_calls.keys()):
                            tool_call = captured_tool_calls[idx]
                            # Debug logging
                            logger.info(f"Emitting tool call: {tool_call['function']['name']}, args: {tool_call['function']['arguments'][:100] if tool_call['function']['arguments'] else 'EMPTY'}")

                            # Send tool_use start with empty input
                            yield anthropic_formatter.format_tool_use_start(
                                content_block_index,
                                tool_call["id"],
                                tool_call["function"]["name"]
                            )

                            # Send the arguments as input_json_delta
                            if tool_call["function"]["arguments"]:
                                yield anthropic_formatter.format_tool_input_delta(
                                    content_block_index,
                                    tool_call["function"]["arguments"]
                                )

                            # Close the tool_use block
                            yield anthropic_formatter.format_content_block_stop(content_block_index)
                            content_block_index += 1

                    # Override finish_reason if tool calls were detected
                    if finish_reason == "stop" and (captured_tool_calls or streaming_parser.has_tool_calls()):
                        finish_reason = "tool_calls"

                    # Map finish reason
                    stop_reason = "end_turn" if finish_reason == "stop" else finish_reason
                    if finish_reason == "tool_calls":
                        stop_reason = "tool_use"

                    yield anthropic_formatter.format_message_delta(stop_reason)

        # Send message_stop
        if thinking_block_started:
            yield anthropic_formatter.format_content_block_stop(content_block_index)
            thinking_block_started = False

        if not text_emitted and not captured_tool_calls:
            yield anthropic_formatter.format_content_block_start(content_block_index, "text")
            yield anthropic_formatter.format_content_block_delta(
                content_block_index,
                "(MiniMax stopped before it could produce a visible reply. Try increasing `max_tokens`.)",
                delta_type="text_delta",
            )
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
