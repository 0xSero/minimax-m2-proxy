"""Session storage and history repair helpers."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from parsers.reasoning import ensure_think_wrapped


@dataclass
class RepairResult:
    """Result metadata for history repair attempts."""

    messages: List[Dict[str, Any]]
    repaired: bool = False
    reason: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "repaired": self.repaired,
            "reason": self.reason,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "message_count": len(self.messages),
        }


class SessionStore:
    """Lightweight session history store with optional SQLite backend."""

    def __init__(
        self,
        enabled: bool,
        backend: str,
        db_path: str,
        ttl_seconds: int,
        max_messages: int,
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._lock = threading.Lock()

        self._memory_store: Dict[str, List[Dict[str, Any]]] = {}
        self._conn: Optional[sqlite3.Connection] = None

        if not self.enabled:
            return

        if self.backend == "sqlite":
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_file, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        session_id TEXT NOT NULL,
                        ts INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session_ts "
                    "ON messages(session_id, ts)"
                )
        elif self.backend == "memory":
            # Nothing to initialize beyond the in-memory dict.
            pass
        else:
            raise ValueError(f"Unsupported session_store_backend: {self.backend}")

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _cleanup_locked(self) -> None:
        if not self.enabled:
            return

        cutoff = int(time.time()) - self.ttl_seconds
        if cutoff <= 0:
            return

        if self.backend == "memory":
            empty_sessions = []
            for session_id, records in self._memory_store.items():
                filtered = [
                    record for record in records if record["ts"] >= cutoff
                ]
                if filtered:
                    self._memory_store[session_id] = filtered[-self.max_messages :]
                else:
                    empty_sessions.append(session_id)
            for session_id in empty_sessions:
                self._memory_store.pop(session_id, None)
        elif self.backend == "sqlite" and self._conn is not None:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM messages WHERE ts < ?",
                    (cutoff,),
                )

    def append_message(self, session_id: str, message: Dict[str, Any]) -> None:
        if not self.enabled or not session_id:
            return

        payload = json.dumps(message, ensure_ascii=False)
        ts = int(time.time())

        with self._lock:
            self._cleanup_locked()
            if self.backend == "memory":
                bucket = self._memory_store.setdefault(session_id, [])
                bucket.append({"ts": ts, "role": message.get("role", ""), "payload": payload})
                if len(bucket) > self.max_messages:
                    self._memory_store[session_id] = bucket[-self.max_messages :]
            elif self.backend == "sqlite" and self._conn is not None:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO messages(session_id, ts, role, payload) VALUES (?, ?, ?, ?)",
                        (session_id, ts, message.get("role", ""), payload),
                    )
                    # Trim to latest max_messages entries per session
                    self._conn.execute(
                        """
                        DELETE FROM messages
                        WHERE session_id = ?
                          AND rowid NOT IN (
                            SELECT rowid FROM messages
                            WHERE session_id = ?
                            ORDER BY ts DESC, rowid DESC
                            LIMIT ?
                          )
                        """,
                        (session_id, session_id, self.max_messages),
                    )

    def _load_session_locked(self, session_id: str) -> List[Dict[str, Any]]:
        if not self.enabled or not session_id:
            return []

        records: List[Dict[str, Any]] = []

        if self.backend == "memory":
            for record in self._memory_store.get(session_id, []):
                try:
                    payload = json.loads(record["payload"])
                except json.JSONDecodeError:
                    continue
                records.append(payload)
        elif self.backend == "sqlite" and self._conn is not None:
            cursor = self._conn.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY ts ASC, rowid ASC",
                (session_id,),
            )
            for row in cursor.fetchall():
                try:
                    records.append(json.loads(row["payload"]))
                except json.JSONDecodeError:
                    continue
        return records

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        if not self.enabled or not session_id:
            return []
        with self._lock:
            self._cleanup_locked()
            return self._load_session_locked(session_id)

    def get_last_assistant(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not session_id:
            return None

        with self._lock:
            self._cleanup_locked()
            records = self._load_session_locked(session_id)

        for message in reversed(records):
            if message.get("role") == "assistant":
                return message
        return None

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_assistant_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Merge reasoning details into content for comparison."""
        msg_copy = dict(message)
        if msg_copy.get("role") != "assistant":
            return msg_copy

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

        return msg_copy

    @staticmethod
    def _assistant_in_history(messages: List[Dict[str, Any]], assistant: Dict[str, Any]) -> bool:
        if not assistant:
            return True

        normalized_assistant = SessionStore._normalize_assistant_message(assistant)

        for message in reversed(messages):
            if message.get("role") == "assistant":
                if SessionStore._normalize_assistant_message(message) == normalized_assistant:
                    return True
        return False

    def inject_or_repair(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str],
        *,
        require_session: bool,
    ) -> RepairResult:
        # Early exits
        if not self.enabled:
            return RepairResult(messages=list(messages), skipped=True, skip_reason="disabled")

        if not session_id:
            skip_reason = "missing_session_id" if require_session else "missing_session_id_optional"
            return RepairResult(messages=list(messages), skipped=True, skip_reason=skip_reason)

        stored_assistant = self.get_last_assistant(session_id)
        if not stored_assistant:
            return RepairResult(messages=list(messages), skipped=True, skip_reason="no_history")

        current_messages = [dict(msg) for msg in messages]
        if self._assistant_in_history(current_messages, stored_assistant):
            return RepairResult(messages=current_messages, skipped=True, skip_reason="assistant_present")

        # Inject stored assistant before first tool/tool_result message
        insert_index = len(current_messages)
        for idx, message in enumerate(current_messages):
            role = message.get("role")
            if role == "tool":
                insert_index = idx
                break
            if role == "user" and message.get("name") == "tool_result":
                insert_index = idx
                break

        repaired_messages = current_messages[:insert_index] + [stored_assistant] + current_messages[insert_index:]
        return RepairResult(messages=repaired_messages, repaired=True, reason="assistant_injected")


def build_session_store() -> SessionStore:
    """Factory helper using global settings."""
    return SessionStore(
        enabled=settings.session_store_enabled,
        backend=settings.session_store_backend,
        db_path=settings.session_store_path,
        ttl_seconds=settings.session_ttl_seconds,
        max_messages=settings.max_messages_per_session,
    )


# Global session store instance used by proxy.main
session_store = build_session_store()
