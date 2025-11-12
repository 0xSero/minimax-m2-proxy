"""Configuration settings for MiniMax-M2 Proxy"""

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Proxy configuration settings"""

    # Backend TabbyAPI configuration
    tabby_url: str = "http://localhost:8000"
    tabby_timeout: int = 300  # seconds

    # Proxy server configuration
    host: str = "0.0.0.0"
    port: int = 8001

    # Feature flags
    enable_thinking_passthrough: bool = True  # Keep <think> blocks in responses
    enable_tool_translation: bool = True      # Translate <minimax:tool_call> to OpenAI/Anthropic
    enable_chinese_char_blocking: bool = True  # Block Chinese character generation

    # Session/history repair settings
    session_store_enabled: bool = False
    session_store_backend: Literal["memory", "sqlite"] = "sqlite"
    session_store_path: str = "conversations.db"
    session_ttl_seconds: int = 3600
    max_messages_per_session: int = 8
    require_session_for_repair: bool = True

    # Reasoning/thinking presentation
    enable_reasoning_split: bool = True
    enable_anthropic_thinking_blocks: bool = True
    anthropic_default_thinking_tokens: int = 128

    # Validation toggles
    require_tool_link_validation: bool = True

    # Simple auth
    auth_api_key: Optional[str] = None

    # Chinese character blocking (fixes tokenizer vocab bleed)
    banned_chinese_strings: list[str] = [
        "、", "。", "，", "的", "了", "是", "在", "有", "个", "人", "这", "我",
        "你", "他", "们", "来", "到", "时", "要", "就", "会", "可", "那", "些"
    ]

    # Model detection
    minimax_model_patterns: list[str] = ["minimax", "m2"]  # Model name patterns that use MiniMax format

    # Logging
    log_level: str = "DEBUG"
    log_raw_responses: bool = False  # Log raw backend responses (debug)
    enable_streaming_debug: bool = False  # Emit detailed streaming traces for troubleshooting
    streaming_debug_path: Optional[str] = None  # Optional file path for streaming trace logs

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
