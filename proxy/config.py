"""Configuration settings for MiniMax-M2 Proxy"""

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

    # Chinese character blocking (fixes tokenizer vocab bleed)
    banned_chinese_strings: list[str] = [
        "、", "。", "，", "的", "了", "是", "在", "有", "个", "人", "这", "我",
        "你", "他", "们", "来", "到", "时", "要", "就", "会", "可", "那", "些"
    ]

    # Logging
    log_level: str = "INFO"
    log_raw_responses: bool = False  # Log raw backend responses (debug)
    enable_streaming_debug: bool = False  # Emit detailed streaming traces for troubleshooting
    streaming_debug_path: str | None = None  # Optional file path for streaming trace logs

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
