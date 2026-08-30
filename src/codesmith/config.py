"""Project-level CodeSmith configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "model": {"provider": "openrouter", "model": "openai/gpt-oss-20b:free", "temperature": 0.1},
    "agent": {"max_iterations": 20, "auto_test": True},
    "context": {
        "max_tokens": 15_000,
        "max_message_tokens": 4_500,
        "max_tool_result_tokens": 3_000,
        "max_summary_tokens": 2_000,
        "max_file_chars": 10_000,
        "max_referenced_files": 5,
        "compact_threshold": 0.80,
        "keep_recent_turns": 2,
        "project_doc_max_bytes": 32_768,
        "project_doc_fallback_filenames": [],
    },
    "permissions": {"file_write": "confirm", "file_delete": "confirm", "shell": "confirm"},
    "search": {"backend": "ripgrep"},
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(repository: str | Path = ".") -> Dict[str, Any]:
    """Load `.codesmith/config.yaml`, returning defaults when absent."""
    path = Path(repository).resolve() / ".codesmith" / "config.yaml"
    if not path.is_file():
        return _merge({}, DEFAULT_CONFIG)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read .codesmith/config.yaml") from exc
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(".codesmith/config.yaml must contain a mapping")
    return _merge(DEFAULT_CONFIG, data)
