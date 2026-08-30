"""OpenRouter-compatible context management for repository-aware agent calls."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class RepositoryMetadata:
    root: str
    projects: Dict[str, List[str]]
    rules_file: str | None = None


@dataclass(frozen=True)
class InstructionSource:
    """One instruction file included in the model's system context."""

    path: str
    content: str


@dataclass(frozen=True)
class ContextLimits:
    """Budgets for model input and local context sources.

    OpenRouter exposes model context sizes and post-request token usage, but its
    models use different tokenizers. CodeSmith therefore uses a conservative
    provider-independent estimate before a request and records actual usage
    returned by OpenRouter afterward.

    The character settings remain accepted for backwards compatibility. New
    configurations should prefer their token equivalents.
    """

    max_tokens: int = 15_000
    max_message_tokens: int = 4_500
    max_tool_result_tokens: int = 3_000
    max_summary_tokens: int = 2_000
    max_file_chars: int = 10_000
    max_referenced_files: int = 5
    compact_threshold: float = 0.80
    keep_recent_turns: int = 2
    project_doc_max_bytes: int = 32 * 1024
    project_doc_fallback_filenames: tuple[str, ...] = ()

    # Deprecated compatibility views used by older configuration and callers.
    max_chars: int = 60_000
    max_message_chars: int = 18_000
    max_tool_result_chars: int = 12_000

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "ContextLimits":
        values = dict(config or {})

        def positive_int(name: str, default: int) -> int:
            value = values.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"context.{name} must be a positive integer")
            return value

        legacy_max_chars = positive_int("max_chars", cls.max_chars)
        legacy_message_chars = positive_int(
            "max_message_chars", cls.max_message_chars
        )
        legacy_tool_chars = positive_int(
            "max_tool_result_chars", cls.max_tool_result_chars
        )
        max_tokens = positive_int(
            "max_tokens", max(256, (legacy_max_chars + 3) // 4)
        )
        if max_tokens < 256:
            raise ValueError("context.max_tokens must be at least 256")

        max_message_tokens = min(
            positive_int(
                "max_message_tokens", max(128, (legacy_message_chars + 3) // 4)
            ),
            max(128, max_tokens // 2),
        )
        max_tool_result_tokens = min(
            positive_int(
                "max_tool_result_tokens", max(128, (legacy_tool_chars + 3) // 4)
            ),
            max(128, max_tokens // 2),
        )
        max_summary_tokens = min(
            positive_int("max_summary_tokens", cls.max_summary_tokens),
            max_message_tokens,
        )

        threshold = values.get("compact_threshold", cls.compact_threshold)
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.5 <= float(threshold) < 1.0
        ):
            raise ValueError(
                "context.compact_threshold must be between 0.5 and 1.0"
            )

        fallback_names = values.get(
            "project_doc_fallback_filenames", cls.project_doc_fallback_filenames
        )
        if not isinstance(fallback_names, (list, tuple)) or not all(
            isinstance(name, str) and name.strip() for name in fallback_names
        ):
            raise ValueError(
                "context.project_doc_fallback_filenames must be a list of names"
            )

        return cls(
            max_tokens=max_tokens,
            max_message_tokens=max_message_tokens,
            max_tool_result_tokens=max_tool_result_tokens,
            max_summary_tokens=max_summary_tokens,
            max_file_chars=positive_int("max_file_chars", cls.max_file_chars),
            max_referenced_files=positive_int(
                "max_referenced_files", cls.max_referenced_files
            ),
            compact_threshold=float(threshold),
            keep_recent_turns=positive_int(
                "keep_recent_turns", cls.keep_recent_turns
            ),
            project_doc_max_bytes=positive_int(
                "project_doc_max_bytes", cls.project_doc_max_bytes
            ),
            project_doc_fallback_filenames=tuple(fallback_names),
            max_chars=legacy_max_chars,
            max_message_chars=legacy_message_chars,
            max_tool_result_chars=legacy_tool_chars,
        )


MARKERS = {
    "python": ("pyproject.toml", "requirements.txt", "setup.py"),
    "java": ("pom.xml", "build.gradle", "build.gradle.kts"),
    "node": ("package.json",),
    "go": ("go.mod",),
    "rust": ("Cargo.toml",),
}

_REFERENCE = re.compile(r"(?<![\w@])@([A-Za-z0-9_.\\/-]+)")
_TOKEN_PART = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(value: str) -> int:
    """Conservatively estimate tokens across OpenRouter model families."""
    if not value:
        return 0
    total = 0
    for part in _TOKEN_PART.findall(value):
        byte_length = len(part.encode("utf-8"))
        if part.isspace():
            total += max(1, (byte_length + 7) // 8)
        elif part.isalnum() or part.replace("_", "").isalnum():
            total += max(1, (byte_length + 3) // 4)
        else:
            total += max(1, (byte_length + 1) // 2)
    return total


def _truncate(value: str, limit: int, label: str = "content") -> str:
    """Legacy character truncation used for repository file reads."""
    if len(value) <= limit:
        return value
    marker = f"\n... [{label} truncated]"
    prefix_length = max(0, limit - len(marker))
    return (value[:prefix_length] + marker)[:limit]


def _truncate_tokens(value: str, limit: int, label: str = "content") -> str:
    if estimate_tokens(value) <= limit:
        return value
    marker = f"\n... [{label} truncated]"
    marker_tokens = estimate_tokens(marker)
    if marker_tokens >= limit:
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(value[:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        return value[:low]
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(value[:middle]) + marker_tokens <= limit:
            low = middle
        else:
            high = middle - 1
    return value[:low] + marker


def discover_repository(root: str | Path = ".") -> RepositoryMetadata:
    """Detect common project types and legacy CodeSmith rules."""
    path = Path(root).resolve()
    projects = {
        language: [marker for marker in markers if (path / marker).is_file()]
        for language, markers in MARKERS.items()
    }
    projects = {language: markers for language, markers in projects.items() if markers}
    rules = path / ".codesmith" / "rules.md"
    return RepositoryMetadata(
        str(path), projects, str(rules) if rules.is_file() else None
    )


def load_rules(root: str | Path = ".") -> str:
    """Read legacy `.codesmith/rules.md`, returning an empty string if absent."""
    path = Path(root).resolve() / ".codesmith" / "rules.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


class ContextManager:
    """Compose prioritized instructions and bounded model-visible messages."""

    def __init__(
        self,
        root: str | Path = ".",
        config: Mapping[str, Any] | None = None,
        working_directory: str | Path | None = None,
        codesmith_home: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        candidate_directory = Path(working_directory or self.root).resolve()
        try:
            candidate_directory.relative_to(self.root)
        except ValueError:
            candidate_directory = self.root
        self.working_directory = candidate_directory
        default_home = Path(
            os.getenv("CODESMITH_HOME", str(Path.home() / ".codesmith"))
        )
        self.codesmith_home = Path(codesmith_home or default_home).resolve()
        self.metadata = discover_repository(self.root)
        self.limits = ContextLimits.from_config(config)

    def repository_summary(self, sources: Sequence[InstructionSource] = ()) -> str:
        projects = ", ".join(
            f"{language} ({', '.join(markers)})"
            for language, markers in sorted(self.metadata.projects.items())
        ) or "no recognized project markers"
        source_names = ", ".join(source.path for source in sources) or "none"
        return (
            "Repository metadata:\n"
            f"- Root: {self.metadata.root}\n"
            f"- Working directory: {self.working_directory}\n"
            f"- Detected projects: {projects}\n"
            f"- Instruction sources: {source_names}"
        )

    @staticmethod
    def _read_instruction(path: Path, label: str) -> InstructionSource | None:
        try:
            data = path.read_bytes()
            if not data or b"\0" in data:
                return None
            content = data.decode("utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if not content:
            return None
        return InstructionSource(label, content)

    def instruction_sources(self) -> list[InstructionSource]:
        """Discover global and project instructions using Codex-style precedence."""
        candidates: list[tuple[Path, str]] = []
        for name in ("AGENTS.override.md", "AGENTS.md"):
            path = self.codesmith_home / name
            if path.is_file():
                candidates.append((path, str(path)))
                break

        legacy_rules = self.root / ".codesmith" / "rules.md"
        if legacy_rules.is_file():
            candidates.append((legacy_rules, ".codesmith/rules.md"))

        relative_directory = self.working_directory.relative_to(self.root)
        directories = [self.root]
        cursor = self.root
        for part in relative_directory.parts:
            cursor = cursor / part
            directories.append(cursor)

        names = (
            "AGENTS.override.md",
            "AGENTS.md",
            *self.limits.project_doc_fallback_filenames,
        )
        for directory in directories:
            for name in names:
                path = directory / name
                if path.is_file():
                    candidates.append((path, str(path.relative_to(self.root))))
                    break

        sources: list[InstructionSource] = []
        used_bytes = 0
        for path, label in candidates:
            source = self._read_instruction(path, label)
            if source is None:
                continue
            remaining = self.limits.project_doc_max_bytes - used_bytes
            if remaining <= 0:
                break
            encoded = source.content.encode("utf-8")
            if len(encoded) > remaining:
                content = encoded[:remaining].decode("utf-8", errors="ignore")
                source = InstructionSource(
                    source.path,
                    content + "\n... [instruction budget reached]",
                )
            sources.append(source)
            used_bytes += min(len(encoded), remaining)
            if used_bytes >= self.limits.project_doc_max_bytes:
                break
        return sources

    def referenced_paths(self, request: str) -> list[str]:
        """Return unique paths explicitly mentioned with ``@`` in a request."""
        paths = []
        for match in _REFERENCE.finditer(request):
            candidate = match.group(1).rstrip(".,:;!?)]}")
            if candidate and candidate not in paths:
                paths.append(candidate)
            if len(paths) >= self.limits.max_referenced_files:
                break
        return paths

    def reference_error(self, relative_path: str) -> str | None:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return "path is outside the repository"
        if not candidate.is_file():
            return "file not found"
        return None

    def _read_reference(self, relative_path: str) -> str:
        error = self.reference_error(relative_path)
        if error:
            return f"[Unavailable @{relative_path}: {error}]"
        candidate = (self.root / relative_path).resolve()
        try:
            data = candidate.read_bytes()
            if b"\0" in data:
                return f"[Unavailable @{relative_path}: binary file]"
            content = data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"[Unavailable @{relative_path}: {exc}]"
        content = _truncate(content, self.limits.max_file_chars, f"@{relative_path}")
        return f"Referenced file @{relative_path}:\n```text\n{content}\n```"

    def build_system_prompt(self, base_prompt: str, rules: str = "") -> str:
        """Combine stable instructions, layered project guidance, and metadata."""
        sources = self.instruction_sources()
        if rules and not any(source.path == ".codesmith/rules.md" for source in sources):
            sources.insert(0, InstructionSource(".codesmith/rules.md", rules.strip()))
        sections = [base_prompt.strip()]
        for source in sources:
            sections.append(f"Instructions from {source.path}:\n{source.content}")
        sections.append(self.repository_summary(sources))
        return _truncate_tokens(
            "\n\n".join(sections),
            self.limits.max_message_tokens,
            "initial repository context",
        )

    def _augment_user_request(self, message: dict) -> dict:
        request = message.get("content")
        if not isinstance(request, str):
            return message
        references = [self._read_reference(path) for path in self.referenced_paths(request)]
        if references:
            message["content"] = _truncate_tokens(
                request + "\n\nExplicitly referenced source:\n\n" + "\n\n".join(references),
                self.limits.max_message_tokens,
                "request context",
            )
        return message

    @staticmethod
    def _message_size(message: Mapping[str, Any]) -> int:
        """Return serialized characters for compatibility and diagnostics."""
        return len(json.dumps(message, ensure_ascii=False, default=str))

    @staticmethod
    def _message_tokens(message: Mapping[str, Any]) -> int:
        return estimate_tokens(json.dumps(message, ensure_ascii=False, default=str))

    def messages_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        return sum(self._message_tokens(message) for message in messages)

    def _compact_tool_calls(self, calls: Sequence[Mapping[str, Any]]) -> list[dict]:
        compacted = copy.deepcopy(list(calls))
        argument_limit = max(64, self.limits.max_tool_result_tokens // 2)
        for call in compacted:
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and estimate_tokens(arguments) > argument_limit:
                try:
                    parsed = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    for name, value in parsed.items():
                        if isinstance(value, str) and estimate_tokens(value) > 128:
                            parsed[name] = (
                                f"[historical {name} omitted after tool execution; "
                                f"{len(value)} characters]"
                            )
                    replacement = json.dumps(parsed, ensure_ascii=False)
                else:
                    replacement = json.dumps(
                        {"context_note": "Historical tool arguments omitted after execution."}
                    )
                if estimate_tokens(replacement) > argument_limit:
                    replacement = json.dumps(
                        {"context_note": "Historical tool arguments omitted after execution."}
                    )
                function["arguments"] = replacement
        return compacted

    def _compact_message(self, message: Mapping[str, Any]) -> dict:
        compacted = copy.deepcopy(dict(message))
        role = compacted.get("role")
        content = compacted.get("content")
        if isinstance(content, str):
            limit = (
                self.limits.max_tool_result_tokens
                if role == "tool"
                else self.limits.max_message_tokens
            )
            compacted["content"] = _truncate_tokens(
                content, limit, f"{role or 'message'} content"
            )
        if isinstance(compacted.get("tool_calls"), list):
            compacted["tool_calls"] = self._compact_tool_calls(compacted["tool_calls"])
        return compacted

    def _task_state_message(self, state: Any) -> dict:
        files_read = sorted(
            str(path) for path in getattr(state, "files_read", set()) if path
        )
        files_modified = sorted(
            str(path) for path in getattr(state, "files_modified", set()) if path
        )
        lines = [
            "Current task state:",
            f"- Iteration: {getattr(state, 'iteration', 0)}/{getattr(state, 'max_iterations', '?')}",
            f"- Files read: {json.dumps(files_read[:20]) if files_read else 'none'}",
            f"- Files modified: {json.dumps(files_modified[:20]) if files_modified else 'none'}",
            f"- Tool calls completed: {len(getattr(state, 'tool_calls', []))}",
        ]
        results = getattr(state, "tool_results", [])
        if results:
            latest = results[-1]
            outcome = "success" if getattr(latest, "ok", False) else "failure"
            lines.append(f"- Latest tool result: {outcome}")
        return {"role": "system", "content": "\n".join(lines)}

    @staticmethod
    def _conversation_units(messages: list[dict]) -> list[list[dict]]:
        """Keep assistant tool calls and their observations as atomic units."""
        units: list[list[dict]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                unit = [message]
                index += 1
                while index < len(messages) and messages[index].get("role") == "tool":
                    unit.append(messages[index])
                    index += 1
                units.append(unit)
            else:
                units.append([message])
                index += 1
        return units

    @staticmethod
    def _conversation_turns(messages: list[dict]) -> list[list[dict]]:
        """Group each historical user question with its complete answer cycle."""
        turns: list[list[dict]] = []
        current: list[dict] = []
        for message in messages:
            if message.get("role") == "user":
                if current:
                    turns.append(current)
                current = [message]
            elif current:
                current.append(message)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _minimal_turn(turn: Sequence[Mapping[str, Any]]) -> list[dict]:
        question = next(
            (dict(message) for message in turn if message.get("role") == "user"),
            None,
        )
        answer = next(
            (
                dict(message)
                for message in reversed(turn)
                if message.get("role") == "assistant" and not message.get("tool_calls")
            ),
            None,
        )
        return [message for message in (question, answer) if message]

    def compact_history(self, messages: Sequence[Mapping[str, Any]]) -> list[dict]:
        """Retain the newest complete turns within the active token budget."""
        compacted = [
            self._compact_message(message)
            for message in messages
            if message.get("role") != "system"
        ]
        selected: list[list[dict]] = []
        used = 0
        for turn in reversed(self._conversation_turns(compacted)):
            turn_size = self.messages_tokens(turn)
            if used + turn_size <= self.limits.max_tokens:
                selected.append(turn)
                used += turn_size
                continue
            if not selected:
                fallback = self._minimal_turn(turn)
                if fallback:
                    selected.append(fallback)
            break
        return [message for turn in reversed(selected) for message in turn]

    def split_for_compaction(
        self,
        messages: Sequence[Mapping[str, Any]],
        force: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Split older summarizable turns from a small verbatim recent suffix."""
        compacted = [
            self._compact_message(message)
            for message in messages
            if message.get("role") != "system"
        ]
        turns = self._conversation_turns(compacted)
        threshold = int(self.limits.max_tokens * self.limits.compact_threshold)
        if not force and self.messages_tokens(compacted) < threshold:
            return [], compacted
        if not turns:
            return [], []
        if len(turns) == 1:
            if not force and self.messages_tokens(turns[0]) < threshold:
                return [], compacted
            minimal = self._minimal_turn(turns[0])
            if self.messages_tokens(minimal) >= self.messages_tokens(turns[0]):
                return [], compacted
            return list(turns[0]), minimal

        recent: list[list[dict]] = []
        recent_budget = max(128, self.limits.max_tokens // 3)
        recent_tokens = 0
        for turn in reversed(turns):
            turn_tokens = self.messages_tokens(turn)
            if recent and (
                len(recent) >= self.limits.keep_recent_turns
                or recent_tokens + turn_tokens > recent_budget
            ):
                break
            recent.append(turn)
            recent_tokens += turn_tokens
        recent.reverse()
        older_count = len(turns) - len(recent)
        older = [message for turn in turns[:older_count] for message in turn]
        retained = [message for turn in recent for message in turn]
        return older, retained

    def compaction_messages(
        self,
        prior_summary: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict]:
        """Build a portable model request for semantic context summarization."""
        transcript = json.dumps(
            [self._compact_message(message) for message in messages],
            ensure_ascii=False,
            default=str,
        )
        transcript = _truncate_tokens(
            transcript,
            self.limits.max_message_tokens,
            "compaction transcript",
        )
        prompt = (
            "Previous compacted summary:\n"
            f"{prior_summary.strip() or 'none'}\n\n"
            "Conversation items to merge into the summary:\n"
            f"{transcript}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "Create a concise continuation summary for a coding agent. "
                    "Preserve the user's objective and constraints, decisions and "
                    "assumptions, completed actions, files read or changed, commands "
                    "and outcomes, errors, unresolved blockers, and the next concrete "
                    "step. Do not invent facts. Return only the summary."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def fallback_summary(
        self,
        prior_summary: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        """Create a deterministic summary if the auxiliary model call fails."""
        sections = []
        if prior_summary.strip():
            sections.append("Prior summary:\n" + prior_summary.strip())
        for turn in self._conversation_turns(list(messages)):
            question = next(
                (message.get("content") for message in turn if message.get("role") == "user"),
                None,
            )
            answer = next(
                (
                    message.get("content")
                    for message in reversed(turn)
                    if message.get("role") == "assistant" and not message.get("tool_calls")
                ),
                None,
            )
            tool_names = []
            for message in turn:
                for call in message.get("tool_calls", []) or []:
                    function = call.get("function") or {}
                    name = function.get("name")
                    if name:
                        tool_names.append(str(name))
            lines = []
            if question:
                lines.append("User: " + str(question))
            if tool_names:
                lines.append("Tools used: " + ", ".join(tool_names))
            if answer:
                lines.append("Outcome: " + str(answer))
            if lines:
                sections.append("\n".join(lines))
        return _truncate_tokens(
            "\n\n".join(sections),
            self.limits.max_summary_tokens,
            "conversation summary",
        )

    def prepare_messages(
        self, messages: Sequence[Mapping[str, Any]], state: Any
    ) -> list[dict]:
        """Prioritize current work while retaining bounded multi-turn history."""
        compacted = [self._compact_message(message) for message in messages]
        leading_system = []
        while compacted and compacted[0].get("role") == "system":
            leading_system.append(compacted.pop(0))

        current_user_index = next(
            (
                index
                for index in range(len(compacted) - 1, -1, -1)
                if compacted[index].get("role") == "user"
            ),
            None,
        )
        if current_user_index is None:
            history_messages = compacted
            current_user = None
            current_messages = []
        else:
            history_messages = compacted[:current_user_index]
            current_user = self._augment_user_request(compacted[current_user_index])
            current_messages = compacted[current_user_index + 1 :]

        task_message = self._task_state_message(state)
        essential = leading_system + [task_message]
        if current_user is not None:
            essential.append(current_user)
        current_unit_candidates = self._conversation_units(current_messages)
        mandatory_unit_tokens = (
            self.messages_tokens(current_unit_candidates[-1])
            if current_unit_candidates
            else 0
        )
        total_essential_budget = max(
            64, self.limits.max_tokens - mandatory_unit_tokens
        )
        essential_budget = max(
            24, total_essential_budget // max(1, len(essential))
        )
        for message in essential:
            if self._message_tokens(message) > essential_budget:
                message["content"] = _truncate_tokens(
                    str(message.get("content", "")),
                    max(8, essential_budget - 16),
                    "priority context",
                )

        used = self.messages_tokens(essential)
        current_units = []
        for unit in reversed(current_unit_candidates):
            unit_size = self.messages_tokens(unit)
            # The latest assistant-tool-result unit is required for a valid
            # OpenRouter continuation. Keep it even if a provider returned an
            # unusually large opaque reasoning block.
            if not current_units or used + unit_size <= self.limits.max_tokens:
                current_units.append(unit)
                used += unit_size
            else:
                break

        historical_turns = []
        for turn in reversed(self._conversation_turns(history_messages)):
            turn_size = self.messages_tokens(turn)
            if used + turn_size <= self.limits.max_tokens:
                historical_turns.append(turn)
                used += turn_size
            else:
                break

        selected = leading_system + [task_message]
        for turn in reversed(historical_turns):
            selected.extend(turn)
        if current_user is not None:
            selected.append(current_user)
        for unit in reversed(current_units):
            selected.extend(unit)
        return selected


@dataclass
class ConversationSession:
    """Conversation state shared across requests and safe to persist as JSON."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    repository: str = ""
    model: str = ""
    summary: str = ""
    history: list[dict] = field(default_factory=list)
    compaction_backlog: list[dict] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    pending_references: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def update(self, state: Any, context_manager: ContextManager) -> None:
        messages = [
            context_manager._compact_message(message)
            for message in getattr(state, "messages", [])
            if message.get("role") != "system"
        ]
        older, recent = context_manager.split_for_compaction(messages)
        self.compaction_backlog = older
        self.history = recent
        self.files_read = set(getattr(state, "files_read", set()))
        self.files_modified = set(getattr(state, "files_modified", set()))
        self.input_tokens += int(getattr(state, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(state, "output_tokens", 0) or 0)
        self.updated_at = _utc_now()

    def estimated_context_tokens(self, context_manager: ContextManager) -> int:
        return estimate_tokens(self.summary) + context_manager.messages_tokens(self.history)

    def needs_compaction(self) -> bool:
        return bool(self.compaction_backlog)

    def prepare_for_forced_compaction(self, context_manager: ContextManager) -> bool:
        older, recent = context_manager.split_for_compaction(self.history, force=True)
        if not older:
            return False
        self.compaction_backlog = older
        self.history = recent
        return True

    async def compact(self, model: Any, context_manager: ContextManager) -> bool:
        if not self.compaction_backlog:
            return False
        response = await model.generate(
            context_manager.compaction_messages(self.summary, self.compaction_backlog),
            [],
        )
        content = str(getattr(response, "content", "") or "").strip()
        if not content:
            raise ValueError("OpenRouter returned an empty compaction summary")
        self.summary = _truncate_tokens(
            content,
            context_manager.limits.max_summary_tokens,
            "conversation summary",
        )
        self.compaction_backlog.clear()
        usage = getattr(response, "usage", None) or {}
        self.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.output_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.updated_at = _utc_now()
        return True

    def compact_without_model(self, context_manager: ContextManager) -> bool:
        if not self.compaction_backlog:
            return False
        self.summary = context_manager.fallback_summary(
            self.summary, self.compaction_backlog
        )
        self.compaction_backlog.clear()
        self.updated_at = _utc_now()
        return True

    def queue_reference(self, path: str, context_manager: ContextManager) -> None:
        error = context_manager.reference_error(path)
        if error:
            raise ValueError(f"Cannot mention {path}: {error}")
        if path not in self.pending_references:
            self.pending_references.append(path)

    def apply_pending_references(self, request: str, consume: bool = True) -> str:
        if not self.pending_references:
            return request
        references = " ".join(f"@{path}" for path in self.pending_references)
        if consume:
            self.pending_references.clear()
        return f"{request}\n\nExplicit file references: {references}"

    def clear(self) -> None:
        self.summary = ""
        self.history.clear()
        self.compaction_backlog.clear()
        self.files_read.clear()
        self.files_modified.clear()
        self.pending_references.clear()
        self.input_tokens = 0
        self.output_tokens = 0
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "repository": self.repository,
            "model": self.model,
            "summary": self.summary,
            "history": self.history,
            "compaction_backlog": self.compaction_backlog,
            "files_read": sorted(self.files_read),
            "files_modified": sorted(self.files_modified),
            "pending_references": self.pending_references,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationSession":
        session_id = str(data.get("session_id") or "")
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid saved session id")
        history = data.get("history") or []
        backlog = data.get("compaction_backlog") or []
        if not isinstance(history, list) or not isinstance(backlog, list):
            raise ValueError("Invalid saved session history")
        return cls(
            session_id=session_id,
            repository=str(data.get("repository") or ""),
            model=str(data.get("model") or ""),
            summary=str(data.get("summary") or ""),
            history=[dict(message) for message in history if isinstance(message, dict)],
            compaction_backlog=[
                dict(message) for message in backlog if isinstance(message, dict)
            ],
            files_read=set(data.get("files_read") or []),
            files_modified=set(data.get("files_modified") or []),
            pending_references=list(data.get("pending_references") or []),
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


class SessionStore:
    """Repository-scoped JSON session persistence, similar to Codex resume."""

    def __init__(
        self,
        repository: str | Path,
        base_directory: str | Path | None = None,
    ) -> None:
        root = Path(repository).resolve()
        project_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        default_base = Path(
            os.getenv("CODESMITH_HOME", str(Path.home() / ".codesmith"))
        ) / "sessions"
        self.directory = Path(base_directory or default_base) / project_key

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid session id")
        return self.directory / f"{session_id}.json"

    def save(self, session: ConversationSession) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(session.session_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, session_id: str) -> ConversationSession:
        path = self._path(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Unknown session: {session_id}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load session {session_id}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Invalid session file: {session_id}")
        return ConversationSession.from_dict(data)

    def list(self) -> list[ConversationSession]:
        if not self.directory.is_dir():
            return []
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                sessions.append(self.load(path.stem))
            except ValueError:
                continue
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)

    def latest(self) -> ConversationSession | None:
        sessions = self.list()
        return sessions[0] if sessions else None
