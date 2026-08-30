"""CodeSmith repository-aware coding-agent CLI."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, TextIO

import requests
from colorama import just_fix_windows_console
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from .agent import AgentRuntime, default_registry
from .config import load_config
from .context import ContextManager, ConversationSession, SessionStore
from .llm import DEFAULT_MODEL, OpenRouterChatProvider, OpenRouterError
from .tools import RepositoryTools


# Enable ANSI styling on Windows consoles. On Unix-like systems this is a no-op.
just_fix_windows_console()


FALLBACK_FREE_MODELS = [
    {
        "id": "openai/gpt-oss-20b:free",
        "name": "OpenAI: gpt-oss-20b (free)",
        "description": "Coding, reasoning, tool use, function calling, and structured outputs",
        "context": "131K",
    },
]
_FREE_MODELS_CACHE = None


DEFAULT_AGENT_PROMPT = """You are CodeSmith, a repository-aware coding agent.
Answer questions about the current project by inspecting the repository with the
provided tools. For project summaries, reviews, and explanations, first use
list_files and read relevant files such as README.md, pyproject.toml, and the
main source files. Do not tell the user how to call tools and do not claim that
you cannot inspect the repository when a suitable tool is available. Use a tool
call whenever repository facts are needed, then give a concise answer based on
the tool results. If the user asks to write, create, implement, or save code,
that is explicit permission to modify the repository: use write_file and create
a sensible filename when none is provided (for example, quick_sort.py for a
quick-sort request). After writing, confirm the exact file path and summarize
what was added. Do not modify files for questions, explanations, or reviews.
Format final answers for terminal readability: use a short heading, bullets for
multiple items, short paragraphs, and concise summaries.
"""


def _format_response(response: str) -> str:
    """Format model Markdown into a readable terminal response."""
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))
    output = []
    in_code_block = False

    def style_inline(value: str) -> str:
        # ANSI styling keeps the CLI dependency-free and works in Docker and
        # regular terminals. Markdown markers themselves are not displayed.
        value = re.sub(r"(\*\*|__)(.*?)(\1)", r"\033[1m\2\033[0m", value)
        value = re.sub(r"(?<!`)`([^`]+)`(?!`)", r"\033[36m\1\033[0m", value)
        return value

    for raw_line in response.strip().splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                output.append("\033[2m┌─ code ─────────────────────────────────────┐\033[0m")
            else:
                output.append("\033[2m└───────────────────────────────────────────┘\033[0m")
            continue
        if in_code_block:
            output.append("\033[36m│ " + line + "\033[0m")
            continue
        if not line.strip():
            output.append(line)
            continue

        # Convert numbered Markdown lists into a consistent terminal bullet.
        line = re.sub(r"^\s*\d+[.)]\s+", "• ", line)
        if line.startswith("• ") or line.startswith("- ") or line.startswith("* "):
            bullet = "• "
            content = line[2:].strip()
            # Wrap the unstyled text so ANSI escape sequences do not affect
            # the line-length calculation.
            plain_content = re.sub(r"(\*\*|__|`)", "", content)
            wrapped = textwrap.wrap(plain_content, width=width - 4) or [""]
            output.append(bullet + style_inline(wrapped[0]))
            output.extend("  " + style_inline(part) for part in wrapped[1:])
        elif line.startswith("#"):
            output.append("\033[1m" + style_inline(line.lstrip("# ").strip()) + "\033[0m")
        else:
            output.extend(style_inline(part) for part in (textwrap.wrap(line, width=width) or [""]))

    return "\n".join(output).strip()


def _confirmation(message: str) -> bool:
    try:
        answer = input(f"Approve {message}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _request_for_command(command: str, value: str | None) -> str:
    if command == "review":
        return "Review the current repository changes and report actionable issues."
    if command == "fix":
        return f"Fix this repository task: {value or ''}"
    if command == "explain":
        return f"Explain this repository task or behavior: {value or ''}"
    return value or "Inspect this repository and summarize its structure."


def _tool_status(name: str, arguments: dict) -> str:
    if name == "read_file":
        return f"Reading {arguments.get('path', 'file')}"
    if name == "list_files":
        return f"Listing files in {arguments.get('path', '.')}"
    if name == "search_code":
        return f"Searching for {arguments.get('query', 'code')}"
    if name == "write_file":
        return f"Editing {arguments.get('path', 'file')}"
    if name == "run_command":
        command = arguments.get("command", "")
        action = "Running tests" if _is_test_command(command) else "Running command"
        return f"{action}: {command}"
    if name == "git_status":
        return "Checking git status"
    if name == "git_diff":
        return "Reading git diff"
    return f"Using {name}"


def _is_test_command(command: str) -> bool:
    """Return whether a shell command invokes a conventional test runner."""
    normalized = " ".join(command.lower().split())
    patterns = (
        r"(^|[;&|]\s*)(py\.test|pytest|tox|nox)(\s|$)",
        r"(^|[;&|]\s*)python3?\s+-m\s+(pytest|unittest)(\s|$)",
        r"(^|[;&|]\s*)(npm|pnpm|yarn)\s+(run\s+)?test(\s|$)",
        r"(^|[;&|]\s*)cargo\s+test(\s|$)",
        r"(^|[;&|]\s*)go\s+test(\s|$)",
        r"(^|[;&|]\s*)(mvn|gradle|\./gradlew)\s+.*\btest\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


class TerminalRenderer:
    """Render agent events in both normal and debug terminal modes."""

    def __init__(
        self,
        debug: bool = False,
        show_work: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.debug = debug
        self.show_work = show_work
        self.stream = stream or sys.stdout
        self.activity_visible = False
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def _print(self, message: str = "", *, end: str = "\n") -> None:
        print(message, end=end, file=self.stream, flush=True)

    def clear_activity(self) -> None:
        if self.activity_visible and self.is_tty:
            self._print("\r\033[2K", end="")
        self.activity_visible = False

    @staticmethod
    def _result_label(name: str, arguments: dict[str, Any], ok: bool) -> str:
        marker = "✓" if ok else "✗"
        path = arguments.get("path", "file")
        if name == "read_file":
            return f"{marker} {'Read' if ok else 'Could not read'} {path}"
        if name == "list_files":
            return f"{marker} {'Listed files' if ok else 'Could not list files'}"
        if name == "search_code":
            return f"{marker} {'Search complete' if ok else 'Search failed'}"
        if name == "write_file":
            return f"{marker} {'Updated' if ok else 'Edit failed for'} {path}"
        if name == "git_status":
            return f"{marker} {'Git status checked' if ok else 'Git status failed'}"
        if name == "git_diff":
            return f"{marker} {'Git diff inspected' if ok else 'Git diff failed'}"
        return f"{marker} {name} {'completed' if ok else 'failed'}"

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "iteration":
            if self.debug:
                self._print(f"\n[iteration {event.get('iteration', '?')}]")
            return

        if kind == "model_start":
            if not self.show_work:
                return
            if self.is_tty:
                self._print("\r\033[2KModel is reasoning...", end="")
                self.activity_visible = True
            else:
                self._print("Model is reasoning...")
            return

        if kind == "model_result":
            self.clear_activity()
            if self.debug:
                count = event.get("tool_call_count", 0)
                self._print(f"Model response received ({count} tool call{'s' if count != 1 else ''})")
            return

        if kind == "model_error":
            self.clear_activity()
            self._print("✗ Model request failed")
            return

        if kind == "tool_call":
            self.clear_activity()
            if self.show_work:
                arguments = event.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                self._print(f"→ {_tool_status(str(event.get('name') or 'tool'), arguments)}")
            return

        if kind == "tool_result":
            if not self.show_work and not self.debug:
                return
            name = str(event.get("name") or "tool")
            arguments = event.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            ok = bool(event.get("ok"))
            metadata = event.get("metadata") or {}
            if name == "run_command":
                command = str(arguments.get("command", ""))
                is_test = _is_test_command(command)
                subject = "Tests" if is_test else "Command"
                outcome = "passed" if ok and is_test else "completed" if ok else "failed"
                details = []
                if metadata.get("exit_code") is not None:
                    details.append(f"exit {metadata['exit_code']}")
                if metadata.get("duration_ms") is not None:
                    details.append(f"{metadata['duration_ms']} ms")
                suffix = f" ({', '.join(details)})" if details else ""
                self._print(f"{'✓' if ok else '✗'} {subject} {outcome}{suffix}")
            else:
                self._print(self._result_label(name, arguments, ok))
            if self.debug and event.get("output"):
                self._print(str(event["output"])[:500])
            return

        if kind == "completed":
            self.clear_activity()
            iterations = event.get("iteration", 0)
            noun = "iteration" if iterations == 1 else "iterations"
            self._print(f"✓ Completed in {iterations} {noun}")
            return

        if kind == "stopped":
            self.clear_activity()
            reason = str(event.get("reason", "unknown reason")).replace("_", " ")
            self._print(f"✗ Stopped: {reason}")
            return

        if kind == "context_compaction_start":
            self.clear_activity()
            self._print("→ Compacting older conversation context")
            return

        if kind == "context_compaction_result":
            marker = "✓" if event.get("ok") else "!"
            method = event.get("method", "model summary")
            self._print(f"{marker} Context compacted ({method})")


def _free_models(args: argparse.Namespace) -> list[dict]:
    """Load all currently free models from OpenRouter, with a safe fallback."""
    global _FREE_MODELS_CACHE
    if _FREE_MODELS_CACHE is not None:
        return _FREE_MODELS_CACHE
    try:
        response = requests.get(
            f"{args.url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=min(args.timeout, 30),
        )
        response.raise_for_status()
        models = []
        for model in response.json().get("data", []):
            model_id = model.get("id", "")
            pricing = model.get("pricing") or {}
            try:
                is_free = model_id.endswith(":free") or (
                    pricing and float(pricing.get("prompt", 1)) == 0
                    and float(pricing.get("completion", 1)) == 0
                )
            except (TypeError, ValueError):
                is_free = model_id.endswith(":free")
            if not is_free:
                continue
            parameters = model.get("supported_parameters") or []
            models.append({
                "id": model_id,
                "name": model.get("name") or model_id,
                "description": "Tool calling supported" if "tools" in parameters else "Free model",
                "context": _context_label(model.get("context_length")),
            })
        _FREE_MODELS_CACHE = sorted(models, key=lambda item: item["name"].lower())
    except (requests.exceptions.RequestException, ValueError, TypeError, KeyError):
        print("Could not fetch the OpenRouter model catalog; using the fallback list.")
        _FREE_MODELS_CACHE = FALLBACK_FREE_MODELS
    return _FREE_MODELS_CACHE


def _context_label(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "unknown"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    return f"{value / 1_000:g}K"


def _show_models(current_model: str, models: list[dict]) -> None:
    """Print all free models in a Codex-like picker."""
    print(f"\nAvailable free OpenRouter models ({len(models)}):\n")
    for index, model in enumerate(models, start=1):
        active = " (active)" if model["id"] == current_model else ""
        print(f"  {index}. {model['name']}{active}")
        print(f"     {model['id']}")
        print(f"     {model['description']} · {model['context']} context\n")


def _select_model(args: argparse.Namespace, session: PromptSession, request: str) -> None:
    """Show model suggestions and optionally select one for this session."""
    choice = request.partition(" ")[2].strip()
    models = _free_models(args)
    _show_models(args.model, models)
    if not choice:
        try:
            choice = session.prompt("Select a model (number, or Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not choice:
        print("Model unchanged.")
        return

    selected = None
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(models):
            selected = models[index]
    else:
        selected = next((model for model in models if model["id"] == choice), None)

    if selected is None:
        print("Unknown model selection. Choose one of the listed numbers or IDs.")
        return
    args.model = selected["id"]
    print(f"Selected model: {selected['name']} ({args.model})")


class _CommandCompleter(Completer):
    """Inline slash-command and model suggestions for the interactive prompt."""

    def __init__(self, models: list[dict]):
        self.models = models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        if " " not in text:
            commands = {
                "/models": "Browse and select an OpenRouter model",
                "/status": "Show session and context usage",
                "/compact": "Summarize older conversation context",
                "/mention": "Attach a file to the next prompt",
                "/resume": "Resume a saved repository session",
                "/new": "Start a new conversation",
                "/clear": "Clear the current conversation",
                "/exit": "Exit CodeSmith",
                "/help": "Show interactive commands",
            }
            word = text
            for command, description in commands.items():
                if command.startswith(word):
                    yield Completion(
                        command,
                        start_position=-len(word),
                        display=command,
                        display_meta=description,
                    )
            return

        command, _, value = text.partition(" ")
        if command != "/models":
            return
        for model in self.models:
            if not value or model["id"].lower().startswith(value.lower()):
                yield Completion(
                    model["id"],
                    start_position=-len(value),
                    display=model["name"],
                    display_meta=model["description"],
                )


async def run_request(
    args: argparse.Namespace,
    request: str,
    session: ConversationSession | None = None,
    session_store: SessionStore | None = None,
) -> int:
    root = Path(args.repository).resolve()
    config = load_config(root)
    context_manager = ContextManager(root, config.get("context"))
    repository = RepositoryTools(
        root,
        confirm=None if args.auto else _confirmation,
        max_read_chars=context_manager.limits.max_file_chars,
    )
    registry = default_registry(repository)
    provider = OpenRouterChatProvider(
        args.api_key,
        args.model,
        args.timeout,
        args.url,
        session_id=session.session_id if session else None,
    )
    # Normal mode reports meaningful progress too; debug mode adds raw result
    # previews and iteration details.
    renderer = TerminalRenderer(debug=args.debug, show_work=True)

    system_prompt = context_manager.build_system_prompt(DEFAULT_AGENT_PROMPT)
    effective_request = (
        session.apply_pending_references(request, consume=False) if session else request
    )
    try:
        state = await AgentRuntime(
            provider,
            registry,
            args.max_iterations,
            event_handler=renderer,
            message_preparer=context_manager.prepare_messages,
        ).run(
            effective_request,
            system_prompt=system_prompt,
            history=session.history if session else None,
            files_read=session.files_read if session else None,
            files_modified=session.files_modified if session else None,
            context_summary=session.summary if session else None,
        )
    except OpenRouterError as exc:
        renderer.clear_activity()
        print(f"\nOpenRouter request failed: {exc}")
        if "rate limit" in str(exc).lower() or "429" in str(exc):
            print("Free-model quota is exhausted. Wait for the reset or add OpenRouter credits.")
        else:
            print("Use /models and select a model marked 'Tool calling supported' for repository tasks.")
        return 1
    if session is not None:
        session.pending_references.clear()
        session.repository = str(root)
        session.model = args.model
        session.update(state, context_manager)
        if session.needs_compaction():
            renderer({"event": "context_compaction_start"})
            try:
                await session.compact(provider, context_manager)
                renderer(
                    {
                        "event": "context_compaction_result",
                        "ok": True,
                        "method": "OpenRouter model summary",
                    }
                )
            except (OpenRouterError, ValueError) as exc:
                session.compact_without_model(context_manager)
                renderer(
                    {
                        "event": "context_compaction_result",
                        "ok": False,
                        "method": "local fallback",
                    }
                )
                if args.debug:
                    print(f"Compaction model request failed: {exc}")
        if session_store is not None:
            try:
                session_store.save(session)
            except OSError as exc:
                print(f"Warning: could not save session: {exc}")
    if args.debug:
        print(f"Iterations: {state.iteration}")
        print(f"Tool calls: {len(state.tool_calls)}")
        print(f"Stop reason: {state.stop_reason}")
        print(f"Tokens: {state.input_tokens} input, {state.output_tokens} output")
    print("\n" + "─" * 60)
    print(_format_response(state.final_response or "CodeSmith finished without a final response."))
    return 0 if state.stop_reason == "completed" else 1


async def _compact_conversation(
    args: argparse.Namespace,
    conversation: ConversationSession,
    context_manager: ContextManager,
) -> tuple[bool, str]:
    """Run portable semantic compaction through a normal OpenRouter request."""
    if not conversation.needs_compaction() and not conversation.prepare_for_forced_compaction(
        context_manager
    ):
        return False, "There is not enough conversation context to compact."
    provider = OpenRouterChatProvider(
        args.api_key,
        args.model,
        args.timeout,
        args.url,
        session_id=conversation.session_id,
    )
    try:
        await conversation.compact(provider, context_manager)
        return True, "Conversation context compacted with an OpenRouter model summary."
    except (OpenRouterError, ValueError) as exc:
        conversation.compact_without_model(context_manager)
        return True, f"Conversation context compacted locally ({exc})."


def _show_context_status(
    conversation: ConversationSession,
    context_manager: ContextManager,
) -> None:
    estimated = conversation.estimated_context_tokens(context_manager)
    maximum = context_manager.limits.max_tokens
    percentage = min(100.0, estimated * 100 / maximum) if maximum else 0.0
    print("\nSession status:")
    print(f"  ID: {conversation.session_id}")
    print(f"  Model: {conversation.model or 'not used yet'}")
    print(
        f"  Active context: approximately {estimated:,}/{maximum:,} tokens "
        f"({percentage:.0f}%)"
    )
    print(
        f"  OpenRouter usage: {conversation.input_tokens:,} input, "
        f"{conversation.output_tokens:,} output tokens"
    )
    print(f"  Compacted summary: {'yes' if conversation.summary else 'no'}")
    print(f"  Files read: {len(conversation.files_read)}")
    print(f"  Files modified: {len(conversation.files_modified)}\n")


def _resume_conversation(
    store: SessionStore,
    prompt_session: PromptSession,
    request: str,
) -> ConversationSession | None:
    session_id = request.partition(" ")[2].strip()
    sessions = store.list()
    if not sessions:
        print("No saved sessions exist for this repository.")
        return None
    if not session_id:
        print("\nRecent sessions:")
        for saved in sessions[:10]:
            summary = saved.summary.replace("\n", " ")[:60] or "no compacted summary"
            print(f"  {saved.session_id}  {saved.updated_at}  {summary}")
        try:
            session_id = prompt_session.prompt(
                "Session ID (Enter for most recent): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not session_id:
            return sessions[0]
    try:
        return store.load(session_id)
    except ValueError as exc:
        print(exc)
        return None


def interactive_loop(args: argparse.Namespace) -> None:
    print("CodeSmith · repository coding agent")
    print("Type /help for commands.\n")
    args.show_work = True
    models = _free_models(args)
    root = Path(args.repository).resolve()
    config = load_config(root)
    context_manager = ContextManager(root, config.get("context"))
    conversation_store = SessionStore(root)
    conversation = ConversationSession(repository=str(root), model=args.model)

    key_bindings = KeyBindings()

    @key_bindings.add("/")
    def _(event):
        """Open slash-command suggestions immediately after typing /."""
        event.current_buffer.insert_text("/")
        event.current_buffer.start_completion(select_first=True)

    @key_bindings.add("enter")
    def _(event):
        """Accept the first visible suggestion when Enter is pressed."""
        buffer = event.current_buffer
        state = buffer.complete_state
        if state and state.completions:
            completion = state.current_completion or state.completions[0]
            buffer.apply_completion(completion)
        buffer.validate_and_handle()

    style = Style.from_dict({
        # Legacy CodeSmith layout, recolored cyan.
        "completion-menu": "bg:#071923 #b8f4ff",
        "completion-menu.completion": "bg:#071923 #b8f4ff",
        "completion-menu.completion.current": "bg:#00a8c6 #001018 bold",
        "completion-menu.meta.completion": "bg:#071923 #ffffff",
        "completion-menu.meta.completion.current": "bg:#00a8c6 #ffffff",
        "scrollbar.background": "bg:#071923",
        "scrollbar.button": "bg:#00c8e8",
        "prompt": "#7defff bold",
        "bottom-toolbar": "bg:#071923 #b8f4ff",
    })

    def bottom_toolbar():
        return (
            f" Model: {args.model}  ·  /status context  ·  /compact  ·  "
            "/new  ·  /help  ·  /exit "
        )

    session = PromptSession(
        history=FileHistory(str(Path.home() / ".codesmith_history")),
        completer=_CommandCompleter(models),
        key_bindings=key_bindings,
        complete_while_typing=True,
        complete_in_thread=False,
        mouse_support=False,
        style=style,
    )
    while True:
        try:
            request = session.prompt(
                FormattedText([("class:prompt", "➜ ")]),
                bottom_toolbar=bottom_toolbar,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if request == "/exit":
            return
        if request == "/help":
            print("\nCommands:")
            print("  /models       Browse and select a free OpenRouter model")
            print("  /status       Show session ID, context, and token usage")
            print("  /compact      Summarize older context with the active model")
            print("  /mention PATH Attach a repository file to the next prompt")
            print("  /resume [ID]  Resume a saved repository session")
            print("  /new          Start a new saved conversation")
            print("  /clear        Clear the current conversation")
            print("  /help         Show this help")
            print("  /exit         Exit CodeSmith\n")
            continue
        if request == "/models" or request.startswith("/models "):
            _select_model(args, session, request)
            conversation.model = args.model
            continue
        if request == "/status":
            _show_context_status(conversation, context_manager)
            continue
        if request == "/compact":
            print("Compacting conversation context...")
            changed, message = asyncio.run(
                _compact_conversation(args, conversation, context_manager)
            )
            print(message)
            if changed:
                try:
                    conversation_store.save(conversation)
                except OSError as exc:
                    print(f"Warning: could not save session: {exc}")
            continue
        if request.startswith("/mention "):
            path = request.partition(" ")[2].strip()
            try:
                conversation.queue_reference(path, context_manager)
                print(f"Attached @{path} to the next prompt.")
            except ValueError as exc:
                print(exc)
            continue
        if request == "/resume" or request.startswith("/resume "):
            resumed = _resume_conversation(conversation_store, session, request)
            if resumed is not None:
                conversation = resumed
                if conversation.model:
                    args.model = conversation.model
                print(f"Resumed session {conversation.session_id}.")
            continue
        if request == "/new":
            try:
                conversation_store.save(conversation)
            except OSError as exc:
                print(f"Warning: could not save session: {exc}")
            conversation = ConversationSession(repository=str(root), model=args.model)
            print(f"Started session {conversation.session_id}.")
            continue
        if request == "/clear":
            conversation.clear()
            try:
                conversation_store.save(conversation)
            except OSError as exc:
                print(f"Warning: could not save session: {exc}")
            print("Conversation context cleared.")
            continue
        if request:
            asyncio.run(
                run_request(
                    args,
                    request,
                    session=conversation,
                    session_store=conversation_store,
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeSmith repository coding agent")
    parser.add_argument("request", nargs="?", help="One-shot task request")
    parser.add_argument("--repository", "-C", default=".", help="Active repository root")
    parser.add_argument("--url", help="OpenRouter API base URL")
    parser.add_argument("--api-key", help="OpenRouter API key (defaults to OPENROUTER_API_KEY)")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, help="OpenRouter request timeout in seconds")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--auto", action="store_true", help="Skip confirmation for confirm-level operations")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-work", action="store_true", help="Show model iterations and repository tool activity")
    subparsers = parser.add_subparsers(dest="command")
    for name in ("review", "fix", "explain"):
        sub = subparsers.add_parser(name)
        sub.add_argument("value", nargs="?")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.repository)
    args.url = args.url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    args.api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    args.model = args.model or config["model"]["model"] or DEFAULT_MODEL
    args.timeout = args.timeout or int(os.getenv("OPENROUTER_TIMEOUT", "600"))
    args.max_iterations = args.max_iterations or config["agent"]["max_iterations"]
    request = _request_for_command(args.command, args.value) if args.command else args.request
    if request:
        raise SystemExit(asyncio.run(run_request(args, request)))
    interactive_loop(args)


if __name__ == "__main__":
    main()
