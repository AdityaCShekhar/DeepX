"""Small, model-agnostic coding-agent runtime.

This module deliberately contains no terminal rendering.  A provider only has
to implement ``generate(messages, tools)`` and may return either
``ModelResponse`` or a compatible object with ``content`` and ``tool_calls``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol, Sequence

from .tools import Permission, RepositoryTools, ToolRegistry, ToolResult


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_details: Any = None
    usage: Dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    async def generate(self, messages: list, tools: list) -> ModelResponse:
        ...


@dataclass
class AgentState:
    user_request: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    files_read: set = field(default_factory=set)
    files_modified: set = field(default_factory=set)
    iteration: int = 0
    max_iterations: int = 20
    completed: bool = False
    final_response: str | None = None
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class _RepositoryTool:
    def __init__(self, name, description, permission, function, schema):
        self.name, self.description, self.permission = name, description, permission
        self._function, self.schema = function, schema

    async def execute(self, arguments):
        return self._function(**arguments)


def default_registry(repository: RepositoryTools) -> ToolRegistry:
    """Build the MVP tool set from a repository-scoped implementation."""
    specs = [
        ("list_files", "List repository files", Permission.SAFE, repository.list_files, {"type": "object", "properties": {"path": {"type": "string"}, "max_depth": {"type": "integer"}, "include_hidden": {"type": "boolean"}}}),
        ("read_file", "Read a UTF-8 text file", Permission.SAFE, repository.read_file, {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}),
        ("search_code", "Search repository code with ripgrep", Permission.SAFE, repository.search_code, {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "context": {"type": "integer"}}, "required": ["query"]}),
        ("write_file", "Create or replace a repository file", Permission.CONFIRM, repository.write_file, {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        ("run_command", "Run a development command in the repository", Permission.CONFIRM, repository.run_command, {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "auto": {"type": "boolean"}}, "required": ["command"]}),
        ("git_status", "Show repository changes", Permission.SAFE, repository.git_status, {"type": "object", "properties": {}}),
        ("git_diff", "Show the current diff", Permission.SAFE, repository.git_diff, {"type": "object", "properties": {}}),
    ]
    return ToolRegistry(_RepositoryTool(*spec) for spec in specs)


class AgentRuntime:
    def __init__(
        self,
        model: ModelProvider,
        tools: ToolRegistry,
        max_iterations: int = 20,
        event_handler: Callable[[Dict[str, Any]], None] | None = None,
        message_preparer: Callable[[list, AgentState], list] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.event_handler = event_handler
        self.message_preparer = message_preparer

    def _emit(self, event: str, **data: Any) -> None:
        if self.event_handler:
            self.event_handler({"event": event, **data})

    async def run(
        self,
        request: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        files_read: Iterable[str] | None = None,
        files_modified: Iterable[str] | None = None,
        context_summary: str | None = None,
    ) -> AgentState:
        state = AgentState(
            user_request=request,
            max_iterations=self.max_iterations,
            files_read=set(files_read or ()),
            files_modified=set(files_modified or ()),
        )
        if system_prompt:
            state.messages.append({"role": "system", "content": system_prompt})
        if context_summary:
            state.messages.append(
                {
                    "role": "system",
                    "content": "Compacted conversation state:\n" + context_summary,
                }
            )
        for message in history or ():
            # The current repository system prompt is rebuilt for every request;
            # persisted system messages would duplicate or stale those rules.
            if message.get("role") != "system":
                state.messages.append(dict(message))
        state.messages.append({"role": "user", "content": request})

        while not state.completed and state.iteration < state.max_iterations:
            state.iteration += 1
            self._emit("iteration", iteration=state.iteration)
            self._emit("model_start", iteration=state.iteration)
            try:
                model_messages = (
                    self.message_preparer(state.messages, state)
                    if self.message_preparer
                    else state.messages
                )
                response = await self.model.generate(
                    model_messages, self.tools.schemas()
                )
            except BaseException as exc:
                self._emit("model_error", iteration=state.iteration, error=str(exc))
                raise
            usage = getattr(response, "usage", None) or {}
            state.input_tokens += int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            state.output_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            calls = getattr(response, "tool_calls", None) or []
            self._emit(
                "model_result",
                iteration=state.iteration,
                tool_call_count=len(calls),
                has_content=bool(getattr(response, "content", "")),
            )
            if not calls:
                state.final_response = getattr(response, "content", "")
                state.messages.append(
                    {"role": "assistant", "content": state.final_response}
                )
                state.completed = True
                state.stop_reason = "completed"
                self._emit(
                    "completed",
                    iteration=state.iteration,
                    tool_call_count=len(state.tool_calls),
                )
                break

            # Keep the assistant tool-call message in the conversation. OpenRouter
            # uses this message when continuing a reasoning/tool-call turn.
            assistant_message = {"role": "assistant", "content": getattr(response, "content", "")}
            reasoning_details = getattr(response, "reasoning_details", None)
            if reasoning_details is not None:
                assistant_message["reasoning_details"] = reasoning_details
            if calls:
                assistant_message["tool_calls"] = [
                    {"id": call.get("id") or f"call_{index}", "type": "function", "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments") or {}),
                    }}
                    for index, call in enumerate(calls)
                ]
            state.messages.append(assistant_message)

            for call in calls:
                name = call.get("name")
                arguments = call.get("arguments") or {}
                self._emit("tool_call", name=name, arguments=arguments)
                if not isinstance(arguments, dict):
                    result = ToolResult(False, "Tool arguments must be an object")
                else:
                    result = await self.tools.execute(name, arguments)
                state.tool_calls.append(call)
                state.tool_results.append(result)
                self._emit(
                    "tool_result",
                    name=name,
                    arguments=arguments,
                    ok=result.ok,
                    output=result.output,
                    permission=result.permission.value,
                    metadata=result.metadata or {},
                )
                if name == "read_file" and result.ok:
                    state.files_read.add(arguments.get("path", ""))
                if name == "write_file" and result.ok:
                    state.files_modified.add(arguments.get("path", ""))
                tool_message = {"role": "tool", "content": result.output}
                if call.get("id"):
                    tool_message["tool_call_id"] = call["id"]
                else:
                    # Retain the name for providers that use the older tool format.
                    tool_message["name"] = name
                state.messages.append(tool_message)

        if not state.completed:
            state.stop_reason = "max_iterations"
            state.final_response = "Agent stopped after reaching the maximum iteration limit."
            state.messages.append(
                {"role": "assistant", "content": state.final_response}
            )
            self._emit(
                "stopped",
                reason=state.stop_reason,
                iteration=state.iteration,
                tool_call_count=len(state.tool_calls),
            )
        return state
