import asyncio
import io

import pytest

from codesmith import runtime_cli
from codesmith.agent import AgentRuntime, ModelResponse, default_registry
from codesmith.context import discover_repository, load_rules
from codesmith.llm import OpenRouterChatProvider, OpenRouterError
from codesmith.runtime_cli import TerminalRenderer
from codesmith.tools import Permission, RepositoryTools


class FakeModel:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(tool_calls=[{"name": "read_file", "arguments": {"path": "hello.txt"}}])
        return ModelResponse(content="Inspected the file successfully.")


def test_agent_continues_after_tool_result(tmp_path):
    (tmp_path / "hello.txt").write_text("hello\n")
    runtime = AgentRuntime(FakeModel(), default_registry(RepositoryTools(tmp_path)))
    state = asyncio.run(runtime.run("Inspect hello.txt"))

    assert state.completed
    assert state.final_response == "Inspected the file successfully."
    assert state.files_read == {"hello.txt"}
    assert state.iteration == 2


def test_runtime_emits_model_and_detailed_tool_events(tmp_path):
    (tmp_path / "hello.txt").write_text("hello\n")
    events = []
    runtime = AgentRuntime(
        FakeModel(),
        default_registry(RepositoryTools(tmp_path)),
        event_handler=events.append,
    )

    asyncio.run(runtime.run("Inspect hello.txt"))

    assert [event["event"] for event in events].count("model_start") == 2
    result = next(event for event in events if event["event"] == "tool_result")
    assert result["arguments"] == {"path": "hello.txt"}
    assert result["metadata"]["path"].endswith("hello.txt")
    assert events[-1]["event"] == "completed"


def test_runtime_preserves_openrouter_reasoning_details_for_tool_continuation(tmp_path):
    reasoning_details = [
        {"type": "reasoning.encrypted", "data": "opaque", "format": "test"}
    ]

    class ReasoningModel:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=[
                        {"id": "read", "name": "read_file", "arguments": {"path": "a.txt"}}
                    ],
                    reasoning_details=reasoning_details,
                )
            assistant = next(
                message
                for message in messages
                if message.get("role") == "assistant" and message.get("tool_calls")
            )
            assert assistant["reasoning_details"] == reasoning_details
            return ModelResponse(content="done")

    (tmp_path / "a.txt").write_text("hello")
    state = asyncio.run(
        AgentRuntime(
            ReasoningModel(), default_registry(RepositoryTools(tmp_path))
        ).run("Read a.txt")
    )

    assert state.final_response == "done"


def test_runtime_injects_compacted_summary_as_system_context(tmp_path):
    class CapturingModel:
        async def generate(self, messages, tools):
            assert any(
                message.get("role") == "system"
                and "Earlier objective" in message.get("content", "")
                for message in messages
            )
            return ModelResponse(content="continued")

    state = asyncio.run(
        AgentRuntime(
            CapturingModel(), default_registry(RepositoryTools(tmp_path))
        ).run(
            "Continue",
            system_prompt="Base",
            context_summary="Earlier objective and completed work.",
        )
    )

    assert state.final_response == "continued"


def test_openrouter_provider_sends_session_id_and_records_usage(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            }

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        return Response()

    monkeypatch.setattr("codesmith.llm.requests.post", fake_post)
    provider = OpenRouterChatProvider(
        "test-key",
        "openai/test-model",
        session_id="session_1234",
    )

    response = provider._generate([{"role": "user", "content": "hello"}], [])

    assert captured["payload"]["session_id"] == "session_1234"
    assert response.usage["prompt_tokens"] == 11


def test_openrouter_429_does_not_assume_free_quota_is_exhausted():
    class Response:
        status_code = 429
        text = ""

        def raise_for_status(self):
            import requests

            raise requests.exceptions.HTTPError("429")

        def json(self):
            return {"error": {"code": 429, "message": "Provider returned error"}}

    with pytest.raises(OpenRouterError, match="OpenRouter rate limit reached") as error:
        OpenRouterChatProvider._raise_for_status(Response())

    assert "free-model" not in str(error.value)


def test_model_picker_only_includes_tool_capable_free_models(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "nvidia/safety:free",
                        "name": "Safety classifier",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": [],
                    },
                    {
                        "id": "nvidia/coder:free",
                        "name": "Coding model",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": ["tools"],
                        "context_length": 131_072,
                    },
                    {
                        "id": "google/preview-coder",
                        "name": "Preview coder",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": ["tools"],
                    },
                ]
            }

    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        return Response()

    monkeypatch.setattr(runtime_cli, "_FREE_MODELS_CACHE", None)
    monkeypatch.setattr(runtime_cli.requests, "get", fake_get)
    args = type(
        "Args",
        (),
        {"url": "https://openrouter.ai/api/v1", "api_key": "key", "timeout": 30},
    )()

    models = runtime_cli._free_models(args)
    model_ids = {model["id"] for model in models}

    assert "supported_parameters=tools" in captured["url"]
    assert "nvidia/safety:free" not in model_ids
    assert "nvidia/coder:free" in model_ids
    assert "google/preview-coder" not in model_ids
    assert "openrouter/free" in model_ids


def test_normal_renderer_reports_edits_tests_and_final_status():
    output = io.StringIO()
    renderer = TerminalRenderer(stream=output)

    renderer({"event": "model_start", "iteration": 1})
    renderer({"event": "model_result", "iteration": 1, "tool_call_count": 1})
    renderer({"event": "tool_call", "name": "write_file", "arguments": {"path": "app.py"}})
    renderer({"event": "tool_result", "name": "write_file", "arguments": {"path": "app.py"}, "ok": True})
    renderer({"event": "tool_call", "name": "run_command", "arguments": {"command": "python -m pytest -q"}})
    renderer({
        "event": "tool_result",
        "name": "run_command",
        "arguments": {"command": "python -m pytest -q"},
        "ok": True,
        "metadata": {"exit_code": 0, "duration_ms": 42},
    })
    renderer({"event": "completed", "iteration": 2})

    rendered = output.getvalue()
    assert "Model is reasoning" in rendered
    assert "Editing app.py" in rendered
    assert "Updated app.py" in rendered
    assert "Running tests: python -m pytest -q" in rendered
    assert "Tests passed (exit 0, 42 ms)" in rendered
    assert "Completed in 2 iterations" in rendered


def test_path_traversal_is_rejected(tmp_path):
    result = RepositoryTools(tmp_path).read_file("../secret.txt")
    assert not result.ok
    assert "outside repository" in result.output


def test_dangerous_command_is_blocked(tmp_path):
    result = RepositoryTools(tmp_path).run_command("rm -rf /")
    assert not result.ok
    assert result.permission == Permission.BLOCKED


def test_git_status_is_repository_scoped(tmp_path):
    result = RepositoryTools(tmp_path).git_status()
    assert result.metadata["command"] == "git status --short"


def test_tool_arguments_are_validated(tmp_path):
    registry = default_registry(RepositoryTools(tmp_path))
    result = asyncio.run(registry.execute("read_file", {}))
    assert not result.ok
    assert "required" in result.output


def test_unknown_tool_becomes_observation(tmp_path):
    registry = default_registry(RepositoryTools(tmp_path))
    result = asyncio.run(registry.execute("does_not_exist", {}))
    assert not result.ok
    assert "Unknown tool" in result.output


def test_repository_markers_are_discovered(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    metadata = discover_repository(tmp_path)
    assert metadata.projects == {"python": ["pyproject.toml"]}


def test_project_rules_are_loaded(tmp_path):
    rules_dir = tmp_path / ".codesmith"
    rules_dir.mkdir()
    (rules_dir / "rules.md").write_text("Use pytest.\n")
    assert load_rules(tmp_path) == "Use pytest.\n"
