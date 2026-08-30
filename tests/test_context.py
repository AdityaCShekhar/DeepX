import asyncio
import json
from types import SimpleNamespace

import pytest

from codesmith.agent import AgentRuntime, ModelResponse, default_registry
from codesmith.context import (
    ContextLimits,
    ContextManager,
    ConversationSession,
    SessionStore,
    estimate_tokens,
)
from codesmith.tools import RepositoryTools


def _state(**overrides):
    values = {
        "iteration": 3,
        "max_iterations": 20,
        "files_read": {"src/app.py"},
        "files_modified": {"tests/test_app.py"},
        "tool_calls": [{}, {}],
        "tool_results": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tool_unit(identifier: str, observation: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": identifier,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": identifier, "content": observation},
    ]


def test_system_context_summarizes_repository_and_bounds_referenced_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "large.py").write_text("value = 1\n" * 200)
    manager = ContextManager(
        tmp_path,
        {"max_chars": 6000, "max_message_chars": 2000, "max_file_chars": 120},
    )

    prompt = manager.build_system_prompt("Base instructions")
    prepared = manager.prepare_messages(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Explain @large.py"},
        ],
        _state(),
    )
    serialized = json.dumps(prepared)

    assert "python (pyproject.toml)" in prompt
    assert "Referenced file @large.py" in serialized
    assert "truncated" in serialized
    assert len(prompt) <= 2000


def test_referenced_file_cannot_escape_repository(tmp_path):
    manager = ContextManager(tmp_path)

    prepared = manager.prepare_messages(
        [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Explain @../secret.txt"},
        ],
        _state(),
    )

    assert "path is outside the repository" in json.dumps(prepared)


def test_message_preparation_keeps_priorities_and_newest_complete_tool_unit(tmp_path):
    manager = ContextManager(
        tmp_path,
        {
            "max_chars": 1400,
            "max_message_chars": 450,
            "max_tool_result_chars": 300,
            "max_file_chars": 200,
        },
    )
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "original request"},
        *_tool_unit("old", "old-observation-" * 20),
        *_tool_unit("middle", "middle-observation-" * 20),
        *_tool_unit("new", "new-observation-" * 20),
    ]

    prepared = manager.prepare_messages(messages, _state())
    serialized = json.dumps(prepared)

    assert "system instructions" in serialized
    assert "original request" in serialized
    assert "Current task state" in serialized
    assert "new-observation" in serialized
    assert "old-observation" not in serialized
    tool_ids = {
        message.get("tool_call_id") for message in prepared if message["role"] == "tool"
    }
    assistant_ids = {
        call["id"]
        for message in prepared
        for call in message.get("tool_calls", [])
    }
    assert "new" in tool_ids
    assert "old" not in tool_ids
    assert tool_ids == assistant_ids
    assert sum(manager._message_size(message) for message in prepared) <= 1400
    assert manager.messages_tokens(prepared) <= manager.limits.max_tokens


def test_historical_write_arguments_remain_valid_json_after_compaction(tmp_path):
    manager = ContextManager(tmp_path, {"max_tool_result_chars": 600})
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "app.py", "content": "x" * 5000}),
                },
            }
        ],
    }

    compacted = manager._compact_message(message)
    arguments = compacted["tool_calls"][0]["function"]["arguments"]

    assert json.loads(arguments)["path"] == "app.py"
    assert len(arguments) <= 300


def test_repository_read_file_has_a_character_limit(tmp_path):
    (tmp_path / "large.txt").write_text("long line\n" * 100)

    result = RepositoryTools(tmp_path, max_read_chars=120).read_file("large.txt")

    assert result.ok
    assert len(result.output) <= 120
    assert "truncated" in result.output
    assert result.metadata["truncated"] is True
    assert result.metadata["total_bytes"] == 1000


def test_invalid_context_limit_is_rejected():
    with pytest.raises(ValueError, match="context.max_chars"):
        ContextLimits.from_config({"max_chars": 0})


def test_runtime_uses_prepared_task_context(tmp_path):
    class CapturingModel:
        def __init__(self):
            self.messages = None

        async def generate(self, messages, tools):
            self.messages = messages
            return ModelResponse(content="done")

    model = CapturingModel()
    manager = ContextManager(tmp_path)
    runtime = AgentRuntime(
        model,
        default_registry(RepositoryTools(tmp_path)),
        message_preparer=manager.prepare_messages,
    )

    state = asyncio.run(runtime.run("Inspect the project", system_prompt="System"))

    assert state.completed
    assert any("Current task state" in message["content"] for message in model.messages)


def test_conversation_session_carries_question_and_answer_to_next_request(tmp_path):
    class CapturingModel:
        def __init__(self):
            self.calls = []

        async def generate(self, messages, tools):
            self.calls.append(messages)
            answer = "Authentication uses tokens." if len(self.calls) == 1 else "Tests added."
            return ModelResponse(content=answer)

    model = CapturingModel()
    manager = ContextManager(tmp_path)
    runtime = AgentRuntime(
        model,
        default_registry(RepositoryTools(tmp_path)),
        message_preparer=manager.prepare_messages,
    )
    session = ConversationSession()

    first = asyncio.run(runtime.run("Explain authentication", system_prompt="System"))
    session.update(first, manager)
    second = asyncio.run(
        runtime.run(
            "Now add tests for it",
            system_prompt="System",
            history=session.history,
            files_read=session.files_read,
            files_modified=session.files_modified,
        )
    )

    second_model_context = json.dumps(model.calls[1])
    assert "Explain authentication" in second_model_context
    assert "Authentication uses tokens." in second_model_context
    assert "Now add tests for it" in second_model_context
    assert second.messages[-1] == {"role": "assistant", "content": "Tests added."}


def test_latest_question_has_priority_in_multi_turn_context(tmp_path):
    manager = ContextManager(
        tmp_path,
        {"max_chars": 1400, "max_message_chars": 450, "max_tool_result_chars": 300},
    )
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]

    prepared = manager.prepare_messages(messages, _state())
    non_system = [message for message in prepared if message["role"] != "system"]

    assert non_system[-1]["content"] == "new question"
    assert any(message.get("content") == "old answer" for message in non_system)


def test_conversation_session_clear_resets_all_context():
    session = ConversationSession(
        history=[{"role": "user", "content": "remember me"}],
        files_read={"app.py"},
        files_modified={"tests.py"},
    )

    session.clear()

    assert session.history == []
    assert session.files_read == set()
    assert session.files_modified == set()


def test_conversation_session_discards_oldest_turns_when_bounded(tmp_path):
    manager = ContextManager(
        tmp_path,
        {"max_chars": 1000, "max_message_chars": 330, "max_tool_result_chars": 300},
    )
    messages = [{"role": "system", "content": "System"}]
    for number in range(4):
        messages.extend(
            [
                {"role": "user", "content": f"question-{number}-" + "q" * 250},
                {"role": "assistant", "content": f"answer-{number}-" + "a" * 250},
            ]
        )
    state = _state(messages=messages)
    state.messages = messages
    session = ConversationSession()

    session.update(state, manager)

    serialized = json.dumps(session.history)
    assert "question-3" in serialized
    assert "answer-3" in serialized
    assert "question-0" not in serialized
    assert sum(manager._message_size(message) for message in session.history) <= 1000


def test_instruction_discovery_layers_global_root_and_nested_overrides(tmp_path):
    project = tmp_path / "project"
    nested = project / "src" / "feature"
    home = tmp_path / "home"
    nested.mkdir(parents=True)
    home.mkdir()
    (home / "AGENTS.md").write_text("Global guidance")
    (project / "AGENTS.md").write_text("Root guidance")
    (nested / "AGENTS.md").write_text("Ignored nested guidance")
    (nested / "AGENTS.override.md").write_text("Nested override")

    manager = ContextManager(
        project,
        working_directory=nested,
        codesmith_home=home,
    )
    sources = manager.instruction_sources()
    prompt = manager.build_system_prompt("Base")

    assert [source.path for source in sources] == [
        str(home / "AGENTS.md"),
        "AGENTS.md",
        "src/feature/AGENTS.override.md",
    ]
    assert prompt.index("Global guidance") < prompt.index("Root guidance")
    assert prompt.index("Root guidance") < prompt.index("Nested override")
    assert "Ignored nested guidance" not in prompt


def test_new_context_configuration_is_token_based():
    limits = ContextLimits.from_config(
        {
            "max_tokens": 2_000,
            "max_message_tokens": 500,
            "max_tool_result_tokens": 300,
        }
    )

    assert limits.max_tokens == 2_000
    assert limits.max_message_tokens == 500
    assert estimate_tokens("short coding context") > 0


def test_semantic_compaction_preserves_recent_turn_and_usage(tmp_path):
    class SummaryModel:
        async def generate(self, messages, tools):
            assert tools == []
            assert "objective and constraints" in messages[0]["content"]
            return ModelResponse(
                content="Objective: add authentication tests. Completed: inspected auth.py.",
                usage={"prompt_tokens": 40, "completion_tokens": 12},
            )

    manager = ContextManager(
        tmp_path,
        {
            "max_tokens": 500,
            "max_message_tokens": 180,
            "max_tool_result_tokens": 128,
            "compact_threshold": 0.5,
            "keep_recent_turns": 1,
        },
    )
    messages = []
    for number in range(3):
        messages.extend(
            [
                {"role": "user", "content": f"question-{number}-" + "q" * 220},
                {"role": "assistant", "content": f"answer-{number}-" + "a" * 220},
            ]
        )
    state = _state(messages=messages)
    state.messages = messages
    session = ConversationSession()
    session.update(state, manager)

    assert session.needs_compaction()
    assert "question-2" in json.dumps(session.history)
    asyncio.run(session.compact(SummaryModel(), manager))

    assert "authentication tests" in session.summary
    assert session.compaction_backlog == []
    assert session.input_tokens == 40
    assert session.output_tokens == 12


def test_session_store_round_trip_is_repository_scoped(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    store = SessionStore(repository, base_directory=tmp_path / "sessions")
    session = ConversationSession(
        repository=str(repository),
        model="openai/test-model",
        summary="Remember the current objective.",
        history=[{"role": "user", "content": "Continue"}],
        files_read={"app.py"},
    )

    path = store.save(session)
    restored = store.load(session.session_id)

    assert path.is_file()
    assert restored.session_id == session.session_id
    assert restored.summary == session.summary
    assert restored.history == session.history
    assert restored.files_read == {"app.py"}
    assert store.latest().session_id == session.session_id


def test_queued_mention_is_validated_and_applied_once(tmp_path):
    (tmp_path / "app.py").write_text("print('ok')\n")
    manager = ContextManager(tmp_path)
    session = ConversationSession()

    session.queue_reference("app.py", manager)
    request = session.apply_pending_references("Explain this")

    assert "@app.py" in request
    assert session.apply_pending_references("Next") == "Next"
    with pytest.raises(ValueError, match="outside the repository"):
        session.queue_reference("../secret.txt", manager)


def test_queued_mention_can_be_retained_until_request_succeeds(tmp_path):
    (tmp_path / "app.py").write_text("print('ok')\n")
    manager = ContextManager(tmp_path)
    session = ConversationSession()
    session.queue_reference("app.py", manager)

    request = session.apply_pending_references("Explain this", consume=False)

    assert "@app.py" in request
    assert session.pending_references == ["app.py"]
