# CodeSmith Coding Agent Specification

**Version:** 0.2  
**Status:** Active development

This document defines the engineering standards for evolving CodeSmith into a
full Python command-line coding agent. Existing working behavior MUST be
preserved unless a migration is explicitly planned.

The normative terms **MUST**, **SHOULD**, and **MAY** mean required,
recommended, and optional respectively.

## 1. Product scope

CodeSmith MUST be able to inspect a repository, search and read source code,
reason about tasks, plan changes, edit files, run development commands, run
tests, analyze failures, iterate on fixes, inspect Git changes, and explain its
work.

The core loop is:

```text
User request -> inspect -> reason -> select tool -> execute -> observe
-> modify -> verify -> fix if necessary -> final response
```

Vector databases, RAG, multi-agent execution, autonomous Git push/deployment,
web interfaces, and IDE extensions are out of scope for the MVP.

## 2. Engineering standards

- Python MUST remain the primary implementation language.
- New code SHOULD target Python 3.11+.
- Public interfaces MUST use type annotations.
- Components MUST have clear, single responsibilities.
- Existing working components SHOULD be extended instead of rewritten.
- Changes SHOULD be minimal and targeted.
- Secrets MUST be loaded from environment variables or local configuration and
  MUST never be committed.
- Errors MUST become useful errors or agent observations; they MUST NOT be
   silently ignored.
- The model provider MUST remain replaceable.

Recommended technologies are Typer, Rich, httpx, Pydantic, pytest, asyncio,
ripgrep, and subprocess-based Git integration. Existing dependencies MAY be
retained when they already satisfy the requirement.

## 3. Architecture

The implementation SHOULD preserve these boundaries:

```text
CLI and terminal renderer
            |
            v
Agent runtime, state, context, permissions
            |
            v
Model provider
            |
            v
Tool registry
            |
            v
Filesystem, search, terminal, and Git tools
```

The runtime MUST own orchestration, conversation state, tool results,
iteration limits, completion, and stop reasons. CLI code MUST own argument
parsing and rendering, not the agent loop.

## 4. CLI standards

The CLI MUST support interactive and one-shot usage:

```text
codesmith
codesmith "request"
codesmith --debug
codesmith --auto "request"
codesmith init
codesmith review
codesmith fix "request"
codesmith explain "request"
```

All supported `codesmith` command forms MUST invoke the same repository-aware
agent runtime.

The terminal UI SHOULD distinguish model activity, tool execution, file
changes, command execution, test results, and final status.

## 5. Agent runtime

Agent state MUST explicitly contain at least:

```text
user_request, messages, tool_calls, tool_results, files_read,
files_modified, iteration, max_iterations, completed, final_response,
stop_reason
```

The default maximum MUST be 20 iterations and MUST be configurable. The loop
MUST continue after both successful and failed tool calls. It MUST stop when
the task is complete, the user cancels, the iteration limit is reached, a
safety policy blocks required work, or invalid model calls repeat.

Complex tasks SHOULD use a short plan. Trivial edits SHOULD NOT require a
formal plan.

## 6. Model provider

Providers MUST implement a model-independent interface equivalent to:

```python
async def generate(messages: list, tools: list) -> ModelResponse: ...
```

Responses MUST support text, structured tool calls, and provider errors. The
initial local provider is Ollama using `qwen3`, the default tool-capable model.
Provider credentials and endpoints MUST come from configuration or environment
variables.

Providers SHOULD support timeouts, safe retries, streaming where available,
and structured tool-call validation.

## 7. Tool standards

Tools MUST implement a common interface and be registered centrally. The MVP
tool set is:

```text
list_files
read_file
search_code
write_file
run_command
git_status
git_diff
```

Tool results MUST report success or failure and human-readable output.
Command results SHOULD include the command, exit code, stdout, stderr, and
duration.

`list_files` MUST support depth limits, hidden-file handling, ignore patterns,
and practical `.gitignore` behavior. `read_file` MUST support line ranges,
binary detection, and output limits. `search_code` SHOULD use ripgrep and
return file names, line numbers, matching lines, and context.

Existing-file changes SHOULD use targeted patches. Whole-file replacement
MUST be avoided unless explicitly appropriate.

## 8. Safety and permissions

Safety MUST be enforced independently inside the tool layer. Every operation
MUST be classified as:

```text
SAFE       execute automatically
CONFIRM    require user approval unless explicitly authorized by --auto
BLOCKED    never execute
```

Default classifications:

| Operation | Permission |
|---|---|
| Read, search, list, Git inspection | SAFE |
| Write, edit, delete | CONFIRM |
| Package installation, Docker, unknown shell commands | CONFIRM |
| Destructive system commands | BLOCKED |

Model-supplied paths MUST resolve beneath the active repository. Traversal,
external absolute paths, and symlink escapes MUST be rejected. Commands MUST
run from the repository root, have timeout/output limits, and be checked by an
independent command policy.

Automatic commits, pushes, deployments, and infrastructure changes are not
MVP features.

## 9. Context and repository awareness

The entire repository MUST NOT be placed into the model context automatically.
Context priority is:

```text
request -> task state -> recent tool results -> relevant source
-> relevant configuration -> Git state
```

The system SHOULD detect `pyproject.toml`, `requirements.txt`, `pom.xml`,
`build.gradle`, `package.json`, `go.mod`, and `Cargo.toml`.

Large files MUST be read by relevant line ranges or search results. Context
limits MUST be configurable. Project rules MAY be loaded from
`.codesmith/rules.md`, with configuration in `.codesmith/config.yaml` or an equivalent
existing mechanism.

## 10. Verification and reporting

The agent MUST distinguish generated code from verified code. When repository
conventions make it possible, it SHOULD run tests, builds, type checks, and
linters after modifications.

The final response MUST report:

- changed files and a concise summary;
- verification commands and their results;
- unresolved failures or the reason execution stopped;
- whether unrelated files were modified.

The agent MUST NOT claim success without evidence when verification was
possible.

## 11. Testing standards

Unit tests MUST cover path security, tool validation, file operations, search,
command policy, permissions, state transitions, and context limits.

Integration tests MUST cover:

```text
request -> model tool call -> tool result -> continued model call
-> final response
```

An end-to-end fixture SHOULD contain a deliberately failing implementation
that the agent can inspect, modify, test, and report as passing.

## 12. Definition of done

The first production-quality milestone is complete when:

- interactive and one-shot CLI modes work;
- a replaceable provider can issue tool calls;
- repository discovery, search, reading, writing, terminal, and Git tools work;
- the agent iterates after tool results and test failures;
- permissions, path boundaries, command restrictions, and iteration limits work;
- final output accurately reports changes and verification;
- unit, integration, and end-to-end tests cover the core loop.

Implementation priority MUST be: audit existing code, stabilize interfaces,
complete the agent loop, improve tools and context, add verification, then
improve safety and UX.

## 13. Implementation task tracker

This section is the current source of truth for implementation progress. It
MUST be updated whenever a meaningful capability is completed or its status
changes.

### Completed

- [x] Create this specification and define engineering standards.
- [x] Add structured `AgentState`, `ModelResponse`, and an async agent loop.
- [x] Add a central tool registry with model-facing tool schemas.
- [x] Add repository-scoped file listing, file reading, file writing, and
  ripgrep search tools.
- [x] Add command execution with repository-root execution, timeout, output
  limits, and exit-code reporting.
- [x] Add Git status and Git diff tools.
- [x] Add SAFE, CONFIRM, and BLOCKED permission classifications.
- [x] Reject repository path traversal, external paths, and resolved symlink
  escapes in the new repository tools.
- [x] Block the initial set of destructive system commands.
- [x] Add an Ollama chat provider that parses structured tool calls.
- [x] Add the CodeSmith one-shot and interactive repository-aware CLI entry point.
- [x] Make repository-aware execution the default `codesmith` CLI behavior.
- [x] Add complete MVP tool parameter schemas and basic runtime argument
  validation.
- [x] Convert unknown and malformed tool calls into recoverable observations.
- [x] Add common repository-marker detection and lightweight repository
  metadata discovery.
- [x] Make repository file listing consult Git ignore rules when available.
- [x] Add `.codesmith/config.yaml` loading with defaults for model and agent
  settings.
- [x] Load `.codesmith/rules.md` into the agent system context.
- [x] Add debug rendering for agent iterations, tool calls, and tool results.
- [x] Add focused tests for agent continuation, path security, and command
  blocking.
- [x] Expand terminal rendering to cover model activity, edits, commands,
  tests, and final status in normal (non-debug) mode.
- [x] Declare test dependencies and run the full unit/integration test suite.
- [x] Add context management with prioritization, large-file limits, explicit
  `@file` references, repository metadata summaries, and bounded multi-turn
  interactive history.
- [x] Add OpenRouter-compatible Codex-style layered instructions, token-aware
  context compaction, resumable sessions, context commands, reasoning-detail
  replay, and provider-reported usage accounting.

### Not started

- [ ] Add targeted patch/edit support for existing files.
- [ ] Add automatic test/build/type-check/lint selection based on repository
  conventions.
- [ ] Add an iterative test-failure repair workflow with verification evidence.
- [ ] Add invalid-tool-call recovery and user cancellation handling throughout
  the runtime and CLI.
- [ ] Add `init` and fully specified `review`, `fix`, and `explain` command
  workflows.
- [ ] Add end-to-end fixture-repository coverage for a real fix task.
- [ ] Add final change summaries that include Git status, Git diff, and
  unresolved failures.
