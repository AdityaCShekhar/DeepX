# CodeSmith

CodeSmith is an OpenRouter-powered coding agent that works inside a repository.
It can inspect files, search code, review Git changes, edit files, and run
development commands through an interactive terminal.

## Features

- Repository-scoped file listing, reading, searching, and writing
- Git status and diff inspection
- Development command and test execution with confirmation for sensitive actions
- OpenRouter reasoning and tool calling
- Interactive workflows and one-shot task shortcuts
- Free, tool-capable OpenRouter model picker
- Token-aware multi-turn context with automatic compaction
- Saved, repository-specific sessions that can be resumed later
- Layered project instructions from `AGENTS.md` files
- macOS, Linux, Windows, and Docker launch options

## Requirements

- Python 3.8 or newer
- An [OpenRouter](https://openrouter.ai/) API key
- Git for status and diff tools
- [ripgrep](https://github.com/BurntSushi/ripgrep) for repository search
- Docker with Compose (optional; the macOS/Linux launcher uses it automatically
  when available)

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/AdityaCShekhar/DeepX.git CodeSmith
cd CodeSmith
python3 -m pip install -e .
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

On PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
```

The editable installation provides `codesmith` and `codesmith-batch` commands.

### Repository launchers

The repository also includes `codesmith` for macOS/Linux and `codesmith.cmd` for
Windows. Add the cloned repository directory to `PATH` if you want to invoke
these launchers from anywhere.

On macOS/Linux, the launcher uses Docker Compose when it is available and falls
back to the local Python source otherwise. The Windows launcher runs the local
Python source.

## Usage

Start an interactive session in the current repository:

```bash
codesmith
```

Target a different repository:

```bash
codesmith -C /path/to/project
```

Use a specific OpenRouter model:

```bash
codesmith --model openrouter/free
codesmith --model openai/gpt-oss-20b:free
```

Convenience task forms are also available:

```bash
codesmith review
codesmith fix
codesmith explain
```

CodeSmith asks before file writes and commands that are not on its safe
development-command list. `--auto` skips those confirmation prompts, while
destructive commands covered by the built-in safety policy remain blocked.

### Command-line options

```text
codesmith [OPTIONS]
codesmith [OPTIONS] {review,fix,explain}

  -C, --repository PATH     Repository root (default: current directory)
  --url URL                 OpenRouter API base URL
  --api-key KEY             API key; defaults to OPENROUTER_API_KEY
  --model MODEL             OpenRouter model ID
  --timeout SECONDS         OpenRouter request timeout
  --max-iterations NUMBER   Maximum agent iterations
  --auto                    Skip confirm-level operation prompts
  --debug                   Show raw iteration, result, and token details
  --show-work               Show repository activity (currently enabled by default)
```

## Interactive commands

Prompts without a slash are sent to the coding agent. The interactive terminal
supports these commands:

```text
/models [NUMBER|ID]  Browse or select a free tool-capable OpenRouter model
/status              Show the session ID, context size, and token usage
/compact             Summarize older conversation context now
/mention PATH        Attach a repository file to the next prompt
/resume [ID]         List or resume a saved session for this repository
/new                 Save the current session and start a new one
/clear               Clear the current conversation context
/help                Show interactive command help
/exit                Exit CodeSmith
```

Files can also be referenced directly in a prompt:

```text
Explain @src/codesmith/context.py
Compare @src/codesmith/agent.py and @src/codesmith/tools.py
```

Referenced files are limited to the active repository and are included only for
that request. `/mention` queues a file for the next prompt.

## Context and saved sessions

Interactive sessions keep recent questions, answers, reasoning details, and
relevant file state. When the configured token threshold is reached, CodeSmith
summarizes older turns through a regular OpenRouter model request. If that
request fails, it uses a deterministic local summary.

Sessions are stored under `~/.codesmith/sessions/` and grouped by repository.
The `/status`, `/compact`, `/resume`, `/new`, and `/clear` commands manage this
state. `/compact` can consume additional OpenRouter quota or billable tokens.

OpenRouter models use different tokenizers, so the active-context count is a
conservative estimate. Provider-reported input and output usage is tracked
separately and shown by `/status`.

## Project instructions

CodeSmith loads instruction files in increasing precedence:

1. `$CODESMITH_HOME/AGENTS.override.md` or `$CODESMITH_HOME/AGENTS.md`.
   `CODESMITH_HOME` defaults to `~/.codesmith`.
2. The backward-compatible repository file `.codesmith/rules.md`, if present.
3. `AGENTS.override.md` or `AGENTS.md` from the repository root down to the
   active working directory.

Instructions closer to the working directory appear later and take precedence.
Combined project instructions have a default 32 KiB limit.

## Configuration

Repository settings can be placed in `.codesmith/config.yaml`. The current
runtime reads the model ID, maximum iteration count, and context settings:

```yaml
model:
  model: openai/gpt-oss-20b:free

agent:
  max_iterations: 20

context:
  max_tokens: 15000
  max_message_tokens: 4500
  max_tool_result_tokens: 3000
  max_summary_tokens: 2000
  max_file_chars: 10000
  max_referenced_files: 5
  compact_threshold: 0.8
  keep_recent_turns: 2
  project_doc_max_bytes: 32768
  project_doc_fallback_filenames: []
```

The following environment variables are supported:

```text
OPENROUTER_API_KEY    OpenRouter API key
OPENROUTER_BASE_URL  API base URL (default: https://openrouter.ai/api/v1)
OPENROUTER_TIMEOUT   Request timeout in seconds (default: 600)
CODESMITH_HOME       Global instructions and session-data directory
```

Command-line values take precedence over environment variables and repository
configuration where applicable.

## Batch generation

`codesmith-batch` generates files directly without the repository-agent loop.

Generate one file:

```bash
codesmith-batch quicksort.py "Write a quicksort implementation with tests"
```

Generate several files from JSON:

```json
{
  "tasks": [
    {
      "output": "quicksort.py",
      "prompt": "Write a quicksort implementation"
    },
    {
      "output": "test_quicksort.py",
      "prompt": "Write tests for quicksort.py",
      "model": "openrouter/free"
    }
  ]
}
```

```bash
codesmith-batch batch.json
codesmith-batch batch.json --model openai/gpt-oss-20b:free
```

Batch outputs are written relative to the current directory.

## Docker

Docker Compose builds the image, passes the OpenRouter configuration, and mounts
the repository at `/work`:

```bash
export OPENROUTER_API_KEY="your-key"
docker compose run --rm codesmith-cli
docker compose run --rm codesmith-cli review
```

To run CodeSmith against another local repository with the built image:

```bash
docker build -t codesmith-cli .
docker run -it --rm \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v /path/to/project:/work \
  -w /work \
  codesmith-cli
```

The Docker image includes Git and ripgrep. No local model server is required.

## Development

Install the test dependency and run the suite:

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

Key modules:

```text
src/codesmith/runtime_cli.py  CLI parsing, terminal UI, and session commands
src/codesmith/agent.py        Agent loop and tool schemas
src/codesmith/context.py      Context budgeting, instructions, and sessions
src/codesmith/tools.py        Repository-scoped file, Git, search, and shell tools
src/codesmith/llm.py          OpenRouter providers
src/codesmith/batch.py        Non-interactive batch generation
```

## Troubleshooting

### `OPENROUTER_API_KEY is not set`

Export the key in the shell that starts CodeSmith. For Docker Compose, the key
must be set on the host before running the service.

### A model cannot use repository tools

Use `/models` in an interactive session and select a model marked as supporting
tool calling, or start CodeSmith with `--model openrouter/free`.

### Repository search fails

Install `rg` locally or use the Docker image, which already includes it.

### A request is rate-limited

Retry later, choose another model, or check the limits and credits associated
with your OpenRouter account. CodeSmith reports HTTP 429 responses without
assuming a particular account limit.

## Contributing

Contributions and issue reports are welcome through the repository's pull
request and issue trackers.
