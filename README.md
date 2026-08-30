# CodeSmith - Code Generation CLI

A powerful CLI tool for code generation using OpenRouter's API. Built with Python, this tool provides an interactive REPL for generating code, reading, and writing files.

## Features

- 🚀 **Code Generation**: Generate code from natural language prompts
- 📝 **File Operations**: Read, write, and manage files seamlessly
- 🧠 **Context Injection**: Include file contents in prompts for context-aware generation
- 📊 **Streaming Support**: Real-time token streaming from OpenRouter
- 🎯 **Multiple Models**: Support for OpenRouter models, including free models
- 🎨 **Clean Output**: Formatted responses with color-coded sections
- 🐳 **Docker Ready**: Easy deployment with Docker and Docker Compose

## Architecture

```
CodeSmith/
├── src/codesmith/          # Installable Python package
│   ├── cli.py              # Interactive CLI and command handling
│   ├── llm.py              # OpenRouter API client
│   ├── tools.py            # File and context utilities
│   └── batch.py            # Batch-generation entry point
├── examples/               # Runnable examples and sample input
├── docs/                   # Project documentation
├── scripts/                # Setup and maintenance helpers
├── codesmith               # macOS/Linux launcher
├── codesmith.cmd           # Windows launcher
├── pyproject.toml          # Package metadata and entry points
├── Dockerfile
└── docker-compose.yml
```

### Components

- **src/codesmith/cli.py**: Interactive REPL with color-coded output and command handling
- **src/codesmith/llm.py**: OpenRouter API client with tool calling, reasoning, streaming, and error handling
- **src/codesmith/tools.py**: File I/O and context injection utilities
- **src/codesmith/batch.py**: Non-interactive batch generation
- **pyproject.toml**: Package metadata, dependencies, and CLI entry points

## Requirements

- Python 3.8+
- An OpenRouter API key (free models are supported)

## Quick Command Setup

To use `codesmith` from anywhere on macOS or Linux:

```bash
# Option 1: Add to PATH (recommended)
export PATH="/path/to/CodeSmith:$PATH"

# Add to ~/.zshrc or ~/.bashrc to make permanent:
echo 'export PATH="/path/to/CodeSmith:$PATH"' >> ~/.zshrc

# Or symlink to /usr/local/bin
sudo ln -sf /path/to/CodeSmith/codesmith /usr/local/bin/codesmith
```

Then simply type:
```bash
codesmith                    # Interactive mode
codesmith -p "Your prompt"   # Single command
```

On Windows, add the CodeSmith repository directory to your User `PATH` in
Settings, restart Windows Terminal, and run:

```powershell
codesmith
codesmith -p "Your prompt"
```

Windows automatically uses `codesmith.cmd`; macOS and Linux use `codesmith`.

## Installation & Setup

### Option 1: OpenRouter API (Recommended)

The easiest way to run everything together:

```bash
# Clone/navigate to the project
cd /path/to/CodeSmith

# Configure your OpenRouter API key
export OPENROUTER_API_KEY="your-key"

# Run the CLI
codesmith
```

### Option 2: Local Installation

1. **Install Python dependencies:**
   ```bash
   python3 -m pip install -e .
   ```

2. **Set your API key and run the CLI:**
   ```bash
   export OPENROUTER_API_KEY="your-key"
   codesmith
   ```

## Usage

### Interactive REPL Mode

```bash
codesmith
```

Or:
```bash
codesmith
```

### Single Prompt Mode

```bash
codesmith -p "Write a Python function to calculate fibonacci"
```

Or:
```bash
codesmith -p "Write a Python function to calculate fibonacci"
```

### Custom OpenRouter URL and Model

```bash
export OPENROUTER_API_KEY="your-key"
codesmith --model openai/gpt-oss-20b:free
```

Or:
```bash
codesmith --model openai/gpt-oss-20b:free
```

## Commands

All commands start with `/`. Here are the available commands:

### Code Generation
Simply type your prompt without a `/` prefix:
```
> Write a Python function to reverse a string
```

### File Generation

**Generate and write a file:**
```
/write <filename>
```
Then enter instructions for the code you want generated and saved.

### File Context

Mention files directly in a prompt. Their contents are included for that
prompt only:
```
> Explain @filename.py and suggest improvements
> Write tests for @utils.py and @models.py
```

Interactive sessions retain recent questions, final answers, OpenRouter
reasoning details, and relevant file state, so follow-up prompts can refer to
earlier work. CodeSmith uses token-based budgets and automatically summarizes
older turns when the configured compaction threshold is reached. The summary is
created with a normal OpenRouter model request; if that request fails, CodeSmith
uses a deterministic local fallback.

CodeSmith saves sessions under `~/.codesmith/sessions/`, grouped by repository.
Use the context commands to manage them:

```text
/status            Show context and OpenRouter token usage
/compact           Summarize older context now
/mention PATH      Attach a file to the next prompt
/resume [ID]       Resume a saved repository session
/new               Start a new saved conversation
/clear             Clear the current conversation
```

`/compact` makes an additional OpenRouter request and may consume billable or
quota-limited tokens. CodeSmith does not call OpenAI's provider-specific
`/responses/compact` endpoint.

### Project instructions

CodeSmith loads instruction files in increasing precedence:

1. `$CODESMITH_HOME/AGENTS.override.md`, or `$CODESMITH_HOME/AGENTS.md`
   (`CODESMITH_HOME` defaults to `~/.codesmith`).
2. The legacy repository `.codesmith/rules.md`, when present.
3. `AGENTS.override.md` or `AGENTS.md` from the repository root down to the
   active working directory. A configured fallback filename can be used when
   neither standard name exists.

Instructions closer to the working directory appear later in the prompt and
therefore take precedence. Combined project instructions default to a 32 KiB
budget.

Context settings can be overridden in `.codesmith/config.yaml`:

```yaml
context:
  max_tokens: 15000
  max_message_tokens: 4500
  max_tool_result_tokens: 3000
  max_summary_tokens: 2000
  compact_threshold: 0.8
  keep_recent_turns: 2
  project_doc_max_bytes: 32768
  project_doc_fallback_filenames: []
```

OpenRouter models use different tokenizers, so pre-request accounting is a
conservative estimate. `/status` also reports the actual input and output usage
returned by OpenRouter after completed requests.

### Utilities

**List available models:**
```
/models
```

`/models` loads the current OpenRouter catalog and lists all free models. Models
show their context size and whether tool calling is supported. Select one by
number, or type `/models <number>` to select it directly. For repository tasks
that need file inspection or writing, choose a model marked `Tool calling
supported`.

The interactive prompt uses the legacy CodeSmith layout with a cyan theme. Type
`/` to open command suggestions, then press Enter to accept the first suggestion.

**Show help:**
```
/help
```

**Exit:**
```
/exit
```

## Usage Examples

### Example 1: Generate a Python Function

```
> Write a Python function to count vowels in a string

⭐ Generating
ℹ Temperature: 0.7 | Top-p: 0.9

```python
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for c in s if c in vowels)

result = count_vowels('Hello World')
print(result)  # Output: 3
```
```

### Example 2: Generate Code with Context

```
@style.py

> Generate a class that follows the patterns in the context file

⭐ Generating
ℹ Temperature: 0.7 | Top-p: 0.9

[generates code following the style patterns]
```

### Example 3: Write Generated Code to File

```
> Write a function to parse CSV files

[AI generates code]

/write csv_parser.py

✓ Successfully wrote 1245 bytes to csv_parser.py
```

### Example 4: Test the Generated Code

```

⭐ Running: python csv_parser.py --test
STDOUT:
All tests passed!
```

## Command-Line Options

```bash
codesmith --help

options:
  -h, --help            show this help message and exit
  --url URL              OpenRouter API base URL (default: https://openrouter.ai/api/v1)
  --api-key API_KEY      OpenRouter API key (or OPENROUTER_API_KEY)
  -m MODEL, --model MODEL
                        Model name (default: openai/gpt-oss-20b:free)
  -p PROMPT, --prompt PROMPT
                        Single prompt to execute and exit
  --no-stream           Disable streaming mode
```

## Automation

Generate and save code automatically without interactive mode!

### Quick Automation Examples

```bash
# Single file generation
codesmith-batch quicksort.py "Write a quicksort implementation"

# Batch generation from JSON config
codesmith-batch batch.json

# Generates and saves files automatically!
```

### Batch Generation

Create a `batch.json` file:

```json
{
  "tasks": [
    {
      "output": "quicksort.py",
      "prompt": "Write a quicksort algorithm with tests"
    },
    {
      "output": "api.py",
      "prompt": "Create a Flask REST API with CRUD endpoints"
    },
    {
      "output": "tests.py",
      "prompt": "Write comprehensive unit tests"
    }
  ]
}
```

Then run:
```bash
codesmith-batch batch.json
```

All files are generated and saved automatically!

### Usage

```bash
# Single file
codesmith-batch <output_file> "<prompt>"

# Batch from JSON
codesmith-batch <config.json>

# With a specific OpenRouter model
codesmith-batch <output_file> "<prompt>" --model openai/gpt-oss-20b:free
```

**See [AUTOMATION.md](docs/AUTOMATION.md) for complete automation guide**

## Docker Usage

The Docker image can run CodeSmith while using OpenRouter; no local model server is required.

```bash
# Build the image
docker build -t codesmith-cli .

# Run with your OpenRouter key
docker run -it --rm \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v $(pwd)/workspace:/workspace \
  codesmith-cli
```

## Features in Detail

### Streaming Support

The CLI streams responses token-by-token as they're generated, providing real-time feedback:

```
> Write a hello world program in Rust
```python
fn main() {
    println!("Hello, world!");
}
```
```

### Context Injection

Include files in your prompts for code generation aware of your codebase:

```
@utils.py
@config.json
> Generate tests for the functions in utils.py
```

The utility files are automatically included in the prompt sent to the model.

### Error Handling

All operations include graceful error handling:

- File not found errors
- Permission issues
- Command execution failures
- OpenRouter connection problems
- Timeout handling

### Clean Output Formatting

- Syntax highlighting for code blocks
- Color-coded messages (info, success, error)
- Organized section headers
- Proper indentation and formatting

## Performance Tips

1. **Use context wisely**: Only include relevant files to keep prompts concise
2. **Model selection**: The default is `openai/gpt-oss-20b:free`; choose another OpenRouter model with `--model` or use `/models`
3. **Streaming**: Works best for faster feedback; disable with `--no-stream` if needed
4. **Command timeouts**: Default is 30 seconds; adjust in code for long operations

## Troubleshooting

### "OPENROUTER_API_KEY is not set"

Export an OpenRouter key before starting CodeSmith:

```bash
export OPENROUTER_API_KEY="your-key"
```

For Docker Compose, export the key on the host before starting the service:

```bash
export OPENROUTER_API_KEY="your-key"
docker compose run --rm codesmith-cli
```

### OpenRouter free-model limits

OpenRouter currently allows 50 free-model requests per day without purchased
credits. Purchasing at least $10 in credits raises the free-model limit to
1,000 requests per day. OpenRouter credits use USD, and purchases may include a
service fee. Limits are account-specific and can change.

If the daily limit is reached, wait for the reset or use an account with
available credits. A `429` response is handled and displayed as a friendly
quota message by CodeSmith.

### Slow responses

- Reduce context file size
- Try another model from `/models`, or use `codesmith --model openrouter/free`
- Check your OpenRouter model and account limits
- Reduce temperature for faster inference (in code)

### Permission denied on file operations

Ensure you have write permissions in the working directory:
```bash
chmod 755 ./workspace
ls -la | grep workspace
```

### Port 11434 already in use

Change the port in docker-compose.yml:
```yaml
ports:
  - "11435:11434"  # Change left number to unused port
```

## Architecture Details

### LLM Module (`src/codesmith/llm.py`)

- **OpenRouterClient**: Manages communication with OpenRouter API
- **Streaming**: Real-time token generation
- **Error Handling**: Connection validation and timeout management
- **Reasoning and tool calls**: Preserves OpenRouter reasoning details across tool iterations

### Tools Module (`src/codesmith/tools.py`)

- **FileTools**: Read, write, and inspect files
- **ContextInjector**: Embed file contents into prompts

### CLI Module (`src/codesmith/cli.py`)

- **CodeSmithCLI**: Main application class
- **REPL Loop**: Interactive command processing
- **Command Handling**: Dispatch to appropriate handlers
- **Output Formatting**: Color-coded, organized output

## Future Enhancements

- [ ] Code execution sandboxing
- [ ] Multi-file context management
- [ ] Syntax highlighting for output
- [ ] Command history and autocomplete
- [ ] Custom prompt templates
- [ ] Response caching
- [ ] Model fine-tuning support
- [ ] Plugin system for custom commands

## License

MIT License - feel free to use and modify for your needs.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Happy coding with CodeSmith!** 🚀
