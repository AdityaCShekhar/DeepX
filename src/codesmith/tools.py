"""Repository-scoped tools used by CodeSmith's CLI and agent runtime.

The original :class:`FileTools` API is kept for backwards compatibility.  The
new ``RepositoryTools`` and ``ToolRegistry`` provide the safer, structured
interface required by an autonomous coding loop.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol


class ToolsError(Exception):
    """Custom exception for tool operations."""
    pass


class Permission(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


@dataclass
class ToolResult:
    """A serializable observation returned to the model."""

    ok: bool
    output: str
    permission: Permission = Permission.SAFE
    metadata: Optional[Dict[str, Any]] = None


class Tool(Protocol):
    name: str
    description: str
    permission: Permission
    schema: Dict[str, Any]

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        ...


def _inside(root: Path, candidate: str) -> Path:
    """Resolve a model-supplied path and reject traversal/symlink escapes."""
    path = (root / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ToolsError(f"Path is outside repository: {candidate}") from exc
    return path


class FileTools:
    """Handle file operations."""

    @staticmethod
    def read_file(filepath: str) -> str:
        """Read a file and return its contents.
        
        Args:
            filepath: Path to the file to read
            
        Returns:
            File contents as string
            
        Raises:
            ToolsError: If file doesn't exist or can't be read
        """
        try:
            path = Path(filepath).resolve()
            if not path.exists():
                raise ToolsError(f"File not found: {filepath}")
            if not path.is_file():
                raise ToolsError(f"Path is not a file: {filepath}")
            return path.read_text()
        except Exception as e:
            raise ToolsError(f"Cannot read file '{filepath}': {str(e)}")

    @staticmethod
    def write_file(filepath: str, content: str) -> str:
        """Write content to a file.
        
        Args:
            filepath: Path to the file to write
            content: Content to write
            
        Returns:
            Success message
            
        Raises:
            ToolsError: If file can't be written
        """
        try:
            path = Path(filepath).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"✓ Successfully wrote {len(content)} bytes to {filepath}"
        except Exception as e:
            raise ToolsError(f"Cannot write file '{filepath}': {str(e)}")

    @staticmethod
    def file_info(filepath: str) -> dict:
        """Get information about a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file info
        """
        try:
            path = Path(filepath).resolve()
            if not path.exists():
                raise ToolsError(f"File not found: {filepath}")
            stat = path.stat()
            return {
                "path": str(path),
                "size": stat.st_size,
                "exists": True,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
            }
        except Exception as e:
            raise ToolsError(f"Cannot get file info: {str(e)}")


class ContextInjector:
    """Inject file context into prompts."""

    @staticmethod
    def inject_files(prompt: str, file_paths: Optional[list] = None) -> str:
        """Inject file contents into the prompt.
        
        Args:
            prompt: Original prompt
            file_paths: List of file paths to include
            
        Returns:
            Enhanced prompt with file context
        """
        if not file_paths:
            return prompt

        context = []
        for filepath in file_paths:
            try:
                content = FileTools.read_file(filepath)
                context.append(f"File: {filepath}\n```\n{content}\n```")
            except ToolsError as e:
                context.append(f"Error reading {filepath}: {str(e)}")

        if context:
            injected = "Context files:\n" + "\n\n".join(context) + "\n\nPrompt:\n" + prompt
            return injected
        return prompt


class RepositoryTools:
    """Concrete tools operating only within ``root``."""

    _ignored_dirs = {".git", ".venv", "__pycache__", "node_modules", "dist", "build"}
    _blocked_commands = re.compile(r"(^|[;&|])\s*(rm\s+-rf\s+(/|~)|mkfs|shutdown|reboot|diskutil\s+eraseDisk)", re.I)
    _safe_commands = {"git", "pytest", "python", "python3", "mvn", "gradle", "npm", "go", "cargo", "ruff", "mypy"}

    def __init__(
        self,
        root: str | Path = ".",
        confirm=None,
        command_timeout: int = 30,
        max_read_chars: int = 10000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolsError(f"Repository does not exist: {root}")
        self.confirm = confirm
        self.command_timeout = command_timeout
        if max_read_chars <= 0:
            raise ToolsError("max_read_chars must be positive")
        self.max_read_chars = max_read_chars

    def _path(self, path: str) -> Path:
        return _inside(self.root, path)

    def list_files(self, path: str = ".", max_depth: int = 4, include_hidden: bool = False) -> ToolResult:
        try:
            base = self._path(path)
        except ToolsError as exc:
            return ToolResult(False, str(exc))
        if not base.is_dir():
            return ToolResult(False, f"Not a directory: {path}")
        files = []
        for item in base.rglob("*"):
            rel = item.relative_to(self.root)
            if len(rel.parts) > max_depth + len(Path(path).parts):
                continue
            if any(part in self._ignored_dirs for part in rel.parts):
                continue
            if self._git_ignored(rel):
                continue
            if not include_hidden and any(part.startswith(".") for part in rel.parts):
                continue
            if item.is_file():
                files.append(str(rel))
        return ToolResult(True, "\n".join(sorted(files)))

    def _git_ignored(self, relative_path: Path) -> bool:
        """Ask Git whether a path is ignored, falling back safely outside Git."""
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", "--", str(relative_path)],
                cwd=self.root,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> ToolResult:
        try:
            target = self._path(path)
        except ToolsError as exc:
            return ToolResult(False, str(exc))
        if not target.is_file():
            return ToolResult(False, f"File not found: {path}")
        try:
            data = target.read_bytes()
            if b"\0" in data:
                return ToolResult(False, f"Binary file: {path}")
            lines = data.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(False, f"Cannot read {path}: {exc}")
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        output = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        truncated = len(output) > self.max_read_chars
        if truncated:
            marker = "\n... [file output truncated; use a narrower line range]"
            output = (output[: max(0, self.max_read_chars - len(marker))] + marker)[
                : self.max_read_chars
            ]
        return ToolResult(
            True,
            output,
            metadata={
                "path": str(target),
                "start_line": start,
                "end_line": end,
                "truncated": truncated,
                "total_bytes": len(data),
            },
        )

    def search_code(self, query: str, path: str = ".", context: int = 1) -> ToolResult:
        try:
            base = self._path(path)
        except ToolsError as exc:
            return ToolResult(False, str(exc))
        try:
            proc = subprocess.run(["rg", "-n", "--no-heading", "--color", "never", "-C", str(context), query, str(base)], cwd=self.root, text=True, capture_output=True, timeout=self.command_timeout)
            if proc.returncode not in (0, 1):
                return ToolResult(False, proc.stderr.strip() or "Search failed")
            return ToolResult(True, proc.stdout[:20000] or "No matches found.")
        except FileNotFoundError:
            return ToolResult(False, "ripgrep (rg) is not installed")
        except subprocess.TimeoutExpired:
            return ToolResult(False, "Search timed out")

    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            target = self._path(path)
        except ToolsError as exc:
            return ToolResult(False, str(exc), Permission.CONFIRM)
        if self.confirm and not self.confirm(path):
            return ToolResult(False, f"Write cancelled: {path}", Permission.CONFIRM)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"Wrote {len(content)} bytes to {path}", Permission.CONFIRM)
        except OSError as exc:
            return ToolResult(False, f"Cannot write {path}: {exc}", Permission.CONFIRM)

    def run_command(self, command: str, timeout: Optional[int] = None, auto: bool = False) -> ToolResult:
        if self._blocked_commands.search(command):
            return ToolResult(False, "Command blocked by safety policy", Permission.BLOCKED)
        executable = command.strip().split(maxsplit=1)[0] if command.strip() else ""
        permission = Permission.SAFE if executable in self._safe_commands else Permission.CONFIRM
        if permission == Permission.CONFIRM and not auto and self.confirm and not self.confirm(command):
            return ToolResult(False, f"Command cancelled: {command}", permission)
        started = time.monotonic()
        try:
            proc = subprocess.run(command, cwd=self.root, shell=True, text=True, capture_output=True, timeout=timeout or self.command_timeout)
            output = (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
            return ToolResult(proc.returncode == 0, output[:30000], permission, {"command": command, "exit_code": proc.returncode, "duration_ms": int((time.monotonic() - started) * 1000)})
        except subprocess.TimeoutExpired as exc:
            return ToolResult(False, f"Command timed out after {timeout or self.command_timeout}s\n{exc.stdout or ''}", permission, {"command": command, "duration_ms": int((time.monotonic() - started) * 1000)})

    def git_status(self) -> ToolResult:
        return self.run_command("git status --short", auto=True)

    def git_diff(self) -> ToolResult:
        return self.run_command("git diff --", auto=True)


class ToolRegistry:
    """Registry that validates names and exposes model-friendly schemas."""

    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools = {tool.name: tool for tool in (tools or [])}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolsError(f"Unknown tool: {name}") from exc

    def schemas(self) -> list:
        return [
            {
                "name": name,
                "description": tool.description,
                "parameters": getattr(tool, "schema", {"type": "object", "properties": {}}),
            }
            for name, tool in self._tools.items()
        ]

    @staticmethod
    def _validate(arguments: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in arguments]
        if missing:
            return f"Missing required tool arguments: {', '.join(missing)}"
        for name, value in arguments.items():
            expected = properties.get(name, {}).get("type")
            valid = {
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
            }.get(expected, True)
            if not valid:
                return f"Invalid type for '{name}': expected {expected}"
        return None

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        try:
            tool = self.get(name)
        except ToolsError as exc:
            return ToolResult(False, str(exc))
        error = self._validate(arguments, getattr(tool, "schema", {}))
        if error:
            return ToolResult(False, error, getattr(tool, "permission", Permission.SAFE))
        try:
            return await tool.execute(arguments)
        except Exception as exc:
            return ToolResult(False, f"Tool {name} failed: {exc}", getattr(tool, "permission", Permission.SAFE))
