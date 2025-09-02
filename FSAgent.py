# file: fs_agent.py
# Requires: pip install google-adk

from __future__ import annotations
import os, io, glob, hashlib, time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.context import ReadonlyContext
from google.genai import types

# --------------------------
# Workspace safety utilities
# --------------------------
@dataclass
class SafeWorkspace:
    root: str
    allow_exts: List[str]
    max_bytes: int

    def _real_root(self) -> str:
        return os.path.realpath(self.root)

    def resolve(self, rel_path: str) -> str:
        """Resolve a user-provided relative path within the workspace root."""
        if rel_path.startswith(("~", "/", "\\")):
            raise ValueError("Use a relative path under the workspace root.")
        abs_target = os.path.realpath(os.path.join(self._real_root(), rel_path))
        if not (abs_target == self._real_root() or abs_target.startswith(self._real_root() + os.sep)):
            raise ValueError("Path escapes the workspace root.")
        return abs_target

    def check_ext(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if self.allow_exts and ext not in self.allow_exts:
            raise ValueError(f"Extension '.{ext}' not allowed in this workspace.")

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        h = hashlib.sha256(); h.update(data); return h.hexdigest()

# --------------------------
# File System Toolset
# --------------------------
class FileSystemToolset(BaseToolset):
    """
    A sandboxed set of file tools intended for code files.
    Register this Toolset directly on an ADK Agent.
    """

    def __init__(self, workspace_root: str, allow_exts: List[str], max_bytes: int):
        self.ws = SafeWorkspace(workspace_root, allow_exts, max_bytes)

        # Wrap bound methods as FunctionTool so ADK can expose them.
        self._tools = [
            FunctionTool(func=self.fs_list_files),
            FunctionTool(func=self.fs_read_file),
            FunctionTool(func=self.fs_upsert_file),
            FunctionTool(func=self.fs_replace_in_file),
            FunctionTool(func=self.fs_append_file),
            FunctionTool(func=self.fs_move_file),
            FunctionTool(func=self.fs_delete_file),
        ]

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None):
        # You could make tools dynamic based on state/role if you like.
        return self._tools

    # ---- Tools (each returns a dict; parameters have explicit type hints) ----

    def fs_list_files(self, pattern: str, max_results: int) -> dict:
        """
        Lists files (relative to the workspace root) matching a glob `pattern`
        like '**/*.py' or 'src/**/*.java'. Intended for code file discovery.
        Args:
            pattern: Glob pattern relative to workspace root.
            max_results: Maximum number of paths to return.
        Returns:
            {'status': 'success', 'files': [relative_paths...]}
        """
        root = self.ws._real_root()
        hits = glob.glob(os.path.join(root, pattern), recursive=True)
        out: List[str] = []
        for p in hits[: max(0, max_results)]:
            if os.path.isfile(p):
                rel = os.path.relpath(p, root)
                out.append(rel)
        return {"status": "success", "files": out}

    def fs_read_file(self, path: str, tool_context: ToolContext) -> dict:
        """
        Reads a text file and returns its content with a sha256 for concurrency control.
        Use this before modifying a file; pass the 'sha256' to fs_upsert_file.
        Args:
            path: Relative path to the file (e.g., 'src/app/main.py').
        Returns:
            {'status':'success','path':path,'content':str,'sha256':str,'size':int}
            or {'status':'error','error_message':str}
        """
        try:
            abs_p = self.ws.resolve(path)
            if not os.path.isfile(abs_p):
                return {"status": "error", "error_message": f"File not found: {path}"}
            b = open(abs_p, "rb").read()
            if len(b) > self.ws.max_bytes:
                return {"status": "error", "error_message": f"File too large (> {self.ws.max_bytes} bytes)."}
            try:
                text = b.decode("utf-8")
            except UnicodeDecodeError:
                # still allow reading non-utf8 code as replacement chars
                text = b.decode("utf-8", errors="replace")
            sha = SafeWorkspace.sha256_bytes(b)
            tool_context.state["temp:last_read_path"] = path
            tool_context.state["temp:last_read_sha256"] = sha
            return {"status": "success", "path": path, "content": text, "sha256": sha, "size": len(b)}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_upsert_file(self, path: str, content: str, expected_sha256: str, overwrite: bool, create_parents: bool) -> dict:
        """
        Creates or updates a text file *atomically* with optimistic concurrency.
        Use when writing a whole file.
        Args:
            path: Relative file path to create/update.
            content: New full text content.
            expected_sha256: If updating, pass the sha256 from fs_read_file. If creating, pass ''.
            overwrite: If file exists and expected_sha256 == '', allow unconditional overwrite when True.
            create_parents: Create parent directories if missing when True.
        Returns:
            {'status':'success','path':path,'sha256':str,'created':bool}
            or {'status':'conflict','current_sha256':str} or {'status':'error',...}
        """
        try:
            abs_p = self.ws.resolve(path)
            self.ws.check_ext(abs_p)
            if create_parents:
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)

            exists = os.path.exists(abs_p)
            new_bytes = content.encode("utf-8")
            if len(new_bytes) > self.ws.max_bytes:
                return {"status": "error", "error_message": f"Write exceeds {self.ws.max_bytes} bytes limit."}

            if exists:
                cur = open(abs_p, "rb").read()
                cur_sha = SafeWorkspace.sha256_bytes(cur)
                # Concurrency check
                if expected_sha256 and expected_sha256 != cur_sha:
                    return {"status": "conflict", "current_sha256": cur_sha}
                if not expected_sha256 and not overwrite:
                    return {"status": "error", "error_message": "File exists; set overwrite=True or supply expected_sha256."}
                # backup
                ts = time.strftime("%Y%m%d-%H%M%S")
                open(abs_p + f".bak.{ts}", "wb").write(cur)

            # Atomic-ish write
            tmp = abs_p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(new_bytes)
            os.replace(tmp, abs_p)
            return {
                "status": "success",
                "path": path,
                "sha256": SafeWorkspace.sha256_bytes(new_bytes),
                "created": (not exists),
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_replace_in_file(self, path: str, find: str, replace: str, count: int, create_backup: bool) -> dict:
        """
        In-place search/replace on a text file (useful for quick refactors).
        Args:
            path: Relative path to modify.
            find: Substring to replace.
            replace: Replacement text.
            count: Number of occurrences to replace (use -1 for all).
            create_backup: When True, writes a .bak timestamped copy before edit.
        Returns:
            {'status':'success','path':path,'replacements':int,'sha256':str}
        """
        try:
            abs_p = self.ws.resolve(path)
            self.ws.check_ext(abs_p)
            if not os.path.isfile(abs_p):
                return {"status": "error", "error_message": f"File not found: {path}"}
            b = open(abs_p, "rb").read()
            if len(b) > self.ws.max_bytes:
                return {"status": "error", "error_message": f"File too large (> {self.ws.max_bytes} bytes)."}
            try:
                text = b.decode("utf-8")
            except UnicodeDecodeError:
                return {"status": "error", "error_message": "File is not valid UTF-8 text."}

            new_text, n = text.replace(find, replace, count if count >= 0 else text.count(find)), (
                text.count(find) if count < 0 else min(count, text.count(find))
            )
            if create_backup:
                ts = time.strftime("%Y%m%d-%H%M%S")
                open(abs_p + f".bak.{ts}", "wb").write(b)
            open(abs_p, "wb").write(new_text.encode("utf-8"))
            return {"status": "success", "path": path, "replacements": n, "sha256": SafeWorkspace.sha256_bytes(new_text.encode("utf-8"))}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_append_file(self, path: str, content: str, create_parents: bool) -> dict:
        """
        Appends text to a file (creates if missing).
        Args:
            path: Relative path.
            content: Text to append.
            create_parents: Create parent directories if missing when True.
        Returns:
            {'status':'success','path':path,'sha256':str,'created':bool}
        """
        try:
            abs_p = self.ws.resolve(path)
            self.ws.check_ext(abs_p)
            if create_parents:
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)
            existed = os.path.exists(abs_p)
            with open(abs_p, "ab") as f:
                f.write(content.encode("utf-8"))
            new_b = open(abs_p, "rb").read()
            return {"status": "success", "path": path, "sha256": SafeWorkspace.sha256_bytes(new_b), "created": (not existed)}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_move_file(self, src: str, dst: str, allow_overwrite: bool, create_parents: bool) -> dict:
        """
        Moves/renames a file within the workspace.
        Args:
            src: Source relative path.
            dst: Destination relative path.
            allow_overwrite: Overwrite destination if exists when True.
            create_parents: Create parent directories for destination when True.
        Returns:
            {'status':'success','path':dst} or error/conflict.
        """
        try:
            abs_src = self.ws.resolve(src)
            abs_dst = self.ws.resolve(dst)
            self.ws.check_ext(abs_src)
            self.ws.check_ext(abs_dst)
            if not os.path.isfile(abs_src):
                return {"status": "error", "error_message": f"Source not found: {src}"}
            if os.path.exists(abs_dst) and not allow_overwrite:
                return {"status": "error", "error_message": "Destination exists; set allow_overwrite=True."}
            if create_parents:
                os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            os.replace(abs_src, abs_dst)
            return {"status": "success", "path": dst}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_delete_file(self, path: str, require_exists: bool) -> dict:
        """
        Deletes a file (no directories).
        Args:
            path: Relative path.
            require_exists: If True and file doesn't exist, return error; if False, succeed silently.
        Returns:
            {'status':'success','path':path}
        """
        try:
            abs_p = self.ws.resolve(path)
            if os.path.isdir(abs_p):
                return {"status": "error", "error_message": "Refusing to delete a directory."}
            if not os.path.exists(abs_p) and require_exists:
                return {"status": "error", "error_message": f"File not found: {path}"}
            if os.path.exists(abs_p):
                os.remove(abs_p)
            return {"status": "success", "path": path}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

# --------------------------
# Assemble the agent
# --------------------------
ALLOWED_CODE_EXTS = [
    "py","java","kt","kts","js","jsx","ts","tsx","c","h","cpp","hpp","cs","go",
    "rb","rs","sql","yaml","yml","toml","ini","xml","json","gradle","md","sh","ps1"
]

def build_agent(workspace_root: str) -> Agent:
    """
    Create an LLM agent set up with the FileSystemToolset.
    """
    fs_tools = FileSystemToolset(
        workspace_root=workspace_root,
        allow_exts=ALLOWED_CODE_EXTS,  # restrict to "code" files
        max_bytes=2_000_000,           # ~2MB safety cap per file
    )

    instruction = f"""
You are a coding assistant with access to file tools for a sandboxed workspace.

Rules:
- Only operate under the workspace root: {workspace_root}
- Paths must be relative. Avoid shell metacharacters.
- Before overwriting a file, call fs_read_file and pass its sha256 to fs_upsert_file, or set overwrite=True explicitly.
- Prefer fs_upsert_file for full-file writes; use fs_replace_in_file for small edits.
- Use fs_list_files to discover targets (e.g., '**/*.py').
- Never edit binary files and do not exceed the size limit.
- Summarize changes briefly after tool calls.
"""

    return Agent(
        name="FileOpsAgent",
        model="gemini-2.0-flash",
        description="Agent that can safely read/write/update code files in a sandboxed workspace.",
        instruction=instruction,
        tools=[fs_tools],  # Toolset exposes the individual FunctionTools
    )

# --------------------------
# Minimal runner example
# --------------------------
APP_NAME = "file_ops_app"
USER_ID = "local_user"
SESSION_ID = "dev_session"

async def main():
    # Set your workspace root (defaults to CWD if env var not set)
    workspace_root = os.getenv("ADK_WORKSPACE_ROOT", os.getcwd())

    # Create session + runner
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    agent = build_agent(workspace_root)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    # Example: ask the agent to create a file
    prompt = "Create src/hello.py that prints 'hello, world', then read it back."
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for e in events:
        if e.is_final_response():
            print(e.content.parts[0].text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
