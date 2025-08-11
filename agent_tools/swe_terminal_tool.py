#!/usr/bin/env python3
"""
SWE Agent Terminal Tool

A stateful terminal tool for multi-agent orchestration systems that can execute
commands like git, maven, and other development tools using subprocess.
The tool maintains state and allows agents to pass multiple commands that wait
for subsequent commands.

Designed to be used as a FunctionTool with Google ADK.
"""

import os
import sys
import time
import json
import shlex
import shutil
import platform
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class CommandStatus(Enum):
    """Status of command execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CommandType(Enum):
    """Type of command being executed."""
    GIT = "git"
    MAVEN = "maven"
    NPM = "npm"
    DOCKER = "docker"
    KUBECTL = "kubectl"
    SHELL = "shell"
    CUSTOM = "custom"


@dataclass
class CommandResult:
    """Result of command execution."""
    command_id: str
    command: str
    command_type: CommandType
    status: CommandStatus
    return_code: int
    stdout: str
    stderr: str
    start_time: float
    end_time: float
    duration: float
    working_directory: str
    environment: Dict[str, str]
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['command_type'] = self.command_type.value
        result['status'] = self.status.value
        return result


@dataclass
class TerminalState:
    """State maintained by the terminal tool."""
    session_id: str
    working_directory: str
    environment: Dict[str, str]
    command_history: List[CommandResult]
    active_processes: Dict[str, subprocess.Popen]
    max_history: int = 100
    timeout_seconds: float = 300.0  # 5 minutes default


class SWETerminalTool:
    """
    SWE Agent Terminal Tool for multi-agent orchestration.
    
    This tool maintains state across multiple command executions and allows
    agents to execute commands in sequence, waiting for subsequent commands.
    """
    
    def __init__(self):
        """Initialize the terminal tool."""
        self._sessions: Dict[str, TerminalState] = {}
        self._session_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        
        # Initialize default environment
        self._default_env = os.environ.copy()
        
        # Command type detection patterns
        self._command_patterns = {
            CommandType.GIT: ["git"],
            CommandType.MAVEN: ["mvn", "mvnw", "mvn.cmd"],
            CommandType.NPM: ["npm", "npx", "yarn"],
            CommandType.DOCKER: ["docker", "docker-compose"],
            CommandType.KUBECTL: ["kubectl", "kubectl.exe"],
            CommandType.SHELL: ["ls", "cd", "pwd", "echo", "cat", "grep", "find"],
        }
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new terminal session.
        
        Args:
            session_id: Optional session ID, auto-generated if not provided
            working_directory: Initial working directory
            environment: Initial environment variables
            
        Returns:
            Dictionary with session information
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if working_directory is None:
            working_directory = os.getcwd()
        
        if environment is None:
            environment = {}
        
        # Merge with default environment
        merged_env = self._default_env.copy()
        merged_env.update(environment)
        
        with self._global_lock:
            if session_id in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} already exists",
                    "session_id": None
                }
            
            # Ensure working directory exists
            try:
                working_directory = os.path.abspath(working_directory)
                if not os.path.exists(working_directory):
                    os.makedirs(working_directory, exist_ok=True)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid working directory: {e}",
                    "session_id": None
                }
            
            # Create session
            session = TerminalState(
                session_id=session_id,
                working_directory=working_directory,
                environment=merged_env,
                command_history=[],
                active_processes={}
            )
            
            self._sessions[session_id] = session
            self._session_locks[session_id] = threading.Lock()
            
            return {
                "success": True,
                "session_id": session_id,
                "working_directory": working_directory,
                "environment": merged_env,
                "message": f"Session {session_id} created successfully"
            }
    
    def execute_command(
        self,
        session_id: str,
        command: str,
        wait_for_completion: bool = True,
        timeout_seconds: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a command in the specified session.
        
        Args:
            session_id: Session ID to execute command in
            command: Command to execute
            wait_for_completion: Whether to wait for command completion
            timeout_seconds: Command timeout in seconds
            working_directory: Override working directory for this command
            environment: Override environment variables for this command
            
        Returns:
            Dictionary with command execution result
        """
        with self._global_lock:
            if session_id not in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found",
                    "command_id": None
                }
        
        session = self._sessions[session_id]
        session_lock = self._session_locks[session_id]
        
        with session_lock:
            # Validate command
            if not command or not command.strip():
                return {
                    "success": False,
                    "error": "Command cannot be empty",
                    "command_id": None
                }
            
            # Generate command ID
            command_id = str(uuid.uuid4())
            
            # Determine command type
            command_type = self._detect_command_type(command)
            
            # Use session working directory if not overridden
            cmd_working_dir = working_directory or session.working_directory
            cmd_env = environment or session.environment
            
            # Create command result placeholder
            start_time = time.time()
            command_result = CommandResult(
                command_id=command_id,
                command=command,
                command_type=command_type,
                status=CommandStatus.PENDING,
                return_code=-1,
                stdout="",
                stderr="",
                start_time=start_time,
                end_time=start_time,
                duration=0.0,
                working_directory=cmd_working_dir,
                environment=cmd_env,
                metadata={
                    "wait_for_completion": wait_for_completion,
                    "timeout_seconds": timeout_seconds or session.timeout_seconds
                }
            )
            
            # Add to history
            session.command_history.append(command_result)
            if len(session.command_history) > session.max_history:
                session.command_history.pop(0)
            
            try:
                # Execute command
                if wait_for_completion:
                    result = self._execute_sync_command(
                        command, cmd_working_dir, cmd_env, 
                        timeout_seconds or session.timeout_seconds
                    )
                    
                    # Update command result
                    end_time = time.time()
                    command_result.status = CommandStatus.COMPLETED
                    command_result.return_code = result["return_code"]
                    command_result.stdout = result["stdout"]
                    command_result.stderr = result["stderr"]
                    command_result.end_time = end_time
                    command_result.duration = end_time - start_time
                    
                    if result["return_code"] != 0:
                        command_result.status = CommandStatus.FAILED
                        command_result.error_message = result["stderr"] or "Command failed"
                    
                    return {
                        "success": True,
                        "command_id": command_id,
                        "session_id": session_id,
                        "command": command,
                        "command_type": command_type.value,
                        "status": command_result.status.value,
                        "return_code": command_result.return_code,
                        "stdout": command_result.stdout,
                        "stderr": command_result.stderr,
                        "working_directory": cmd_working_dir,
                        "duration": command_result.duration,
                        "wait_for_completion": wait_for_completion
                    }
                else:
                    # Execute asynchronously
                    command_result.status = CommandStatus.RUNNING
                    thread = threading.Thread(
                        target=self._execute_async_command,
                        args=(command_id, session_id, command, cmd_working_dir, cmd_env, timeout_seconds or session.timeout_seconds)
                    )
                    thread.daemon = True
                    thread.start()
                    
                    return {
                        "success": True,
                        "command_id": command_id,
                        "session_id": session_id,
                        "command": command,
                        "command_type": command_type.value,
                        "status": CommandStatus.RUNNING.value,
                        "message": "Command started asynchronously",
                        "wait_for_completion": wait_for_completion
                    }
                    
            except Exception as e:
                command_result.status = CommandStatus.FAILED
                command_result.error_message = str(e)
                command_result.end_time = time.time()
                command_result.duration = command_result.end_time - start_time
                
                return {
                    "success": False,
                    "error": str(e),
                    "command_id": command_id,
                    "session_id": session_id
                }
    
    def execute_multiple_commands(
        self,
        session_id: str,
        commands: List[str],
        wait_between_commands: bool = True,
        timeout_seconds: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute multiple commands in sequence.
        
        Args:
            session_id: Session ID to execute commands in
            commands: List of commands to execute
            wait_between_commands: Whether to wait between commands
            timeout_seconds: Timeout for each command
            working_directory: Override working directory
            environment: Override environment variables
            
        Returns:
            Dictionary with execution results
        """
        if not commands:
            return {
                "success": False,
                "error": "No commands provided",
                "results": []
            }
        
        results = []
        current_wd = working_directory
        
        for i, command in enumerate(commands):
            # Execute command
            result = self.execute_command(
                session_id=session_id,
                command=command,
                wait_for_completion=True,
                timeout_seconds=timeout_seconds,
                working_directory=current_wd,
                environment=environment
            )
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Command {i+1} failed: {result.get('error', 'Unknown error')}",
                    "results": results,
                    "failed_command_index": i,
                    "failed_command": command
                }
            
            results.append(result)
            
            # Update working directory for next command if it was a cd command
            if command.strip().startswith("cd "):
                try:
                    # Extract directory from cd command
                    parts = shlex.split(command)
                    if len(parts) >= 2:
                        new_dir = parts[1]
                        if new_dir == "-":
                            # Go back to previous directory
                            if len(results) > 1:
                                prev_result = results[-2]
                                current_wd = prev_result.get("working_directory", current_wd)
                        else:
                            # Change to specified directory
                            if os.path.isabs(new_dir):
                                current_wd = new_dir
                            else:
                                current_wd = os.path.join(current_wd or ".", new_dir)
                        current_wd = os.path.abspath(current_wd)
                except Exception:
                    pass  # Ignore cd parsing errors
            
            # Wait between commands if requested
            if wait_between_commands and i < len(commands) - 1:
                time.sleep(0.1)  # Small delay to ensure proper sequencing
        
        return {
            "success": True,
            "session_id": session_id,
            "total_commands": len(commands),
            "results": results,
            "final_working_directory": current_wd
        }
    
    def get_session_status(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a session.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Dictionary with session status
        """
        with self._global_lock:
            if session_id not in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
        
        session = self._sessions[session_id]
        
        # Count commands by status
        status_counts = {}
        for status in CommandStatus:
            status_counts[status.value] = 0
        
        for cmd in session.command_history:
            status_counts[cmd.status.value] += 1
        
        return {
            "success": True,
            "session_id": session_id,
            "working_directory": session.working_directory,
            "total_commands": len(session.command_history),
            "status_counts": status_counts,
            "active_processes": len(session.active_processes),
            "environment_variables": len(session.environment)
        }
    
    def get_command_history(
        self,
        session_id: str,
        limit: int = 50,
        command_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get command history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of commands to return
            command_type: Filter by command type
            status: Filter by command status
            
        Returns:
            Dictionary with filtered command history
        """
        with self._global_lock:
            if session_id not in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
        
        session = self._sessions[session_id]
        
        # Filter commands
        filtered_commands = []
        for cmd in reversed(session.command_history):  # Most recent first
            if command_type and cmd.command_type.value != command_type:
                continue
            if status and cmd.status.value != status:
                continue
            
            filtered_commands.append(cmd.to_dict())
            if len(filtered_commands) >= limit:
                break
        
        return {
            "success": True,
            "session_id": session_id,
            "commands": filtered_commands,
            "total_commands": len(filtered_commands),
            "filters": {
                "command_type": command_type,
                "status": status,
                "limit": limit
            }
        }
    
    def change_working_directory(
        self,
        session_id: str,
        new_directory: str
    ) -> Dict[str, Any]:
        """
        Change working directory for a session.
        
        Args:
            session_id: Session ID
            new_directory: New working directory path
            
        Returns:
            Dictionary with result
        """
        with self._global_lock:
            if session_id not in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
        
        session = self._sessions[session_id]
        
        try:
            # Resolve path
            if os.path.isabs(new_directory):
                abs_path = os.path.abspath(new_directory)
            else:
                abs_path = os.path.abspath(os.path.join(session.working_directory, new_directory))
            
            # Check if directory exists
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"Directory does not exist: {abs_path}"
                }
            
            if not os.path.isdir(abs_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {abs_path}"
                }
            
            # Update working directory
            old_directory = session.working_directory
            session.working_directory = abs_path
            
            return {
                "success": True,
                "session_id": session_id,
                "old_directory": old_directory,
                "new_directory": abs_path,
                "message": f"Working directory changed from {old_directory} to {abs_path}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to change directory: {e}"
            }
    
    def list_sessions(self) -> Dict[str, Any]:
        """
        List all active sessions.
        
        Returns:
            Dictionary with session information
        """
        with self._global_lock:
            sessions_info = []
            for session_id, session in self._sessions.items():
                sessions_info.append({
                    "session_id": session_id,
                    "working_directory": session.working_directory,
                    "total_commands": len(session.command_history),
                    "active_processes": len(session.active_processes),
                    "environment_variables": len(session.environment)
                })
            
            return {
                "success": True,
                "total_sessions": len(sessions_info),
                "sessions": sessions_info
            }
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        Close a session and clean up resources.
        
        Args:
            session_id: Session ID to close
            
        Returns:
            Dictionary with result
        """
        with self._global_lock:
            if session_id not in self._sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
        
        session = self._sessions[session_id]
        
        # Terminate active processes
        for process_id, process in session.active_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass  # Ignore cleanup errors
        
        # Remove session
        del self._sessions[session_id]
        del self._session_locks[session_id]
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Session {session_id} closed successfully"
        }
    
    def _detect_command_type(self, command: str) -> CommandType:
        """Detect the type of command being executed."""
        command_parts = shlex.split(command)
        if not command_parts:
            return CommandType.SHELL
        
        base_command = command_parts[0].lower()
        
        for cmd_type, patterns in self._command_patterns.items():
            if any(base_command.startswith(pattern) for pattern in patterns):
                return cmd_type
        
        return CommandType.CUSTOM
    
    def _execute_sync_command(
        self,
        command: str,
        working_directory: str,
        environment: Dict[str, str],
        timeout_seconds: float
    ) -> Dict[str, Any]:
        """Execute a command synchronously."""
        # Detect shell
        shell = self._detect_shell()
        
        # Build command arguments
        if shell in ["bash", "sh"]:
            args = [shell, "-c", command]
        elif shell in ["powershell", "pwsh"]:
            args = [shell, "-Command", command]
        elif shell == "cmd":
            args = ["cmd", "/c", command]
        else:
            args = [shell, "-c", command]
        
        # Execute command
        start_time = time.time()
        try:
            process = subprocess.Popen(
                args,
                cwd=working_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )
            
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            timed_out = False
            
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            timed_out = True
        except Exception as e:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "timed_out": False
            }
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "duration": duration
        }
    
    def _execute_async_command(
        self,
        command_id: str,
        session_id: str,
        command: str,
        working_directory: str,
        environment: Dict[str, str],
        timeout_seconds: float
    ):
        """Execute a command asynchronously."""
        try:
            result = self._execute_sync_command(
                command, working_directory, environment, timeout_seconds
            )
            
            # Update command result
            with self._session_locks[session_id]:
                session = self._sessions[session_id]
                for cmd in session.command_history:
                    if cmd.command_id == command_id:
                        cmd.status = CommandStatus.COMPLETED
                        cmd.return_code = result["return_code"]
                        cmd.stdout = result["stdout"]
                        cmd.stderr = result["stderr"]
                        cmd.end_time = time.time()
                        cmd.duration = cmd.end_time - cmd.start_time
                        
                        if result["return_code"] != 0:
                            cmd.status = CommandStatus.FAILED
                            cmd.error_message = result["stderr"] or "Command failed"
                        
                        if result["timed_out"]:
                            cmd.status = CommandStatus.TIMEOUT
                            cmd.error_message = "Command timed out"
                        
                        break
                        
        except Exception as e:
            # Update command result with error
            with self._session_locks[session_id]:
                session = self._sessions[session_id]
                for cmd in session.command_history:
                    if cmd.command_id == command_id:
                        cmd.status = CommandStatus.FAILED
                        cmd.error_message = str(e)
                        cmd.end_time = time.time()
                        cmd.duration = cmd.end_time - cmd.start_time
                        break
    
    def _detect_shell(self) -> str:
        """Detect the default shell for the current platform."""
        system = platform.system().lower()
        
        if system == "windows":
            if shutil.which("pwsh"):
                return "pwsh"
            if shutil.which("powershell"):
                return "powershell"
            return "cmd"
        else:
            if shutil.which("bash"):
                return "bash"
            return "sh"


# Global instance for FunctionTool usage
_swe_terminal_tool = SWETerminalTool()


# FunctionTool-compatible functions
def create_terminal_session(
    session_id: Optional[str] = None,
    working_directory: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Create a new terminal session."""
    return _swe_terminal_tool.create_session(session_id, working_directory, environment)


def execute_terminal_command(
    session_id: str,
    command: str,
    wait_for_completion: bool = True,
    timeout_seconds: Optional[float] = None,
    working_directory: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Execute a command in the specified session."""
    return _swe_terminal_tool.execute_command(
        session_id, command, wait_for_completion, timeout_seconds, 
        working_directory, environment
    )


def execute_multiple_commands(
    session_id: str,
    commands: List[str],
    wait_between_commands: bool = True,
    timeout_seconds: Optional[float] = None,
    working_directory: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Execute multiple commands in sequence."""
    return _swe_terminal_tool.execute_multiple_commands(
        session_id, commands, wait_between_commands, timeout_seconds,
        working_directory, environment
    )


def get_session_status(session_id: str) -> Dict[str, Any]:
    """Get the status of a session."""
    return _swe_terminal_tool.get_session_status(session_id)


def get_command_history(
    session_id: str,
    limit: int = 50,
    command_type: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Get command history for a session."""
    return _swe_terminal_tool.get_command_history(session_id, limit, command_type, status)


def change_working_directory(session_id: str, new_directory: str) -> Dict[str, Any]:
    """Change working directory for a session."""
    return _swe_terminal_tool.change_working_directory(session_id, new_directory)


def list_terminal_sessions() -> Dict[str, Any]:
    """List all active sessions."""
    return _swe_terminal_tool.list_sessions()


def close_terminal_session(session_id: str) -> Dict[str, Any]:
    """Close a session and clean up resources."""
    return _swe_terminal_tool.close_session(session_id)
