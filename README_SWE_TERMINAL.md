# SWE Terminal Tool

A stateful terminal tool for multi-agent orchestration systems that can execute commands like git, maven, and other development tools using subprocess. The tool maintains state and allows agents to pass multiple commands that wait for subsequent commands.

## Features

- **Stateful Sessions**: Maintains working directory, environment variables, and command history across multiple command executions
- **Multi-Command Execution**: Execute multiple commands in sequence with proper working directory management
- **Development Tool Support**: Built-in support for git, maven, npm, docker, kubectl, and other development tools
- **Cross-Platform**: Works on Windows, macOS, and Linux with automatic shell detection
- **Thread-Safe**: Designed for multi-agent environments with proper locking mechanisms
- **Google ADK Compatible**: Can be used as FunctionTool with Google Agent Development Kit
- **Comprehensive Logging**: Full command history with execution details, timing, and error handling

## Architecture

The SWE Terminal Tool is built around the concept of **sessions** that maintain state:

- **TerminalState**: Contains working directory, environment, command history, and active processes
- **CommandResult**: Detailed information about each command execution including timing and output
- **CommandType**: Automatic detection of command types (git, maven, shell, etc.)
- **CommandStatus**: Tracking of command execution states (pending, running, completed, failed, timeout)

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from agent_tools.swe_terminal_tool import (
    create_terminal_session,
    execute_terminal_command,
    execute_multiple_commands
)

# Create a new session
result = create_terminal_session()
session_id = result["session_id"]

# Execute a single command
result = execute_terminal_command(session_id, "git --version")
print(f"Git version: {result['stdout']}")

# Execute multiple commands
commands = [
    "echo 'Starting build'",
    "mvn clean compile",
    "echo 'Build completed'"
]
result = execute_multiple_commands(session_id, commands)
```

### Session Management

```python
from agent_tools.swe_terminal_tool import (
    create_terminal_session,
    change_working_directory,
    get_session_status,
    close_terminal_session
)

# Create session with custom parameters
session = create_terminal_session(
    session_id="build-session",
    working_directory="/path/to/project",
    environment={"JAVA_HOME": "/usr/lib/jvm/java-11"}
)

# Change working directory
change_working_directory(session["session_id"], "/path/to/subdirectory")

# Get session status
status = get_session_status(session["session_id"])
print(f"Commands executed: {status['total_commands']}")

# Close session when done
close_terminal_session(session["session_id"])
```

### Command History and Filtering

```python
from agent_tools.swe_terminal_tool import get_command_history

# Get recent command history
history = get_command_history(session_id, limit=10)

# Filter by command type
git_commands = get_command_history(session_id, command_type="git")

# Filter by status
failed_commands = get_command_history(session_id, status="failed")
```

## API Reference

### Core Functions

#### `create_terminal_session(session_id=None, working_directory=None, environment=None)`

Creates a new terminal session.

- **session_id**: Optional custom session ID (auto-generated if not provided)
- **working_directory**: Initial working directory (defaults to current directory)
- **environment**: Additional environment variables to merge with system environment

Returns: Dictionary with session information

#### `execute_terminal_command(session_id, command, wait_for_completion=True, timeout_seconds=None, working_directory=None, environment=None)`

Executes a single command in the specified session.

- **session_id**: Session ID to execute command in
- **command**: Command string to execute
- **wait_for_completion**: Whether to wait for command completion
- **timeout_seconds**: Command timeout in seconds
- **working_directory**: Override working directory for this command
- **environment**: Override environment variables for this command

Returns: Dictionary with command execution result

#### `execute_multiple_commands(session_id, commands, wait_between_commands=True, timeout_seconds=None, working_directory=None, environment=None)`

Executes multiple commands in sequence.

- **session_id**: Session ID to execute commands in
- **commands**: List of command strings
- **wait_between_commands**: Whether to wait between commands
- **timeout_seconds**: Timeout for each command
- **working_directory**: Override working directory
- **environment**: Override environment variables

Returns: Dictionary with execution results for all commands

### Management Functions

#### `get_session_status(session_id)`

Gets the current status of a session.

#### `get_command_history(session_id, limit=50, command_type=None, status=None)`

Retrieves command history with optional filtering.

#### `change_working_directory(session_id, new_directory)`

Changes the working directory for a session.

#### `list_terminal_sessions()`

Lists all active sessions.

#### `close_terminal_session(session_id)`

Closes a session and cleans up resources.

## Command Type Detection

The tool automatically detects command types based on the base command:

- **git**: `git`, `git.exe`
- **maven**: `mvn`, `mvnw`, `mvn.cmd`
- **npm**: `npm`, `npx`, `yarn`
- **docker**: `docker`, `docker-compose`
- **kubectl**: `kubectl`, `kubectl.exe`
- **shell**: `ls`, `cd`, `pwd`, `echo`, `cat`, `grep`, `find`
- **custom**: Any other command

## Working Directory Management

The tool intelligently manages working directories:

- Commands that change directory (like `cd`) automatically update the session's working directory
- Subsequent commands in multi-command sequences use the updated working directory
- The `change_working_directory()` function provides explicit directory management
- All paths are resolved to absolute paths for consistency

## Error Handling

The tool provides comprehensive error handling:

- **Command Execution Errors**: Captures stderr and return codes
- **Timeout Handling**: Configurable timeouts with graceful termination
- **Invalid Commands**: Graceful handling of non-existent commands
- **Session Validation**: Prevents operations on invalid or closed sessions
- **Resource Cleanup**: Automatic cleanup of processes and temporary resources

## Thread Safety

The SWE Terminal Tool is designed for multi-agent environments:

- **Global Lock**: Protects session creation/deletion operations
- **Session Locks**: Individual locks for each session's command execution
- **Process Management**: Safe handling of concurrent command execution
- **State Consistency**: Ensures consistent state across multiple threads

## Google ADK Integration

The tool is designed to work with Google Agent Development Kit:

```python
from google.generativeai.types import FunctionTool
from agent_tools.swe_terminal_tool import execute_terminal_command

# Create FunctionTool for ADK
terminal_tool = FunctionTool(
    name="execute_terminal_command",
    description="Execute a terminal command in a session",
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session ID"},
            "command": {"type": "string", "description": "Command to execute"},
            "wait_for_completion": {"type": "boolean", "description": "Wait for completion"}
        },
        "required": ["session_id", "command"]
    }
)

# Use in ADK agent
agent = LlmAgent(
    tools=[terminal_tool],
    # ... other configuration
)
```

## Examples

### Git Workflow

```python
# Initialize git repository and make first commit
commands = [
    "git init",
    "git config user.name 'SWE Agent'",
    "git config user.email 'agent@example.com'",
    "echo '# My Project' > README.md",
    "git add README.md",
    "git commit -m 'Initial commit'"
]

result = execute_multiple_commands(session_id, commands)
```

### Maven Build Process

```python
# Clean and build Maven project
commands = [
    "mvn clean",
    "mvn compile",
    "mvn test",
    "mvn package"
]

result = execute_multiple_commands(session_id, commands)
```

### Docker Operations

```python
# Build and run Docker container
commands = [
    "docker build -t myapp .",
    "docker run -d -p 8080:8080 myapp",
    "docker ps"
]

result = execute_multiple_commands(session_id, commands)
```

## Testing

Run the comprehensive test suite:

```bash
python -m examples.test_swe_terminal_tool
```

Run the demo script:

```bash
python -m examples.demo_swe_terminal_tool
```

## Performance Considerations

- **Command History**: Limited to 100 commands per session by default (configurable)
- **Timeout**: Default 5-minute timeout for commands (configurable)
- **Memory Usage**: Command output is stored in memory; consider limits for long-running processes
- **Concurrent Sessions**: Each session maintains its own state and locks

## Security Considerations

- **Command Injection**: Commands are executed as-is; validate inputs in your application layer
- **Environment Variables**: Sensitive environment variables should be managed carefully
- **Working Directory**: Ensure proper path validation to prevent directory traversal
- **Process Management**: Long-running processes are tracked and can be terminated

## Troubleshooting

### Common Issues

1. **Command Not Found**: Ensure the command is available in the system PATH
2. **Permission Denied**: Check if the tool has permission to execute the command
3. **Working Directory Issues**: Verify the directory exists and is accessible
4. **Session Not Found**: Ensure the session ID is valid and the session hasn't been closed

### Debug Information

Enable debug logging by checking the command results:

```python
result = execute_terminal_command(session_id, "your_command")
if not result["success"]:
    print(f"Error: {result['error']}")
    print(f"Session ID: {result['session_id']}")
    print(f"Command: {result['command']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test examples
3. Open an issue on GitHub
4. Check the command history for error details
