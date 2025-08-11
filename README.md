# Multi-Agent Shell Tool

A powerful Python shell tool designed for multi-agent orchestration systems, providing secure command execution, permission management, resource monitoring, and comprehensive audit trails.

## Features

### 🔐 **Agent-Aware Execution**
- **Permission-based access control** with 5 levels: Read-Only, Limited, Standard, Elevated, Admin
- **Command safety validation** that classifies commands by risk level
- **Agent registration and management** with unique identifiers and metadata

### 🛡️ **Security & Safety**
- **Command classification** into Safe, Low-Risk, Medium-Risk, High-Risk, and Restricted levels
- **Automatic blocking** of dangerous commands (e.g., `rm -rf /`, `format C:`)
- **Permission validation** before command execution

### 📊 **Resource Monitoring**
- **Real-time monitoring** of CPU, memory, disk I/O, and network usage
- **Resource limits** and thresholds for agent operations
- **Performance tracking** with execution duration and resource consumption

### 🔄 **Multi-Agent Coordination**
- **Session management** for persistent shell state across agent interactions
- **Concurrent agent support** with thread-safe operations
- **Workflow orchestration** for complex multi-step operations

### 📝 **Audit & Compliance**
- **Comprehensive audit trails** for all command executions
- **Filtered audit queries** by agent, session, time range, and success status
- **Compliance reporting** with detailed execution metadata

### 🚀 **Enhanced Functionality**
- **Cross-platform support** (Windows, macOS, Linux)
- **Function calling interface** for seamless integration with AI agents
- **Error handling and recovery** with detailed error messages
- **Timeout management** for long-running commands

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **Core**: `psutil>=5.9.0` for system resource monitoring
- **Optional**: `pywinpty>=0.5.7` for enhanced Windows support
- **Development**: `pytest`, `black`, `flake8`, `mypy` for testing and code quality

## Quick Start

### Basic Usage

```python
from agent_tools.multi_agent_shell_tool import MultiAgentShellTool, AgentPermission

# Create the tool
shell_tool = MultiAgentShellTool(
    max_concurrent_agents=10,
    enable_audit_logging=True,
    audit_log_file="audit.log"
)

# Register an agent
shell_tool.register_agent(
    agent_id="dev-agent-1",
    agent_name="Development Agent 1",
    permissions=AgentPermission.STANDARD
)

# Execute a command
result = shell_tool.call({
    "command": "ls -la",
    "agent_id": "dev-agent-1",
    "agent_name": "Development Agent 1"
})

print(f"Command executed: {result['success']}")
print(f"Output: {result['stdout']}")
```

### Permission Levels

```python
from agent_tools.multi_agent_shell_tool import AgentPermission

# Different permission levels
agents = [
    ("monitor", "Monitoring Agent", AgentPermission.LIMITED),      # Safe + Low-Risk
    ("dev", "Development Agent", AgentPermission.STANDARD),       # Safe + Low-Risk + Medium-Risk
    ("deploy", "Deployment Agent", AgentPermission.ELEVATED),     # Safe + Low-Risk + Medium-Risk + High-Risk
    ("admin", "Admin Agent", AgentPermission.ADMIN)               # All commands (except Restricted)
]
```

## Command Safety Classification

The tool automatically classifies commands by risk level:

| Level | Description | Examples | Default Permission Required |
|-------|-------------|----------|----------------------------|
| **Safe** | Read-only, no system impact | `ls`, `pwd`, `whoami` | Read-Only |
| **Low-Risk** | Minimal system impact | `git status`, `docker ps` | Limited |
| **Medium-Risk** | Moderate system impact | `kubectl get pods`, `docker build` | Standard |
| **High-Risk** | Significant system impact | `sudo systemctl restart`, `docker run` | Elevated |
| **Restricted** | Dangerous, always blocked | `rm -rf /`, `format C:` | None (blocked) |

## Advanced Features

### Resource Monitoring

```python
# Enable resource monitoring
shell_tool = MultiAgentShellTool(enable_resource_monitoring=True)

# Execute command with monitoring
result = shell_tool.call({
    "command": "docker build .",
    "agent_id": "build-agent",
    "agent_name": "Build Agent"
})

# Check resource usage
resource_usage = result['resource_usage']
print(f"CPU change: {resource_usage['cpu_percent_change']:.2f}%")
print(f"Memory change: {resource_usage['memory_percent_change']:.2f}%")
```

### Audit Trail

```python
# Get comprehensive audit trail
audit_entries = shell_tool.get_audit_trail(
    agent_id="dev-agent-1",
    start_time=time.time() - 3600,  # Last hour
    limit=50
)

for entry in audit_entries:
    print(f"{entry['command']} -> {'✓' if entry['success'] else '✗'}")
    print(f"  Safety: {entry['safety_level']}")
    print(f"  Duration: {entry['duration_seconds']:.2f}s")
```

### Multi-Agent Workflows

```python
# Simulate a deployment workflow
workflow = [
    ("dev-agent", "git pull origin main"),
    ("build-agent", "docker build -t app:latest ."),
    ("test-agent", "docker run --rm app:latest npm test"),
    ("deploy-agent", "kubectl apply -f k8s/"),
    ("monitor-agent", "kubectl get pods")
]

for agent_id, command in workflow:
    try:
        result = shell_tool.call({
            "command": command,
            "agent_id": agent_id,
            "agent_name": f"{agent_id.title()} Agent"
        })
        print(f"✓ {agent_id}: {command}")
    except Exception as e:
        print(f"✗ {agent_id}: {e}")
```

## Configuration Options

```python
shell_tool = MultiAgentShellTool(
    # Agent management
    max_concurrent_agents=20,
    
    # Session management
    enable_persistent_sessions=True,
    session_idle_timeout_seconds=1800,  # 30 minutes
    
    # Audit and logging
    enable_audit_logging=True,
    audit_log_file="audit.log",
    log_level="INFO",
    
    # Resource monitoring
    enable_resource_monitoring=True,
    resource_limits={
        "max_cpu_percent": 80.0,
        "max_memory_percent": 90.0,
        "max_disk_io_mb": 1000
    },
    
    # Command validation
    enable_command_validation=True
)
```

## Error Handling

The tool provides comprehensive error handling:

```python
try:
    result = shell_tool.call({
        "command": "sudo systemctl restart ssh",
        "agent_id": "monitor-agent",
        "agent_name": "Monitoring Agent"
    })
except PermissionError as e:
    print(f"Permission denied: {e}")
    # Agent lacks sufficient permissions for this command
except ValueError as e:
    print(f"Invalid request: {e}")
    # Agent not registered or invalid parameters
except Exception as e:
    print(f"Execution error: {e}")
    # Command execution failed
```

## Examples

### Basic Demo
```bash
python examples/demo_multi_agent_shell_tool.py
```

### Shell Tool Demo
```bash
python examples/demo_shell_tool.py
```

### Streaming Demo
```bash
python examples/demo_shell_stream.py
```

## API Reference

### Core Classes

- **`MultiAgentShellTool`**: Main tool class for multi-agent shell operations
- **`AgentPermission`**: Enumeration of agent permission levels
- **`CommandValidator`**: Static class for command safety validation
- **`ResourceMonitor`**: Class for monitoring system resources during execution

### Key Methods

- **`register_agent()`**: Register a new agent with permissions
- **`call()`**: Execute a shell command with agent context
- **`get_audit_trail()`**: Retrieve filtered audit trail entries
- **`get_system_stats()`**: Get current system statistics
- **`cleanup()`**: Clean up resources and stop the tool

## Security Considerations

1. **Command Validation**: All commands are validated against safety rules
2. **Permission Enforcement**: Commands require appropriate agent permissions
3. **Resource Limits**: Configurable limits prevent resource exhaustion
4. **Audit Logging**: Complete audit trail for compliance and debugging
5. **Session Isolation**: Agent sessions are isolated and managed securely

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please open an issue on the repository or contact the development team.


