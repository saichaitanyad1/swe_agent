#!/usr/bin/env python3
"""
Demo script for SWE Terminal Tool

This script demonstrates basic usage of the SWE terminal tool for
multi-agent orchestration systems.

Usage:
    python -m examples.demo_swe_terminal_tool
"""

import os
import sys
import time

# Import the SWE terminal tool functions
from agent_tools.swe_terminal_tool import (
    create_terminal_session,
    execute_terminal_command,
    execute_multiple_commands,
    get_session_status,
    get_command_history,
    change_working_directory,
    list_terminal_sessions,
    close_terminal_session
)


def demo_basic_usage():
    """Demonstrate basic usage of the SWE terminal tool."""
    print("=" * 60)
    print(" SWE Terminal Tool Demo - Basic Usage")
    print("=" * 60)
    
    # Create a new session
    print("\n1. Creating a new terminal session...")
    result = create_terminal_session()
    if not result["success"]:
        print(f"✗ Failed to create session: {result['error']}")
        return None
    
    session_id = result["session_id"]
    print(f"✓ Session created: {session_id}")
    print(f"  Working directory: {result['working_directory']}")
    
    # Execute a simple command
    print("\n2. Executing a simple command...")
    cmd_result = execute_terminal_command(session_id, "echo 'Hello from SWE Agent!'")
    if cmd_result["success"]:
        print(f"✓ Command executed successfully")
        print(f"  Command: {cmd_result['command']}")
        print(f"  Output: {cmd_result['stdout'].strip()}")
        print(f"  Duration: {cmd_result['duration']:.3f}s")
    else:
        print(f"✗ Command failed: {cmd_result['error']}")
    
    # Execute multiple commands
    print("\n3. Executing multiple commands...")
    commands = [
        "echo 'Starting multi-command sequence'",
        "echo 'Current directory:'",
        "dir" if os.name == 'nt' else "pwd",
        "echo 'Directory contents:'",
        "dir" if os.name == 'nt' else "ls -la"
    ]
    
    multi_result = execute_multiple_commands(session_id, commands)
    if multi_result["success"]:
        print(f"✓ Multiple commands executed successfully")
        print(f"  Total commands: {multi_result['total_commands']}")
        print(f"  Final working directory: {multi_result['final_working_directory']}")
        
        # Show results of each command
        for i, cmd_result in enumerate(multi_result["results"]):
            print(f"  Command {i+1}: {cmd_result['command']}")
            print(f"    Status: {cmd_result['status']}")
            print(f"    Output: {cmd_result['stdout'].strip()[:50]}...")
    else:
        print(f"✗ Multiple commands failed: {multi_result['error']}")
    
    # Get session status
    print("\n4. Getting session status...")
    status = get_session_status(session_id)
    if status["success"]:
        print(f"✓ Session status retrieved")
        print(f"  Total commands: {status['total_commands']}")
        print(f"  Status breakdown: {status['status_counts']}")
        print(f"  Working directory: {status['working_directory']}")
    
    # Get command history
    print("\n5. Getting command history...")
    history = get_command_history(session_id, limit=5)
    if history["success"]:
        print(f"✓ Command history retrieved")
        print(f"  Recent commands:")
        for cmd in history["commands"]:
            print(f"    - {cmd['command']} -> {cmd['status']}")
    
    return session_id


def demo_git_operations(session_id):
    """Demonstrate git operations."""
    print("\n" + "=" * 60)
    print(" SWE Terminal Tool Demo - Git Operations")
    print("=" * 60)
    
    # Check git version
    print("\n1. Checking git version...")
    git_version = execute_terminal_command(session_id, "git --version")
    if git_version["success"]:
        print(f"✓ Git version: {git_version['stdout'].strip()}")
        print(f"  Command type: {git_version['command_type']}")
    else:
        print(f"✗ Git not available: {git_version['stderr']}")
        return
    
    # Check if we're in a git repository
    print("\n2. Checking git status...")
    git_status = execute_terminal_command(session_id, "git status")
    if git_status["success"]:
        print(f"✓ Git status retrieved")
        print(f"  Output: {git_status['stdout'].strip()[:100]}...")
    else:
        print(f"✓ Not in a git repository (expected): {git_status['stderr']}")
    
    # Show git help
    print("\n3. Getting git help...")
    git_help = execute_terminal_command(session_id, "git help")
    if git_help["success"]:
        print(f"✓ Git help retrieved")
        print(f"  Output length: {len(git_help['stdout'])} characters")
        print(f"  Duration: {git_help['duration']:.3f}s")
    else:
        print(f"✗ Git help failed: {git_help['error']}")


def demo_maven_operations(session_id):
    """Demonstrate Maven operations."""
    print("\n" + "=" * 60)
    print(" SWE Terminal Tool Demo - Maven Operations")
    print("=" * 60)
    
    # Check Maven version
    print("\n1. Checking Maven version...")
    mvn_version = execute_terminal_command(session_id, "mvn --version")
    if mvn_version["success"]:
        print(f"✓ Maven version retrieved")
        print(f"  Command type: {mvn_version['command_type']}")
        print(f"  Output: {mvn_version['stdout'].strip()[:100]}...")
        print(f"  Duration: {mvn_version['duration']:.3f}s")
    else:
        print(f"✓ Maven not available (expected): {mvn_version['stderr']}")
    
    # Try Maven help
    print("\n2. Getting Maven help...")
    mvn_help = execute_terminal_command(session_id, "mvn help")
    if mvn_help["success"]:
        print(f"✓ Maven help retrieved")
        print(f"  Output length: {len(mvn_help['stdout'])} characters")
    else:
        print(f"✓ Maven help failed as expected: {mvn_help['stderr']}")


def demo_working_directory_management(session_id):
    """Demonstrate working directory management."""
    print("\n" + "=" * 60)
    print(" SWE Terminal Tool Demo - Working Directory Management")
    print("=" * 60)
    
    # Get current working directory
    print("\n1. Getting current working directory...")
    pwd_result = execute_terminal_command(session_id, "cd" if os.name == 'nt' else "pwd")
    if pwd_result["success"]:
        current_dir = pwd_result["stdout"].strip()
        print(f"✓ Current directory: {current_dir}")
    else:
        print(f"✗ Failed to get current directory: {pwd_result['error']}")
        return
    
    # Create a temporary directory and change to it
    print("\n2. Creating and changing to temporary directory...")
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="swe_demo_")
    
    change_result = change_working_directory(session_id, temp_dir)
    if change_result["success"]:
        print(f"✓ Changed working directory")
        print(f"  From: {change_result['old_directory']}")
        print(f"  To: {change_result['new_directory']}")
        
        # Verify the change
        verify_result = execute_terminal_command(session_id, "echo 'Verifying directory change'")
        if verify_result["success"]:
            print(f"✓ Command executed in new directory: {verify_result['working_directory']}")
    else:
        print(f"✗ Failed to change directory: {change_result['error']}")
    
    # Change back to original directory
    print("\n3. Changing back to original directory...")
    change_back = change_working_directory(session_id, current_dir)
    if change_back["success"]:
        print(f"✓ Changed back to original directory: {change_back['new_directory']}")
    else:
        print(f"✗ Failed to change back: {change_back['error']}")
    
    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary directory")
    except Exception as e:
        print(f"⚠ Warning: Could not clean up temp directory: {e}")


def demo_session_management():
    """Demonstrate session management features."""
    print("\n" + "=" * 60)
    print(" SWE Terminal Tool Demo - Session Management")
    print("=" * 60)
    
    # List current sessions
    print("\n1. Listing current sessions...")
    sessions = list_terminal_sessions()
    if sessions["success"]:
        print(f"✓ Current sessions: {sessions['total_sessions']}")
        for session in sessions["sessions"]:
            print(f"  - {session['session_id']}: {session['working_directory']}")
    else:
        print(f"✗ Failed to list sessions: {sessions['error']}")
    
    # Create multiple sessions
    print("\n2. Creating multiple sessions...")
    session_ids = []
    for i in range(3):
        result = create_terminal_session(session_id=f"demo-session-{i}")
        if result["success"]:
            session_ids.append(result["session_id"])
            print(f"  ✓ Created session {i+1}: {result['session_id']}")
        else:
            print(f"  ✗ Failed to create session {i+1}: {result['error']}")
    
    # List sessions again
    print("\n3. Listing sessions after creation...")
    sessions_after = list_terminal_sessions()
    if sessions_after["success"]:
        print(f"✓ Total sessions now: {sessions_after['total_sessions']}")
    
    # Close demo sessions
    print("\n4. Closing demo sessions...")
    for session_id in session_ids:
        close_result = close_terminal_session(session_id)
        if close_result["success"]:
            print(f"  ✓ Closed session: {session_id}")
        else:
            print(f"  ✗ Failed to close session {session_id}: {close_result['error']}")
    
    # Final session count
    final_sessions = list_terminal_sessions()
    if final_sessions["success"]:
        print(f"\n✓ Final session count: {final_sessions['total_sessions']}")


def main():
    """Main demo function."""
    print("SWE Terminal Tool Demo")
    print("This demo showcases the capabilities of the SWE terminal tool")
    print("for multi-agent orchestration systems.")
    print("=" * 80)
    
    try:
        # Run basic usage demo
        session_id = demo_basic_usage()
        if not session_id:
            print("✗ Basic usage demo failed, aborting other demos")
            return False
        
        # Run other demos
        demo_git_operations(session_id)
        demo_maven_operations(session_id)
        demo_working_directory_management(session_id)
        
        # Session management demo (creates its own sessions)
        demo_session_management()
        
        # Clean up main session
        print(f"\nCleaning up main demo session {session_id}...")
        cleanup_result = close_terminal_session(session_id)
        if cleanup_result["success"]:
            print(f"✓ Main demo session closed successfully")
        else:
            print(f"✗ Failed to close main demo session: {cleanup_result['error']}")
        
        print("\n" + "=" * 80)
        print(" Demo Completed Successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
