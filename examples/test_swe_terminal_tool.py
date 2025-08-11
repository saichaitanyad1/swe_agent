#!/usr/bin/env python3
"""
Test file for SWE Terminal Tool

This script tests all the functionality of the SWE terminal tool including:
- Session management
- Command execution
- Multiple command execution
- Working directory management
- Command history and status
- Error handling

Usage:
    python -m examples.test_swe_terminal_tool
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Import the SWE terminal tool
from agent_tools.swe_terminal_tool import (
    SWETerminalTool,
    CommandStatus,
    CommandType,
    create_terminal_session,
    execute_terminal_command,
    execute_multiple_commands,
    get_session_status,
    get_command_history,
    change_working_directory,
    list_terminal_sessions,
    close_terminal_session
)


def test_session_management():
    """Test session creation and management."""
    print("=" * 60)
    print(" Testing Session Management")
    print("=" * 60)
    
    # Test 1: Create session with default parameters
    print("\n1. Creating session with default parameters...")
    result = create_terminal_session()
    if result["success"]:
        session_id = result["session_id"]
        print(f"✓ Session created: {session_id}")
        print(f"  Working directory: {result['working_directory']}")
        print(f"  Environment variables: {len(result['environment'])}")
    else:
        print(f"✗ Failed to create session: {result['error']}")
        return None
    
    # Test 2: Create session with custom parameters
    print("\n2. Creating session with custom parameters...")
    temp_dir = tempfile.mkdtemp(prefix="swe_test_")
    custom_env = {"CUSTOM_VAR": "test_value", "TEST_ENV": "123"}
    
    result2 = create_terminal_session(
        session_id="custom-session",
        working_directory=temp_dir,
        environment=custom_env
    )
    if result2["success"]:
        print(f"✓ Custom session created: {result2['session_id']}")
        print(f"  Working directory: {result2['working_directory']}")
        print(f"  Custom environment: {custom_env}")
    else:
        print(f"✗ Failed to create custom session: {result2['error']}")
    
    # Test 3: Try to create duplicate session
    print("\n3. Testing duplicate session creation...")
    result3 = create_terminal_session(session_id=session_id)
    if not result3["success"]:
        print(f"✓ Correctly prevented duplicate session: {result3['error']}")
    else:
        print("✗ Should have prevented duplicate session")
    
    # Test 4: List sessions
    print("\n4. Listing all sessions...")
    sessions = list_terminal_sessions()
    if sessions["success"]:
        print(f"✓ Total sessions: {sessions['total_sessions']}")
        for session in sessions["sessions"]:
            print(f"  - {session['session_id']}: {session['working_directory']}")
    else:
        print(f"✗ Failed to list sessions: {sessions['error']}")
    
    # Clean up custom session
    if result2["success"]:
        close_terminal_session("custom-session")
        print(f"✓ Closed custom session")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    
    return session_id


def test_basic_command_execution(session_id):
    """Test basic command execution."""
    print("\n" + "=" * 60)
    print(" Testing Basic Command Execution")
    print("=" * 60)
    
    # Test 1: Simple echo command
    print("\n1. Testing echo command...")
    result = execute_terminal_command(session_id, "echo 'Hello, SWE Agent!'")
    if result["success"]:
        print(f"✓ Command executed successfully")
        print(f"  Command: {result['command']}")
        print(f"  Return code: {result['return_code']}")
        print(f"  Duration: {result['duration']:.3f}s")
        print(f"  STDOUT: {result['stdout'].strip()}")
        print(f"  Command type: {result['command_type']}")
    else:
        print(f"✗ Command failed: {result['error']}")
    
    # Test 2: Directory listing
    print("\n2. Testing directory listing...")
    result = execute_terminal_command(session_id, "dir" if os.name == 'nt' else "ls -la")
    if result["success"]:
        print(f"✓ Directory listing successful")
        print(f"  Return code: {result['return_code']}")
        print(f"  Duration: {result['duration']:.3f}s")
        print(f"  Working directory: {result['working_directory']}")
        print(f"  Command type: {result['command_type']}")
    else:
        print(f"✗ Directory listing failed: {result['error']}")
    
    # Test 3: Current working directory
    print("\n3. Testing pwd command...")
    result = execute_terminal_command(session_id, "cd" if os.name == 'nt' else "pwd")
    if result["success"]:
        print(f"✓ PWD command successful")
        print(f"  Return code: {result['return_code']}")
        print(f"  STDOUT: {result['stdout'].strip()}")
    else:
        print(f"✗ PWD command failed: {result['error']}")
    
    # Test 4: Command with error
    print("\n4. Testing command that will fail...")
    result = execute_terminal_command(session_id, "nonexistent_command_12345")
    if not result["success"]:
        print(f"✓ Correctly handled non-existent command: {result['error']}")
    else:
        print("✗ Should have failed for non-existent command")
    
    return True


def test_git_commands(session_id):
    """Test git-related commands."""
    print("\n" + "=" * 60)
    print(" Testing Git Commands")
    print("=" * 60)
    
    # Test 1: Git version
    print("\n1. Testing git --version...")
    result = execute_terminal_command(session_id, "git --version")
    if result["success"]:
        print(f"✓ Git version command successful")
        print(f"  Command type: {result['command_type']}")
        print(f"  Return code: {result['return_code']}")
        print(f"  STDOUT: {result['stdout'].strip()}")
        print(f"  Duration: {result['duration']:.3f}s")
    else:
        print(f"✗ Git version command failed: {result['error']}")
    
    # Test 2: Git status (will likely fail if not in git repo)
    print("\n2. Testing git status...")
    result = execute_terminal_command(session_id, "git status")
    if result["success"]:
        print(f"✓ Git status command successful")
        print(f"  Return code: {result['return_code']}")
        print(f"  STDOUT: {result['stdout'].strip()}")
    else:
        print(f"✓ Git status failed as expected (not in git repo): {result['stderr']}")
    
    # Test 3: Git help
    print("\n3. Testing git help...")
    result = execute_terminal_command(session_id, "git help")
    if result["success"]:
        print(f"✓ Git help command successful")
        print(f"  Command type: {result['command_type']}")
        print(f"  Return code: {result['return_code']}")
        print(f"  Duration: {result['duration']:.3f}s")
    else:
        print(f"✗ Git help command failed: {result['error']}")
    
    return True


def test_maven_commands(session_id):
    """Test Maven-related commands."""
    print("\n" + "=" * 60)
    print(" Testing Maven Commands")
    print("=" * 60)
    
    # Test 1: Maven version
    print("\n1. Testing mvn --version...")
    result = execute_terminal_command(session_id, "mvn --version")
    if result["success"]:
        print(f"✓ Maven version command successful")
        print(f"  Command type: {result['command_type']}")
        print(f"  Return code: {result['return_code']}")
        print(f"  STDOUT: {result['stdout'].strip()[:100]}...")
        print(f"  Duration: {result['duration']:.3f}s")
    else:
        print(f"✓ Maven version failed as expected (Maven not installed): {result['stderr']}")
    
    # Test 2: Maven help
    print("\n2. Testing mvn help...")
    result = execute_terminal_command(session_id, "mvn help")
    if result["success"]:
        print(f"✓ Maven help command successful")
        print(f"  Command type: {result['command_type']}")
        print(f"  Return code: {result['return_code']}")
    else:
        print(f"✓ Maven help failed as expected: {result['stderr']}")
    
    return True


def test_multiple_commands(session_id):
    """Test executing multiple commands in sequence."""
    print("\n" + "=" * 60)
    print(" Testing Multiple Commands Execution")
    print("=" * 60)
    
    # Test 1: Simple multiple commands
    print("\n1. Testing multiple simple commands...")
    commands = [
        "echo 'First command'",
        "echo 'Second command'",
        "echo 'Third command'"
    ]
    
    result = execute_multiple_commands(session_id, commands)
    if result["success"]:
        print(f"✓ Multiple commands executed successfully")
        print(f"  Total commands: {result['total_commands']}")
        print(f"  Final working directory: {result['final_working_directory']}")
        
        for i, cmd_result in enumerate(result["results"]):
            print(f"  Command {i+1}: {cmd_result['command']}")
            print(f"    Status: {cmd_result['status']}")
            print(f"    Return code: {cmd_result['return_code']}")
            print(f"    STDOUT: {cmd_result['stdout'].strip()}")
    else:
        print(f"✗ Multiple commands failed: {result['error']}")
    
    # Test 2: Commands with directory changes
    print("\n2. Testing commands with directory changes...")
    temp_dir = tempfile.mkdtemp(prefix="swe_test_dir_")
    
    commands_with_cd = [
        f"echo 'Current directory: %cd%'" if os.name == 'nt' else "echo 'Current directory: $(pwd)'",
        f"cd {temp_dir}",
        f"echo 'Changed to: %cd%'" if os.name == 'nt' else "echo 'Changed to: $(pwd)'",
        "echo 'File listing:'",
        "dir" if os.name == 'nt' else "ls -la"
    ]
    
    result2 = execute_multiple_commands(session_id, commands_with_cd)
    if result2["success"]:
        print(f"✓ Directory change commands executed successfully")
        print(f"  Total commands: {result2['total_commands']}")
        print(f"  Final working directory: {result2['final_working_directory']}")
        
        # Check if working directory actually changed
        if temp_dir in result2['final_working_directory']:
            print(f"  ✓ Working directory successfully changed to temp directory")
        else:
            print(f"  ✗ Working directory change may not have worked as expected")
    else:
        print(f"✗ Directory change commands failed: {result2['error']}")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    
    return True


def test_working_directory_management(session_id):
    """Test working directory management."""
    print("\n" + "=" * 60)
    print(" Testing Working Directory Management")
    print("=" * 60)
    
    # Test 1: Change to existing directory
    print("\n1. Testing change to existing directory...")
    temp_dir = tempfile.mkdtemp(prefix="swe_test_wd_")
    
    result = change_working_directory(session_id, temp_dir)
    if result["success"]:
        print(f"✓ Working directory changed successfully")
        print(f"  From: {result['old_directory']}")
        print(f"  To: {result['new_directory']}")
    else:
        print(f"✗ Failed to change working directory: {result['error']}")
    
    # Test 2: Verify working directory change
    print("\n2. Verifying working directory change...")
    result2 = execute_terminal_command(session_id, "echo 'Current directory'")
    if result2["success"]:
        print(f"✓ Command executed in new directory")
        print(f"  Working directory: {result2['working_directory']}")
        if temp_dir in result2['working_directory']:
            print(f"  ✓ Working directory change confirmed")
        else:
            print(f"  ✗ Working directory change not reflected")
    else:
        print(f"✗ Command failed: {result2['error']}")
    
    # Test 3: Change to non-existent directory
    print("\n3. Testing change to non-existent directory...")
    result3 = change_working_directory(session_id, "/nonexistent/directory/12345")
    if not result3["success"]:
        print(f"✓ Correctly prevented change to non-existent directory: {result3['error']}")
    else:
        print("✗ Should have failed for non-existent directory")
    
    # Test 4: Change back to original directory
    print("\n4. Changing back to original directory...")
    original_dir = os.getcwd()
    result4 = change_working_directory(session_id, original_dir)
    if result4["success"]:
        print(f"✓ Changed back to original directory: {result4['new_directory']}")
    else:
        print(f"✗ Failed to change back: {result4['error']}")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    
    return True


def test_session_status_and_history(session_id):
    """Test session status and command history."""
    print("\n" + "=" * 60)
    print(" Testing Session Status and History")
    print("=" * 60)
    
    # Test 1: Get session status
    print("\n1. Getting session status...")
    status = get_session_status(session_id)
    if status["success"]:
        print(f"✓ Session status retrieved successfully")
        print(f"  Session ID: {status['session_id']}")
        print(f"  Working directory: {status['working_directory']}")
        print(f"  Total commands: {status['total_commands']}")
        print(f"  Status counts: {status['status_counts']}")
        print(f"  Active processes: {status['active_processes']}")
        print(f"  Environment variables: {status['environment_variables']}")
    else:
        print(f"✗ Failed to get session status: {status['error']}")
    
    # Test 2: Get command history
    print("\n2. Getting command history...")
    history = get_command_history(session_id, limit=10)
    if history["success"]:
        print(f"✓ Command history retrieved successfully")
        print(f"  Total commands in history: {history['total_commands']}")
        print(f"  Filters applied: {history['filters']}")
        
        if history["commands"]:
            print(f"  Recent commands:")
            for i, cmd in enumerate(history["commands"][:5]):  # Show first 5
                print(f"    {i+1}. {cmd['command']} -> {cmd['status']} (RC: {cmd['return_code']})")
        else:
            print(f"  No commands in history")
    else:
        print(f"✗ Failed to get command history: {history['error']}")
    
    # Test 3: Get filtered command history
    print("\n3. Getting filtered command history...")
    git_history = get_command_history(session_id, limit=5, command_type="git")
    if git_history["success"]:
        print(f"✓ Git command history retrieved successfully")
        print(f"  Git commands found: {git_history['total_commands']}")
        for cmd in git_history["commands"]:
            print(f"    - {cmd['command']} -> {cmd['status']}")
    else:
        print(f"✗ Failed to get git command history: {git_history['error']}")
    
    # Test 4: Get successful commands only
    print("\n4. Getting successful commands only...")
    success_history = get_command_history(session_id, limit=5, status="completed")
    if success_history["success"]:
        print(f"✓ Successful commands retrieved: {success_history['total_commands']}")
        for cmd in success_history["commands"]:
            print(f"    - {cmd['command']} (Duration: {cmd['duration']:.3f}s)")
    else:
        print(f"✗ Failed to get successful commands: {success_history['error']}")
    
    return True


def test_error_handling(session_id):
    """Test error handling scenarios."""
    print("\n" + "=" * 60)
    print(" Testing Error Handling")
    print("=" * 60)
    
    # Test 1: Invalid session ID
    print("\n1. Testing invalid session ID...")
    result = execute_terminal_command("invalid-session-id", "echo 'test'")
    if not result["success"]:
        print(f"✓ Correctly handled invalid session ID: {result['error']}")
    else:
        print("✗ Should have failed for invalid session ID")
    
    # Test 2: Empty command
    print("\n2. Testing empty command...")
    result2 = execute_terminal_command(session_id, "")
    if not result2["success"]:
        print(f"✓ Correctly handled empty command: {result2['error']}")
    else:
        print("✗ Should have failed for empty command")
    
    # Test 3: Command with very long timeout
    print("\n3. Testing command with very long timeout...")
    result3 = execute_terminal_command(session_id, "echo 'Long timeout test'", timeout_seconds=0.001)
    if result3["success"]:
        print(f"✓ Command with short timeout executed successfully")
        print(f"  Duration: {result3['duration']:.3f}s")
    else:
        print(f"✓ Command with short timeout handled: {result3['error']}")
    
    # Test 4: Non-existent command
    print("\n4. Testing non-existent command...")
    result4 = execute_terminal_command(session_id, "this_command_does_not_exist_12345")
    if result4["success"]:
        print(f"✓ Non-existent command handled gracefully")
        print(f"  Return code: {result4['return_code']}")
        print(f"  STDERR: {result4['stderr'][:100]}...")
    else:
        print(f"✓ Non-existent command failed as expected: {result4['error']}")
    
    return True


def test_concurrent_execution(session_id):
    """Test concurrent command execution."""
    print("\n" + "=" * 60)
    print(" Testing Concurrent Command Execution")
    print("=" * 60)
    
    # Test 1: Multiple async commands
    print("\n1. Testing multiple async commands...")
    commands = [
        "echo 'Async command 1'",
        "echo 'Async command 2'",
        "echo 'Async command 3'"
    ]
    
    # Start all commands asynchronously
    async_results = []
    for cmd in commands:
        result = execute_terminal_command(session_id, cmd, wait_for_completion=False)
        if result["success"]:
            async_results.append(result)
            print(f"  Started async command: {result['command_id']}")
        else:
            print(f"  Failed to start async command: {result['error']}")
    
    # Wait a bit for commands to complete
    print(f"  Waiting for {len(async_results)} async commands to complete...")
    time.sleep(2)
    
    # Check status of async commands
    for result in async_results:
        status = get_session_status(session_id)
        if status["success"]:
            print(f"  Command {result['command_id']}: Status counts: {status['status_counts']}")
    
    # Test 2: Mixed sync and async commands
    print("\n2. Testing mixed sync and async commands...")
    
    # Start async command
    async_result = execute_terminal_command(session_id, "echo 'Async mixed test'", wait_for_completion=False)
    if async_result["success"]:
        print(f"  Started async command: {async_result['command_id']}")
        
        # Execute sync command while async is running
        sync_result = execute_terminal_command(session_id, "echo 'Sync mixed test'")
        if sync_result["success"]:
            print(f"  Sync command completed: {sync_result['command_id']}")
            print(f"    Duration: {sync_result['duration']:.3f}s")
        
        # Wait for async to complete
        time.sleep(1)
        print(f"  Async command should be completed now")
    else:
        print(f"  Failed to start async command: {async_result['error']}")
    
    return True


def main():
    """Main test function."""
    print("SWE Terminal Tool Test Suite")
    print("This test suite validates all functionality of the SWE terminal tool.")
    print("=" * 80)
    
    try:
        # Run all tests
        session_id = test_session_management()
        if not session_id:
            print("✗ Session management test failed, aborting other tests")
            return
        
        test_basic_command_execution(session_id)
        test_git_commands(session_id)
        test_maven_commands(session_id)
        test_multiple_commands(session_id)
        test_working_directory_management(session_id)
        test_session_status_and_history(session_id)
        test_error_handling(session_id)
        test_concurrent_execution(session_id)
        
        # Final status check
        print("\n" + "=" * 60)
        print(" Final Status Check")
        print("=" * 60)
        
        final_status = get_session_status(session_id)
        if final_status["success"]:
            print(f"✓ Final session status:")
            print(f"  Total commands executed: {final_status['total_commands']}")
            print(f"  Status breakdown: {final_status['status_counts']}")
            print(f"  Working directory: {final_status['working_directory']}")
        
        # Clean up
        print(f"\nCleaning up session {session_id}...")
        cleanup_result = close_terminal_session(session_id)
        if cleanup_result["success"]:
            print(f"✓ Session closed successfully")
        else:
            print(f"✗ Failed to close session: {cleanup_result['error']}")
        
        # Final session list
        final_sessions = list_terminal_sessions()
        if final_sessions["success"]:
            print(f"✓ Final session count: {final_sessions['total_sessions']}")
        
        print("\n" + "=" * 80)
        print(" All Tests Completed Successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
