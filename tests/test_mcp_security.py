# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP security module.

These tests verify that the MCP command validation properly prevents
command injection attacks and other security vulnerabilities.
"""

import re

import pytest

from vllm_mlx.mcp.security import (
    ALLOWED_COMMANDS,
    MCPCommandValidator,
    MCPSecurityError,
    ToolExecutionAudit,
    ToolSandbox,
)
from vllm_mlx.mcp.types import MCPServerConfig, MCPTransport


class TestMCPCommandValidator:
    """Tests for MCPCommandValidator class."""

    def test_allowed_command_passes(self):
        """Test that allowed commands pass validation."""
        # Use check_path_exists=False for unit tests (we're testing whitelist logic)
        validator = MCPCommandValidator(check_path_exists=False)

        # These should not raise
        validator.validate_command("npx", "test-server")
        validator.validate_command("uvx", "test-server")
        validator.validate_command("python", "test-server")
        validator.validate_command("node", "test-server")

    def test_disallowed_command_fails(self):
        """Test that disallowed commands are rejected."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("bash", "test-server")

        assert "not in the allowed commands whitelist" in str(exc_info.value)

    def test_command_injection_semicolon_blocked(self):
        """Test that command injection via semicolon is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx; rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_pipe_blocked(self):
        """Test that command injection via pipe is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx | cat /etc/passwd", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_and_blocked(self):
        """Test that command injection via && is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx && rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_backtick_blocked(self):
        """Test that command injection via backticks is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx `whoami`", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_command_injection_dollar_paren_blocked(self):
        """Test that command injection via $() is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("npx $(whoami)", "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_path_traversal_blocked(self):
        """Test that path traversal is blocked."""
        validator = MCPCommandValidator(check_path_exists=False)

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_command("../../../bin/bash", "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestArgumentValidation:
    """Tests for argument validation."""

    def test_safe_args_pass(self):
        """Test that safe arguments pass validation."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_args(
            ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "test"
        )
        validator.validate_args(["--db-path", "data.db"], "test")
        validator.validate_args(["--port", "8080"], "test")

    def test_injection_in_args_blocked(self):
        """Test that command injection in arguments is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["-y", "; rm -rf /"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_backtick_in_args_blocked(self):
        """Test that backticks in arguments are blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["--path", "`cat /etc/passwd`"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)

    def test_dollar_expansion_in_args_blocked(self):
        """Test that $() in arguments is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_args(["--cmd", "$(whoami)"], "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestEnvironmentValidation:
    """Tests for environment variable validation."""

    def test_safe_env_passes(self):
        """Test that safe environment variables pass."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_env({"API_KEY": "secret123", "DEBUG": "true"}, "test")

    def test_ld_preload_blocked(self):
        """Test that LD_PRELOAD is blocked (library injection)."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"LD_PRELOAD": "/tmp/malicious.so"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_path_modification_blocked(self):
        """Test that PATH modification is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"PATH": "/tmp/fake:/usr/bin"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_pythonpath_blocked(self):
        """Test that PYTHONPATH is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"PYTHONPATH": "/tmp/malicious"}, "test-server")

        assert "not allowed for security reasons" in str(exc_info.value)

    def test_injection_in_env_value_blocked(self):
        """Test that injection in env values is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_env({"SAFE_VAR": "value; rm -rf /"}, "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestURLValidation:
    """Tests for SSE URL validation."""

    def test_https_url_passes(self):
        """Test that HTTPS URLs pass validation."""
        validator = MCPCommandValidator()

        # These should not raise
        validator.validate_url("https://example.com/sse", "test")
        validator.validate_url("https://api.service.com:8443/mcp", "test")

    def test_localhost_http_passes(self):
        """Test that localhost HTTP URLs pass (for development)."""
        validator = MCPCommandValidator()

        # These should not raise (but may warn)
        validator.validate_url("http://localhost:3000/sse", "test")
        validator.validate_url("http://localhost/mcp", "test")

    def test_non_http_scheme_blocked(self):
        """Test that non-HTTP schemes are blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("file:///etc/passwd", "test-server")

        assert "must use http:// or https://" in str(exc_info.value)

    def test_ftp_scheme_blocked(self):
        """Test that FTP scheme is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("ftp://malicious.com/file", "test-server")

        assert "must use http:// or https://" in str(exc_info.value)

    def test_injection_in_url_blocked(self):
        """Test that injection in URL is blocked."""
        validator = MCPCommandValidator()

        with pytest.raises(MCPSecurityError) as exc_info:
            validator.validate_url("https://example.com/sse; rm -rf /", "test-server")

        assert "dangerous pattern" in str(exc_info.value)


class TestUnsafeMode:
    """Tests for unsafe mode (development only)."""

    def test_unsafe_mode_allows_any_command(self):
        """Test that unsafe mode allows any command (with warning)."""
        validator = MCPCommandValidator(allow_unsafe=True)

        # These should not raise even though they're dangerous
        validator.validate_command("bash", "test")
        validator.validate_command("/bin/sh -c 'dangerous'", "test")

    def test_unsafe_mode_allows_any_args(self):
        """Test that unsafe mode allows any arguments."""
        validator = MCPCommandValidator(allow_unsafe=True)

        # These should not raise
        validator.validate_args(["; rm -rf /"], "test")


class TestCustomWhitelist:
    """Tests for custom command whitelist."""

    def test_custom_whitelist_extends_default(self):
        """Test that custom whitelist extends the default."""
        validator = MCPCommandValidator(
            custom_whitelist={"my-custom-mcp-server"},
            check_path_exists=False,
        )

        # Default commands should still work
        validator.validate_command("npx", "test")

        # Custom command should now work
        validator.validate_command("my-custom-mcp-server", "test")

    def test_custom_whitelist_only(self):
        """Test using only custom whitelist."""
        validator = MCPCommandValidator(
            allowed_commands={"custom-only"},
            check_path_exists=False,
        )

        # Custom command should work
        validator.validate_command("custom-only", "test")

        # Default commands should now fail
        with pytest.raises(MCPSecurityError):
            validator.validate_command("npx", "test")


class TestMCPServerConfigSecurity:
    """Tests for security validation in MCPServerConfig."""

    def test_valid_stdio_config(self):
        """Test that valid stdio config passes security validation."""
        # This should not raise
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert config.command == "npx"

    def test_invalid_command_rejected(self):
        """Test that invalid commands are rejected."""
        with pytest.raises(ValueError) as exc_info:
            MCPServerConfig(
                name="malicious-server",
                transport=MCPTransport.STDIO,
                command="bash",
                args=["-c", "rm -rf /"],
            )

        assert "not in the allowed commands whitelist" in str(exc_info.value)

    def test_command_injection_in_config_rejected(self):
        """Test that command injection in config is rejected."""
        with pytest.raises(ValueError) as exc_info:
            MCPServerConfig(
                name="injection-server",
                transport=MCPTransport.STDIO,
                command="npx; rm -rf /",
            )

        assert "dangerous pattern" in str(exc_info.value)

    def test_valid_sse_config(self):
        """Test that valid SSE config passes validation."""
        config = MCPServerConfig(
            name="sse-server",
            transport=MCPTransport.SSE,
            url="https://api.example.com/mcp",
        )
        assert config.url == "https://api.example.com/mcp"

    def test_skip_security_validation(self):
        """Test that skip_security_validation allows any command (with warning)."""
        # This should not raise even with dangerous command
        config = MCPServerConfig(
            name="unsafe-server",
            transport=MCPTransport.STDIO,
            command="bash",
            args=["-c", "echo hello"],
            skip_security_validation=True,
        )
        assert config.command == "bash"


class TestDefaultWhitelist:
    """Tests for the default command whitelist."""

    def test_default_whitelist_contains_expected_commands(self):
        """Test that the default whitelist contains expected safe commands."""
        assert "npx" in ALLOWED_COMMANDS
        assert "uvx" in ALLOWED_COMMANDS
        assert "python" in ALLOWED_COMMANDS
        assert "python3" in ALLOWED_COMMANDS
        assert "node" in ALLOWED_COMMANDS
        assert "docker" in ALLOWED_COMMANDS

    def test_default_whitelist_excludes_dangerous_commands(self):
        """Test that dangerous commands are not in default whitelist."""
        assert "bash" not in ALLOWED_COMMANDS
        assert "sh" not in ALLOWED_COMMANDS
        assert "zsh" not in ALLOWED_COMMANDS
        assert "rm" not in ALLOWED_COMMANDS
        assert "curl" not in ALLOWED_COMMANDS
        assert "wget" not in ALLOWED_COMMANDS


# ============================================================================
# Tool Sandbox Tests
# ============================================================================


class TestToolSandbox:
    """Tests for ToolSandbox class."""

    def test_sandbox_allows_safe_tool_execution(self):
        """Test that safe tool execution is allowed."""
        sandbox = ToolSandbox()

        # Should not raise
        sandbox.validate_tool_execution(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
        )

    def test_sandbox_blocks_blocklisted_tool(self):
        """Test that blocklisted tools are blocked."""
        sandbox = ToolSandbox(blocked_tools={"dangerous_tool", "shell"})

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="dangerous_tool",
                server_name="test",
                arguments={},
            )

        assert "blocked by security policy" in str(exc_info.value)

    def test_sandbox_allowlist_mode(self):
        """Test that allowlist mode only permits specific tools."""
        sandbox = ToolSandbox(allowed_tools={"safe_tool", "another_safe"})

        # Allowed tool should pass
        sandbox.validate_tool_execution(
            tool_name="safe_tool",
            server_name="test",
            arguments={},
        )

        # Non-allowed tool should fail
        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="unknown_tool",
                server_name="test",
                arguments={},
            )

        assert "not in the allowed tools list" in str(exc_info.value)

    def test_sandbox_blocks_path_traversal_in_args(self):
        """Test that path traversal in arguments is blocked."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": "../../etc/passwd"},
            )

        assert "blocked pattern" in str(exc_info.value)

    def test_sandbox_blocks_etc_access(self):
        """Test that /etc/ access is blocked."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": "/etc/shadow"},
            )

        assert "blocked pattern" in str(exc_info.value)

    def test_sandbox_blocks_proc_access(self):
        """Test that /proc/ access is blocked."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": "/proc/self/environ"},
            )

        assert "blocked pattern" in str(exc_info.value)

    def test_sandbox_validates_nested_arguments(self):
        """Test that nested arguments are validated."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="complex_tool",
                server_name="test",
                arguments={
                    "config": {
                        "nested": {
                            "path": "/etc/passwd",
                        }
                    }
                },
            )

        assert "blocked pattern" in str(exc_info.value)

    def test_sandbox_validates_list_arguments(self):
        """Test that list arguments are validated."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="multi_file",
                server_name="filesystem",
                arguments={"files": ["/tmp/safe.txt", "../../../etc/passwd"]},
            )

        assert "blocked pattern" in str(exc_info.value)

    def test_sandbox_disabled_allows_all(self):
        """Test that disabled sandbox allows all executions."""
        sandbox = ToolSandbox(enabled=False)

        # Should not raise even with blocked patterns
        sandbox.validate_tool_execution(
            tool_name="shell",
            server_name="dangerous",
            arguments={"path": "/etc/passwd"},
        )


class TestToolSandboxRateLimiting:
    """Tests for sandbox rate limiting."""

    def test_rate_limit_allows_within_limit(self):
        """Test that calls within rate limit are allowed."""
        sandbox = ToolSandbox(max_calls_per_minute=10)

        # Should allow up to 10 calls
        for i in range(10):
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": f"/tmp/file{i}.txt"},
            )

    def test_rate_limit_blocks_over_limit(self):
        """Test that calls over rate limit are blocked."""
        sandbox = ToolSandbox(max_calls_per_minute=5)

        # Make 5 calls
        for i in range(5):
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": f"/tmp/file{i}.txt"},
            )

        # 6th call should be blocked
        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": "/tmp/file6.txt"},
            )

        assert "Rate limit exceeded" in str(exc_info.value)

    def test_rate_limit_disabled_with_zero(self):
        """Test that rate limiting is disabled with max_calls_per_minute=0."""
        sandbox = ToolSandbox(max_calls_per_minute=0)

        # Should allow unlimited calls
        for i in range(100):
            sandbox.validate_tool_execution(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": f"/tmp/file{i}.txt"},
            )


class TestToolSandboxAuditLogging:
    """Tests for sandbox audit logging."""

    def test_record_successful_execution(self):
        """Test recording a successful tool execution."""
        sandbox = ToolSandbox()

        audit = sandbox.record_execution(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
            success=True,
            execution_time_ms=50.5,
        )

        assert audit.tool_name == "read_file"
        assert audit.server_name == "filesystem"
        assert audit.success is True
        assert audit.execution_time_ms == 50.5
        assert audit.error_message is None

    def test_record_failed_execution(self):
        """Test recording a failed tool execution."""
        sandbox = ToolSandbox()

        audit = sandbox.record_execution(
            tool_name="write_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt", "content": "data"},
            success=False,
            error_message="Permission denied",
        )

        assert audit.tool_name == "write_file"
        assert audit.success is False
        assert audit.error_message == "Permission denied"

    def test_audit_log_retrieval(self):
        """Test retrieving audit log entries."""
        sandbox = ToolSandbox()

        # Record multiple executions
        sandbox.record_execution("tool1", "server1", {}, True)
        sandbox.record_execution("tool2", "server2", {}, False, "error")
        sandbox.record_execution("tool1", "server1", {}, True)

        # Get all entries
        entries = sandbox.get_audit_log()
        assert len(entries) == 3

        # Filter by tool
        tool1_entries = sandbox.get_audit_log(tool_filter="tool1")
        assert len(tool1_entries) == 2

        # Filter by server
        server2_entries = sandbox.get_audit_log(server_filter="server2")
        assert len(server2_entries) == 1

        # Filter errors only
        error_entries = sandbox.get_audit_log(errors_only=True)
        assert len(error_entries) == 1
        assert error_entries[0].tool_name == "tool2"

    def test_audit_log_limit(self):
        """Test audit log limit parameter."""
        sandbox = ToolSandbox()

        # Record many executions
        for i in range(10):
            sandbox.record_execution(f"tool{i}", "server", {}, True)

        # Get limited entries
        entries = sandbox.get_audit_log(limit=5)
        assert len(entries) == 5
        # Should return most recent
        assert entries[-1].tool_name == "tool9"

    def test_audit_log_sensitive_data_redaction(self):
        """Test that sensitive data is redacted in audit log."""
        sandbox = ToolSandbox()

        audit = sandbox.record_execution(
            tool_name="api_call",
            server_name="test",
            arguments={
                "url": "https://api.example.com",
                "api_key": "secret-key-12345",
                "password": "my-password",
                "data": {"token": "bearer-token"},
            },
            success=True,
        )

        # Sensitive fields should be redacted
        assert audit.arguments["api_key"] == "[REDACTED]"
        assert audit.arguments["password"] == "[REDACTED]"
        assert audit.arguments["data"]["token"] == "[REDACTED]"
        # Non-sensitive fields should remain
        assert audit.arguments["url"] == "https://api.example.com"

    def test_audit_log_truncates_long_values(self):
        """Test that long argument values are truncated."""
        sandbox = ToolSandbox()

        long_content = "x" * 2000

        audit = sandbox.record_execution(
            tool_name="write_file",
            server_name="filesystem",
            arguments={"content": long_content},
            success=True,
        )

        # Should be truncated
        assert len(audit.arguments["content"]) < len(long_content)
        assert "truncated" in audit.arguments["content"]

    def test_audit_callback(self):
        """Test that audit callback is called."""
        callback_records = []

        def callback(audit: ToolExecutionAudit):
            callback_records.append(audit)

        sandbox = ToolSandbox(audit_callback=callback)

        sandbox.record_execution("tool1", "server1", {}, True)
        sandbox.record_execution("tool2", "server2", {}, False, "error")

        assert len(callback_records) == 2
        assert callback_records[0].tool_name == "tool1"
        assert callback_records[1].tool_name == "tool2"

    def test_clear_audit_log(self):
        """Test clearing the audit log."""
        sandbox = ToolSandbox()

        sandbox.record_execution("tool1", "server1", {}, True)
        sandbox.record_execution("tool2", "server2", {}, True)

        count = sandbox.clear_audit_log()
        assert count == 2

        entries = sandbox.get_audit_log()
        assert len(entries) == 0


class TestToolSandboxHighRiskTools:
    """Tests for high-risk tool detection."""

    def test_high_risk_tool_blocked_by_default(self):
        """High-risk tools should be blocked unless explicitly allowlisted."""
        sandbox = ToolSandbox()

        with pytest.raises(MCPSecurityError) as exc_info:
            sandbox.validate_tool_execution(
                tool_name="execute_command",
                server_name="test",
                arguments={"cmd": "ls"},
            )

        assert "High-risk tool 'execute_command' is blocked" in str(exc_info.value)
        assert "allowed_high_risk_tools" in str(exc_info.value)

    def test_high_risk_tool_allowed_by_short_name(self, caplog):
        """Short-name allowlist entries should permit trusted high-risk tools."""
        import logging

        sandbox = ToolSandbox(allowed_high_risk_tools={"run_shell"})

        with caplog.at_level(logging.WARNING):
            sandbox.validate_tool_execution(
                tool_name="run_shell",
                server_name="test",
                arguments={},
            )

        assert "Allowing high-risk tool 'test__run_shell'" in caplog.text

    def test_high_risk_tool_allowed_by_full_name(self, caplog):
        """Full-name allowlist entries should permit trusted high-risk tools."""
        import logging

        sandbox = ToolSandbox(allowed_high_risk_tools={"trusted__execute_command"})

        with caplog.at_level(logging.WARNING):
            sandbox.validate_tool_execution(
                tool_name="execute_command",
                server_name="trusted",
                arguments={"cmd": "ls"},
            )

        assert "Allowing high-risk tool 'trusted__execute_command'" in caplog.text


class TestCustomBlockedPatterns:
    """Tests for custom blocked argument patterns."""

    def test_custom_blocked_pattern(self):
        """Test adding custom blocked patterns."""
        custom_patterns = [
            re.compile(r"\.\."),  # Block any ..
            re.compile(r"~"),  # Block home dir
        ]

        sandbox = ToolSandbox(blocked_arg_patterns=custom_patterns)

        # Should block ..
        with pytest.raises(MCPSecurityError):
            sandbox.validate_tool_execution(
                tool_name="tool",
                server_name="server",
                arguments={"path": "a..b"},
            )

        # Should block ~
        with pytest.raises(MCPSecurityError):
            sandbox.validate_tool_execution(
                tool_name="tool",
                server_name="server",
                arguments={"path": "~/secret"},
            )
