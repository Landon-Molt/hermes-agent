"""Regression tests for sudo detection and sudo password handling."""

import base64
import struct

import tools.terminal_tool as terminal_tool


def _valid_macho64() -> bytes:
    header_size = 32
    command_size = 72
    file_size = header_size + command_size
    header = b"\xcf\xfa\xed\xfe" + struct.pack(
        "<iiIIIII", 0x0100000C, 0, 2, 1, command_size, 0, 0
    )
    segment = struct.pack(
        "<II16sQQQQIIII",
        0x19,
        command_size,
        b"__TEXT",
        0,
        file_size,
        0,
        file_size,
        7,
        5,
        0,
        0,
    )
    return header + segment


def _wire(data: bytes) -> str:
    probe = base64.b64encode(data[: 64 * 1024]).decode("ascii")
    return (
        f"__HERMES_GATEWAY_GUARD_V1__:{len(data)}:{probe}\n"
        + data.decode("latin-1")
    )


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "once per session" in description


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_validate_workdir_allows_unicode_filesystem_paths():
    assert terminal_tool._validate_workdir(
        "/Users/alice/Documents/Obs_Hermes_Data/项目-projects/客户拜访"
    ) is None
    assert terminal_tool._validate_workdir("/tmp/テスト") is None
    assert terminal_tool._validate_workdir("/home/jürgen/über projekt") is None


def test_validate_workdir_still_blocks_metachars_in_unicode_paths():
    # Widening to Unicode letters must not open the injection boundary:
    # shell metacharacters and control chars stay rejected even when mixed
    # with non-ASCII path segments.
    assert terminal_tool._validate_workdir("/tmp/テスト; rm -rf /")
    assert terminal_tool._validate_workdir("/tmp/项目$(whoami)")
    assert terminal_tool._validate_workdir("/tmp/über`id`")
    assert terminal_tool._validate_workdir("/tmp/テスト\nwhoami")
    assert terminal_tool._validate_workdir("/tmp/项目|cat /etc/passwd")
    assert terminal_tool._validate_workdir("/tmp/ü\x00ber")


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2


class _GuardReadEnv:
    def __init__(self, output=""):
        self.output = output
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return {"returncode": 0, "output": self.output}


def _local_guard_env(tmp_path, monkeypatch):
    env = terminal_tool._LocalEnvironment(cwd=str(tmp_path), timeout=60)
    commands = []

    def _unexpected_execute(command, **_kwargs):
        commands.append(command)
        raise AssertionError("local guard read must not call env.execute")

    monkeypatch.setattr(env, "execute", _unexpected_execute)
    return env, commands


def test_gateway_guard_local_macho_never_falls_back_to_env_execute(
    tmp_path, monkeypatch
):
    executable = tmp_path / "python"
    executable.write_bytes(_valid_macho64())
    env, commands = _local_guard_env(tmp_path, monkeypatch)

    result = terminal_tool._read_script_for_gateway_guard(
        env=env,
        env_type="local",
        script_path=str(executable),
        guard_cwd=str(tmp_path),
        max_bytes=1024,
    )

    assert result == ""
    assert commands == []


def test_gateway_guard_local_nul_shell_content_is_preserved(tmp_path, monkeypatch):
    script = tmp_path / "wrapper.sh"
    script.write_bytes(b"#!/bin/bash\nhermes gateway restart\x00tail\n")
    env, commands = _local_guard_env(tmp_path, monkeypatch)

    result = terminal_tool._read_script_for_gateway_guard(
        env=env,
        env_type="local",
        script_path=str(script),
        guard_cwd=str(tmp_path),
        max_bytes=1024,
    )

    assert result == "#!/bin/bash\nhermes gateway restart\x00tail\n"
    assert commands == []


def test_gateway_guard_remote_nul_content_is_preserved(tmp_path):
    content = b"#!/bin/bash\nhermes gateway stop\x00tail\n"
    output = _wire(content)
    env = _GuardReadEnv(output)

    result = terminal_tool._read_script_for_gateway_guard(
        env=env,
        env_type="ssh",
        script_path="/remote/wrapper.sh",
        guard_cwd=str(tmp_path),
        max_bytes=1024,
    )

    assert result == output
    assert len(env.commands) == 1
    command = env.commands[0]
    assert "wc -c" in command
    assert "base64" in command
    assert "head -c 65536" in command


def test_gateway_guard_remote_malformed_output_fails_closed(tmp_path):
    env = _GuardReadEnv("truncated backend output")

    result = terminal_tool._read_script_for_gateway_guard(
        env=env,
        env_type="ssh",
        script_path="/remote/wrapper.sh",
        guard_cwd=str(tmp_path),
        max_bytes=1024,
    )

    assert result == "__HERMES_GATEWAY_GUARD_UNSAFE__"


def test_gateway_guard_local_missing_path_never_executes_remote_fallback(
    tmp_path, monkeypatch
):
    env, commands = _local_guard_env(tmp_path, monkeypatch)

    result = terminal_tool._read_script_for_gateway_guard(
        env=env,
        env_type="local",
        script_path=str(tmp_path / "missing.sh"),
        guard_cwd=str(tmp_path),
        max_bytes=1024,
    )

    assert result is None
    assert commands == []
