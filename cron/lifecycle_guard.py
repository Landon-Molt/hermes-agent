"""Gateway lifecycle guard for cron job creation (#30719).

An agent running inside a gateway can schedule a cron job that calls
``hermes gateway restart`` (or ``launchctl kickstart ai.hermes.gateway``
or ``systemctl restart hermes-gateway``).  When the cron fires, the
gateway dies, the supervisor (launchd KeepAlive / systemd Restart=)
revives it, auto-resume picks up the offending session, and the resumed
turn re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command.  It is enforced at
``cron.jobs.create_job`` so it fires on every job-creation path: the
``hermes cron create`` CLI subcommand AND the agent's ``cronjob`` model
tool (which calls ``create_job`` directly, bypassing the CLI layer).

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``hermes gateway``, ``launchctl ... hermes-gateway``,
``systemctl ... hermes-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.  A cron ``prompt`` is fed to a future LLM, not a
shell, so an over-broad substring match on English ("Kong API gateway
autoscaling and restart behavior") would produce a high false-positive
rate without preventing the actual foot-gun, which requires a real
command shape.

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` blocks direct
commands and shell scripts they reference when ``_HERMES_GATEWAY=1``. It also
rejects ``launchctl submit`` in gateway sessions because launchd treats that
primitive as a persistent KeepAlive job, not a one-shot task. ``hermes gateway
stop|restart`` separately refuse to self-target from inside the gateway.
Blocking cron specs at creation time as well means the agent gets an immediate,
informative rejection instead of scheduling a job that will only fail
(silently) when it fires.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import shlex
import stat
import struct
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each branch
# is anchored on a concrete command identifier so a match can only fire on
# actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `hermes gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside a
    # gateway is benign (a no-op or "already running" error), and a
    # legitimate cron job might start a sibling profile's gateway.
    r"(?:hermes\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a hermes-gateway label. macOS launchd
    # labels look like `ai.hermes.gateway` / `hermes-gateway`. Requiring the
    # gateway identifier prevents blocking unrelated hermes services (e.g.
    # `launchctl unload ai.hermes.update-checker.plist`).
    # `submit` and `bootstrap` are included alongside the direct verbs
    # (kickstart/etc.): `launchctl submit -l ai.hermes.gateway-<suffix> --
    # <helper-script>` (or `launchctl bootstrap gui/<uid> <plist>`) creates
    # a NEW keepalive job wrapping an arbitrary helper, which is how a
    # blocked direct restart/kill gets laundered into a persistent restart
    # loop instead (#62891) — same foot-gun, indirect shape. Neutral-label
    # submissions that dodge this text anchor are caught separately by
    # `contains_launchctl_submit_command` (execution-aware, label-independent).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    r"|(?:p?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


# A backslash immediately followed by a newline is a POSIX shell line
# continuation — the shell joins the two lines before parsing. Every branch
# above uses `[^\n]*` between its verb and the gateway identifier so the
# match can't span unrelated lines of a longer cron prompt/script, but that
# also means a real multi-line shell invocation split across continuation
# lines (e.g. `launchctl submit \` / `  -l ai.hermes.gateway-... \` / `  -- ...`,
# the exact reported shape in #62891) would otherwise slip past. Collapse
# continuations to a single space before matching, mirroring what the shell
# itself does, rather than loosening `[^\n]*` and risking false positives
# across genuinely separate lines.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(normalized))


_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")


# Directory names that sit directly under a `Library` path component and
# mark a FileProvider-backed subtree: `Mobile Documents` is iCloud Drive;
# `CloudStorage` hosts every third-party FileProvider domain (Dropbox,
# OneDrive, Google Drive, Box, ...) on modern macOS.
_CLOUD_PLACEHOLDER_MARKERS = frozenset({"Mobile Documents", "CloudStorage"})


def _is_cloud_placeholder_path(path: Path) -> bool:
    """Return True for paths inside a macOS FileProvider-backed subtree.

    ``O_NONBLOCK`` does not make regular-file reads non-blocking.  Opening an
    evicted FileProvider placeholder below ``~/Library/Mobile Documents``
    (iCloud Drive) or ``~/Library/CloudStorage`` (Dropbox / OneDrive /
    Google Drive and other third-party providers) can therefore wait
    indefinitely for hydration.  The lifecycle guard runs before a terminal
    command's timeout starts, so it must identify this boundary from path
    metadata and fail closed without opening the file.
    """
    parts = path.parts
    return any(
        parts[index - 1] == "Library" and part in _CLOUD_PLACEHOLDER_MARKERS
        for index, part in enumerate(parts)
        if index
    )

# Executables whose arguments are DATA, not commands: search patterns, SQL
# statements, log filters. None of these can execute their argument text, so
# a lifecycle-shaped string inside their arguments (a grep pattern hunting
# for `systemctl restart hermes-gateway` in syslog, a SQL LIKE literal over a
# restart-events table) is diagnostics, not a lifecycle command. Deliberately
# conservative: no `awk` (system()), no `sed` (`s///e`), no `echo`/`printf`
# (routinely piped into a shell), no `mysql` (`\\!` and `system` escapes).
_DATA_SINK_EXECUTABLES = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ag", "ack", "journalctl", "sqlite3", "psql"}
)
# Argument shapes that can smuggle execution back INTO a data sink: command
# and process substitution anywhere, sqlite3 dot-commands (`.shell ...`),
# psql backslash escapes (`\! ...`). Any hit disables masking for the whole
# segment — fail closed to the plain regex verdict.
_UNSAFE_DATA_ARG_MARKERS = ("`", "$(", "<(", ">(", "\\!")
# A data sink piped into a shell/interpreter can feed matched lines straight
# to execution (`grep 'systemctl restart hermes-gateway' f | sh`); never mask
# such a line.
_PIPE_TO_INTERPRETER = re.compile(
    r"\|\s*&?\s*(?:sudo\s+)?(?:sh|bash|dash|ksh|zsh|xargs|eval|source)\b"
)

# Executable-image magic numbers: ELF, PE/COFF, Mach-O (universal + thin,
# both endiannesses). Magic is only a classifier; the corresponding format
# must also pass bounded structural validation before the guard skips it.
_BINARY_MAGIC_PREFIXES = (
    b"\x7fELF",
    b"MZ",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
)
_EXECUTABLE_PROBE_BYTES = 64 * 1024
_REMOTE_SCRIPT_WIRE_PREFIX = "__HERMES_GATEWAY_GUARD_V1__:"
_REMOTE_SCRIPT_UNSAFE_SENTINEL = "__HERMES_GATEWAY_GUARD_UNSAFE__"




_ReadRemoteScriptFn = Callable[[str], Optional[str]]


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments, honoring quotes and comments."""
    normalized = command.replace("\\\n", "")
    for line in normalized.splitlines() or [normalized]:
        try:
            lexer = shlex.shlex(
                line,
                posix=True,
                punctuation_chars=";&|()",
            )
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            continue

        segment: list[str] = []
        for token in tokens:
            if token and set(token) <= _CONTROL_CHARS:
                if segment:
                    yield segment
                    segment = []
                continue
            segment.append(token)
        if segment:
            yield segment


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    for index, token in enumerate(segment):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        return index
    return None


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: the label of a submitted/bootstrapped job is
    chosen by whoever writes it, so a neutral name (``ai.hermes.svc-reload-tmp``)
    defeats any label-anchored regex (#62891, second reproduction). Both verbs
    register a NEW persistent launchd job (``submit`` jobs get KeepAlive
    semantics; ``bootstrap`` loads an arbitrary plist), which is never safe to
    do from inside the gateway process.
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        if Path(segment[index]).name == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _mask_data_sink_arguments(text: str) -> str:
    """Replace data-sink executables' arguments with a neutral placeholder.

    The lifecycle regex is command-shaped, but it cannot tell an EXECUTED
    ``systemctl restart hermes-gateway`` from the same characters appearing
    as *data* — a grep/rg pattern, a journalctl filter, a SQL string literal
    passed to sqlite3/psql. Those diagnostics commands were being rejected
    (false positives blocking legitimate cron prompts), e.g.::

        grep -c 'systemctl restart hermes-gateway' /var/log/syslog
        sqlite3 db "SELECT msg FROM log WHERE msg LIKE '%systemctl restart hermes-gateway%'"

    This masker shell-tokenizes each line and, for command segments whose
    executable is a known data sink (``_DATA_SINK_EXECUTABLES``), replaces
    every argument with ``arg``. The caller then re-runs the lifecycle regex
    on the masked text: a match that survives masking sits OUTSIDE any data
    argument and is a real command.

    Strictly fail-closed: masking is skipped (leaving the original,
    regex-matching text in place) whenever the line pipes into a shell or
    interpreter, any argument carries an execution-capable marker
    (substitution, sqlite3 ``.``-commands, psql ``\\!``), or the line cannot
    be tokenized at all. Masking can therefore only ever ALLOW a command the
    plain regex would have blocked — never block one it would have allowed —
    so it runs solely as a second-pass exemption check.
    """
    lines_out: list[str] = []
    changed = False
    for line in text.splitlines() or [text]:
        if _PIPE_TO_INTERPRETER.search(line):
            lines_out.append(line)
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            lines_out.append(line)
            continue

        segments: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if token and set(token) <= _CONTROL_CHARS:
                segments.append(current)
                segments.append([token])
                current = []
                continue
            current.append(token)
        segments.append(current)

        rebuilt: list[str] = []
        for segment in segments:
            if not segment:
                continue
            index = _command_token_index(segment)
            if index is not None and Path(segment[index]).name in _DATA_SINK_EXECUTABLES:
                arguments = segment[index + 1 :]
                if not any(
                    argument.startswith(".")
                    or any(marker in argument for marker in _UNSAFE_DATA_ARG_MARKERS)
                    for argument in arguments
                ):
                    changed = True
                    rebuilt.extend(segment[: index + 1])
                    rebuilt.extend("arg" for _ in arguments)
                    continue
            rebuilt.extend(segment)
        lines_out.append(" ".join(rebuilt))
    if not changed:
        return text
    return "\n".join(lines_out)


def _lifecycle_command_scan_with_data_exemption(text: str) -> bool:
    """Lifecycle-regex scan that exempts matches living inside data arguments.

    Two-pass: the cheap regex first (the overwhelmingly common no-match case
    pays nothing extra); on a raw match, re-scan with data-sink arguments
    masked out. Only a match that survives masking — i.e. one in actual
    command position — blocks.
    """
    if not contains_gateway_lifecycle_command(text):
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return contains_gateway_lifecycle_command(_mask_data_sink_arguments(normalized))


def _direct_lifecycle_scan(command: str) -> bool:
    """Pure-string direct scans: lifecycle regex (data-exempted) + submit."""
    return _lifecycle_command_scan_with_data_exemption(
        command
    ) or contains_launchctl_submit_command(command)


def _expand_candidate_path(candidate: str) -> Optional[Path]:
    """Sanitize a tokenized path candidate at the ingestion boundary.

    Candidate tokens come from shlex-splitting arbitrary command text —
    including text recursively decoded from binaries or remote reads — so
    they can carry NUL bytes or other junk no real filesystem path can
    contain. Every OS-facing ``Path`` operation downstream (``expanduser``,
    ``os.open``, ``resolve``) raises a *different* exception for the same
    junk (``ValueError: embedded null byte``, ``RuntimeError: Could not
    determine home directory`` when HOME is unset under launchd, OSError
    for over-long paths). Rejecting here — once, before any OS call — is
    the whole-class fix; catching per-syscall was the whack-a-mole that
    produced #76762, #77703, #77780, and #78256.

    Returns ``None`` for candidates that cannot be a real path. If ``~user``
    cannot be expanded, preserve the literal word because POSIX shells do too.
    """
    if not candidate or "\x00" in candidate:
        return None
    path = Path(candidate)
    try:
        return path.expanduser()
    except (ValueError, RuntimeError, OSError):
        return path


def _resolve_terminal_script_path(candidate: str, cwd: Optional[str]) -> Optional[Path]:
    path = _expand_candidate_path(candidate)
    if path is None:
        return None
    if not path.is_absolute():
        try:
            path = Path(cwd or Path.cwd()) / path
        except OSError:
            # Path.cwd() can raise when the process cwd was deleted.
            return None
    return path


def _iter_referenced_shell_scripts(
    command: str,
    *,
    cwd: Optional[str] = None,
) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        executable = segment[index]
        executable_name = Path(executable).name

        if executable_name in {".", "source"}:
            if len(segment) > index + 1:
                resolved = _resolve_terminal_script_path(segment[index + 1], cwd)
                if resolved is not None:
                    yield resolved
            continue

        if executable_name in _SHELL_EXECUTABLES:
            arguments = segment[index + 1 :]
            arg_index = 0
            while arg_index < len(arguments):
                argument = arguments[arg_index]
                if argument == "--":
                    arg_index += 1
                    break
                if argument in {"-c", "--command"}:
                    break
                if argument in _SHELL_OPTIONS_WITH_VALUES:
                    arg_index += 2
                    continue
                if argument.startswith("-"):
                    arg_index += 1
                    continue
                break
            if arg_index < len(arguments) and arguments[arg_index] not in {
                "-c",
                "--command",
            }:
                resolved = _resolve_terminal_script_path(arguments[arg_index], cwd)
                if resolved is not None:
                    yield resolved
            continue

        # A bare "/" token is pathlib's division operator in Python sources
        # (e.g. `Path.home() / ".hermes"`), not an executable reference.
        # Resolving it walks to the filesystem root and fails the
        # regular-file check below, hard-blocking innocent .py scripts
        # (#77131). Skip pure-separator tokens.
        if executable.strip("/"):
            if "/" in executable or executable.endswith((".sh", ".bash", ".zsh")):
                resolved = _resolve_terminal_script_path(executable, cwd)
                if resolved is not None:
                    yield resolved


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` for recursive scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None or Path(segment[index]).name not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in {"-c", "--command"}:
                yield arguments[arg_index + 1]
                break


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path is not None and path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


def _region_fits(offset: int, length: int, file_size: int) -> bool:
    return offset >= 0 and length >= 0 and offset <= file_size - length


def _valid_elf_image(data: bytes, file_size: int) -> bool:
    if len(data) < 16 or data[:4] != b"\x7fELF":
        return False
    elf_class, encoding, version = data[4], data[5], data[6]
    if elf_class not in {1, 2} or encoding not in {1, 2} or version != 1:
        return False
    endian = "<" if encoding == 1 else ">"
    fmt = endian + ("HHIIIIIHHHHHH" if elf_class == 1 else "HHIQQQIHHHHHH")
    header_size = 52 if elf_class == 1 else 64
    program_header_size = 32 if elf_class == 1 else 56
    try:
        fields = struct.unpack_from(fmt, data, 16)
    except struct.error:
        return False
    (
        image_type,
        machine,
        image_version,
        _entry,
        program_offset,
        _section_offset,
        _flags,
        encoded_header_size,
        encoded_program_size,
        program_count,
        *_rest,
    ) = fields
    if (
        image_type not in {2, 3}
        or machine == 0
        or image_version != 1
        or encoded_header_size != header_size
        or encoded_program_size < program_header_size
        or program_count < 1
        or not _region_fits(
            program_offset, encoded_program_size * program_count, file_size
        )
        or program_offset + encoded_program_size * program_count > len(data)
    ):
        return False
    has_load_segment = False
    for index in range(program_count):
        offset = program_offset + index * encoded_program_size
        try:
            if elf_class == 1:
                values = struct.unpack_from(endian + "IIIIIIII", data, offset)
                segment_type, file_offset, file_bytes, memory_bytes = (
                    values[0], values[1], values[4], values[5]
                )
            else:
                values = struct.unpack_from(endian + "IIQQQQQQ", data, offset)
                segment_type, file_offset, file_bytes, memory_bytes = (
                    values[0], values[2], values[5], values[6]
                )
        except struct.error:
            return False
        if segment_type == 1:
            if file_bytes > memory_bytes or not _region_fits(
                file_offset, file_bytes, file_size
            ):
                return False
            has_load_segment = True
    return has_load_segment


def _valid_thin_macho_image(data: bytes, file_size: int) -> bool:
    variants = {
        b"\xce\xfa\xed\xfe": ("<", False),
        b"\xfe\xed\xfa\xce": (">", False),
        b"\xcf\xfa\xed\xfe": ("<", True),
        b"\xfe\xed\xfa\xcf": (">", True),
    }
    variant = variants.get(data[:4])
    if variant is None:
        return False
    endian, is_64_bit = variant
    header_size = 32 if is_64_bit else 28
    fmt = endian + ("iiIIIII" if is_64_bit else "iiIIII")
    try:
        fields = struct.unpack_from(fmt, data, 4)
    except struct.error:
        return False
    if is_64_bit:
        cpu_type, _cpu_subtype, file_type, command_count, command_bytes, *_ = fields
    else:
        cpu_type, _cpu_subtype, file_type, command_count, command_bytes, _flags = fields
    command_end = header_size + command_bytes
    if (
        cpu_type == 0
        or file_type != 2
        or command_count < 1
        or command_count > 65535
        or not _region_fits(header_size, command_bytes, file_size)
        or command_end > len(data)
    ):
        return False
    position = header_size
    has_segment = False
    for _ in range(command_count):
        try:
            command, command_size = struct.unpack_from(endian + "II", data, position)
        except struct.error:
            return False
        if command_size < 8 or position + command_size > command_end:
            return False
        if command == 0x19 and command_size >= 72:
            try:
                file_offset, file_bytes = struct.unpack_from(
                    endian + "QQ", data, position + 40
                )
            except struct.error:
                return False
            if not _region_fits(file_offset, file_bytes, file_size):
                return False
            has_segment = True
        elif command == 0x1 and command_size >= 56:
            try:
                file_offset, file_bytes = struct.unpack_from(
                    endian + "II", data, position + 32
                )
            except struct.error:
                return False
            if not _region_fits(file_offset, file_bytes, file_size):
                return False
            has_segment = True
        position += command_size
    return has_segment and position == command_end


def _valid_fat_macho_image(data: bytes, file_size: int) -> bool:
    variants = {
        b"\xca\xfe\xba\xbe": (">", False),
        b"\xbe\xba\xfe\xca": ("<", False),
        b"\xca\xfe\xba\xbf": (">", True),
        b"\xbf\xba\xfe\xca": ("<", True),
    }
    variant = variants.get(data[:4])
    if variant is None:
        return False
    endian, is_64_bit = variant
    try:
        architecture_count = struct.unpack_from(endian + "I", data, 4)[0]
    except struct.error:
        return False
    entry_size = 32 if is_64_bit else 20
    table_end = 8 + architecture_count * entry_size
    if architecture_count < 1 or architecture_count > 128 or table_end > len(data):
        return False
    for index in range(architecture_count):
        position = 8 + index * entry_size
        try:
            if is_64_bit:
                _cpu, _subtype, offset, size, _align, _reserved = struct.unpack_from(
                    endian + "iiQQII", data, position
                )
            else:
                _cpu, _subtype, offset, size, _align = struct.unpack_from(
                    endian + "iiIII", data, position
                )
        except struct.error:
            return False
        if size < 4 or offset < table_end or not _region_fits(offset, size, file_size):
            return False
        if offset + 4 > len(data):
            return False
        slice_data = data[offset : min(offset + size, len(data))]
        if not _valid_thin_macho_image(slice_data, size):
            return False
    return True


def _valid_pe_image(data: bytes, file_size: int) -> bool:
    if len(data) < 64 or data[:2] != b"MZ":
        return False
    try:
        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    except struct.error:
        return False
    if pe_offset < 64 or pe_offset + 24 > len(data):
        return False
    if data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        return False
    try:
        machine, section_count, _time, _symbols, _symbol_count, optional_size, _flags = (
            struct.unpack_from("<HHIIIHH", data, pe_offset + 4)
        )
    except struct.error:
        return False
    optional_offset = pe_offset + 24
    section_offset = optional_offset + optional_size
    if (
        machine == 0
        or section_count < 1
        or section_count > 96
        or optional_size < 60
        or section_offset + section_count * 40 > len(data)
    ):
        return False
    try:
        optional_magic = struct.unpack_from("<H", data, optional_offset)[0]
        image_size = struct.unpack_from("<I", data, optional_offset + 56)[0]
    except struct.error:
        return False
    if optional_magic not in {0x10B, 0x20B} or image_size == 0:
        return False
    for index in range(section_count):
        position = section_offset + index * 40
        try:
            raw_size, raw_offset = struct.unpack_from("<II", data, position + 16)
        except struct.error:
            return False
        if raw_size and not _region_fits(raw_offset, raw_size, file_size):
            return False
    return True


def _is_structurally_valid_executable_image(data: bytes, file_size: int) -> bool:
    """Reject magic-only impostors that a shell could interpret after ENOEXEC."""
    if file_size < 1 or len(data) < min(file_size, 4):
        return False
    if data.startswith(b"\x7fELF"):
        return _valid_elf_image(data, file_size)
    if data.startswith(b"MZ"):
        return _valid_pe_image(data, file_size)
    if data[:4] in {
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
    }:
        return _valid_thin_macho_image(data, file_size)
    if data[:4] in {
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }:
        return _valid_fat_macho_image(data, file_size)
    return False


def _read_referenced_script(path: Path) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads.

    This is the shared choke point for every local script read the guard
    performs (the terminal walk in ``_contains_unsafe_gateway_action`` AND
    the cron-script scan in ``_read_script_for_scanning``), so the
    cloud-placeholder refusal lives here: a FileProvider path must never be
    opened — not even to discover whether the file is hydrated — because an
    evicted placeholder's ``open()`` can hang preflight indefinitely
    (#88052). The lexical check covers direct cloud paths; the resolved
    check covers local launchers that are symlinks into a cloud subtree.
    """
    if _is_cloud_placeholder_path(path):
        return None, True
    try:
        resolved = path.resolve(strict=False)
    except (OSError, ValueError):
        # OSError: unreadable/long paths. ValueError: embedded NUL byte
        # from a binary's decoded contents tokenized as a path — a
        # guarded path must never crash the guard (#76762).
        resolved = path
    if _is_cloud_placeholder_path(resolved):
        return None, True
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            # Directories are not scripts. Docker Desktop writes
            # ``fpath=(~/.docker/completions …)`` into ``~/.zshrc``; the
            # walk then treats that dir as a referenced script and used
            # to fail-closed, blocking ``source ~/.zshrc`` (#86753).
            # Devices/sockets stay fail-closed.
            if stat.S_ISDIR(metadata.st_mode):
                return None, False
            return None, True
        read_limit = min(metadata.st_size, _MAX_REFERENCED_SCRIPT_BYTES + 1)
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read < read_limit:
            chunk = os.read(descriptor, read_limit - bytes_read)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        data = b"".join(chunks)
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    # A concurrent short read is ambiguous: never let a truncated safety view
    # authorize a command that will execute a different/full file.
    if len(data) != read_limit:
        return None, True
    if data.startswith(_BINARY_MAGIC_PREFIXES):
        if _is_structurally_valid_executable_image(data, metadata.st_size):
            return None, False
        return None, True
    if metadata.st_size > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return data.decode("utf-8", errors="replace"), False


def _latin1_bytes(text: str) -> Optional[bytes]:
    try:
        return text.encode("latin-1")
    except UnicodeEncodeError:
        return None


def _sanitize_remote_script_text(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Validate a bounded remote-read envelope before recursive scanning."""
    if not text:
        return None, False
    if text == _REMOTE_SCRIPT_UNSAFE_SENTINEL:
        return None, True
    if text.startswith(_REMOTE_SCRIPT_WIRE_PREFIX):
        header, separator, payload = text.partition("\n")
        if not separator:
            return None, True
        fields = header[len(_REMOTE_SCRIPT_WIRE_PREFIX) :].split(":", 1)
        if len(fields) != 2:
            return None, True
        try:
            file_size = int(fields[0])
            probe = base64.b64decode(fields[1], validate=True)
        except (ValueError, binascii.Error):
            return None, True
        expected_probe_size = min(file_size, _EXECUTABLE_PROBE_BYTES)
        if file_size < 0 or len(probe) != expected_probe_size:
            return None, True
        if probe.startswith(_BINARY_MAGIC_PREFIXES):
            if _is_structurally_valid_executable_image(probe, file_size):
                return None, False
            return None, True
        if file_size > _MAX_REFERENCED_SCRIPT_BYTES:
            return None, True
        if len(payload.encode("utf-8", errors="replace")) != file_size:
            return None, True
        return payload, False

    # Compatibility for third-party callbacks that still return a raw string.
    # They do not carry an independent size/probe envelope, so magic-only data
    # must prove its structure from the returned bytes or fail closed.
    raw = _latin1_bytes(text)
    if raw is not None and raw.startswith(_BINARY_MAGIC_PREFIXES):
        if _is_structurally_valid_executable_image(raw, len(raw)):
            return None, False
        return None, True
    if len(text.encode("utf-8", errors="replace")) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return text, False


def _contains_unsafe_gateway_action(
    command: str,
    *,
    cwd: Optional[str],
    depth: int,
    visited: set[Path],
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    if _direct_lifecycle_scan(command):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    for payload in _iter_shell_command_payloads(command):
        if _contains_unsafe_gateway_action(
            payload,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True

    for script_path in _iter_referenced_shell_scripts(command, cwd=cwd):
        # Do not touch a FileProvider path even to discover whether the file
        # is hydrated. The lexical check covers direct cloud paths; the
        # resolved check below covers local launchers that are symlinks into
        # a cloud subtree. _read_referenced_script repeats both checks as the
        # shared choke point, so every caller stays covered even if this
        # walk-level short-circuit is bypassed.
        if _is_cloud_placeholder_path(script_path):
            return True
        try:
            resolved = script_path.resolve(strict=False)
        except (OSError, ValueError):
            # OSError: unreadable/long paths. ValueError: embedded NUL byte
            # from a binary's decoded contents tokenized as a path — a
            # guarded path must never crash the guard (#76762).
            resolved = script_path
        if _is_cloud_placeholder_path(resolved):
            return True
        if resolved in visited:
            continue
        visited.add(resolved)
        if read_remote_script is not None:
            # A supplied reader represents the backend that will actually
            # execute this path and is authoritative. Never let a coincident
            # host path shadow SSH/container/sandbox content.
            try:
                remote_content = read_remote_script(str(script_path))
            except Exception:
                # A backend/protocol failure means the executable content
                # could not be inspected. Block without logging remote paths
                # or exception payloads, which may contain private data.
                logger.warning("lifecycle guard backend script read failed; blocking")
                return True
            script_text, unsafe = _sanitize_remote_script_text(remote_content)
        else:
            script_text, unsafe = _read_referenced_script(script_path)
        if unsafe:
            return True
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's
        # directory, not the original command's cwd.
        script_dir = _resolve_script_directory(str(resolved)) or cwd
        if _contains_unsafe_gateway_action(
            script_text,
            cwd=script_dir,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str,
    *,
    cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts.

    Total by construction: this function returns a verdict for *every*
    input and never raises. The direct scans below are pure string
    operations; the referenced-script walk touches the filesystem, remote
    backends, and shlex on arbitrary decoded bytes, so it is best-effort
    defense-in-depth — any unexpected failure inside it is logged and
    treated as "walk found nothing" rather than crashing the caller.

    This is the contract #76762 established ("a guarded path must never
    crash the guard") enforced at the boundary instead of per-syscall: a
    guard crash propagates out of ``tools/terminal_tool.py`` and breaks
    every terminal command until the gateway restarts (#77780, #78256),
    which is strictly worse than either verdict.
    """
    try:
        # Includes the direct regex/submit scans at depth 0.
        return _contains_unsafe_gateway_action(
            command,
            cwd=cwd,
            depth=0,
            visited=set(),
            read_remote_script=read_remote_script,
        )
    except Exception:
        logger.warning(
            "lifecycle guard referenced-script walk failed; "
            "falling back to direct-scan verdict",
            exc_info=True,
        )
        # Pure string scans of the top-level command — cannot raise.
        try:
            return _direct_lifecycle_scan(command)
        except Exception:
            # The data-argument masker tokenizes arbitrary text; if even
            # that fails, fall to the raw regex + submit scan so the guard
            # stays total.
            return contains_gateway_lifecycle_command(
                command
            ) or contains_launchctl_submit_command(command)




def _resolve_script_path(script_path: str) -> Optional[Path]:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.

    Returns ``None`` for values that cannot be a real path (NUL bytes,
    unexpandable ``~``) — the same ingestion contract as
    ``_expand_candidate_path``; such a value can never name a file the
    scheduler would execute, so there is nothing to scan.
    """
    from hermes_constants import get_hermes_home

    raw = _expand_candidate_path(script_path)
    if raw is None:
        return None
    if raw.is_absolute():
        return raw
    try:
        return get_hermes_home() / "scripts" / raw
    except (RuntimeError, OSError):
        # get_hermes_home() falls back to Path.home(), which raises when
        # neither HERMES_HOME nor HOME is resolvable (launchd/systemd
        # environments) — same ingestion contract: nothing to scan.
        return None


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded terminal-script scanner.

    Non-regular or oversized inputs fail closed by returning a lifecycle-shaped
    sentinel, while missing/unreadable/unresolvable paths remain empty so
    ordinary scheduler path validation can report them.
    """
    resolved = _resolve_script_path(script_path)
    if resolved is None:
        return ""
    script_text, unsafe = _read_referenced_script(resolved)
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a
    gateway-lifecycle command pattern.

    ``prompt`` is scanned directly.  ``script``, when supplied, is read from
    disk and concatenated for the scan.  Both are considered together so a
    job cannot slip through by splitting the command across the prompt and
    the script.

    Callers should let the exception propagate when they want the create to
    fail with a ``ValueError``-shaped error (the agent's ``cronjob`` tool
    surfaces this as a tool error; the CLI prints it in red and exits 1).
    """
    combined = prompt or ""
    python_script = False
    if script:
        resolved_script = _resolve_script_path(script)
        if resolved_script is not None:
            try:
                real_script = resolved_script.resolve(strict=False)
            except (OSError, ValueError):
                real_script = resolved_script
            if _is_cloud_placeholder_path(resolved_script) or _is_cloud_placeholder_path(
                real_script
            ):
                # Attribute the refusal correctly: the script is not known to
                # contain a lifecycle command — it lives on a cloud-synced
                # FileProvider path (iCloud Drive / ~/Library/CloudStorage)
                # that the guard refuses to open because an evicted
                # placeholder can hang preflight indefinitely (#88052).
                # Fail closed with the real reason instead of implying a
                # dangerous lifecycle command.
                raise GatewayLifecycleBlocked(
                    "Blocked: the cron script lives on a cloud-synced path "
                    "(iCloud Drive / ~/Library/CloudStorage). Opening an "
                    "evicted FileProvider placeholder can hang the guard's "
                    "preflight scan indefinitely, so it is refused without "
                    "being read. Move the script to a local, non-cloud path "
                    "(e.g. ~/.hermes/scripts/) and recreate the job."
                )
        python_script = resolved_script is not None and resolved_script.suffix == ".py"
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if python_script:
        # Python is executed by the interpreter, never through a POSIX
        # shell: the shell-script reference walk is a false-positive
        # generator on Python sources (pathlib's "/" operator resolves to
        # the filesystem root and trips the regular-file check, blocking
        # every innocent .py cron script, #77131). The direct command
        # regex below still scans the full text, so a literal
        # `hermes gateway restart` embedded in a .py script is still
        # blocked. Non-regular/oversized script files still fail closed
        # via the lifecycle-shaped sentinel in _read_script_for_scanning.
        unsafe = _lifecycle_command_scan_with_data_exemption(combined)
    else:
        script_dir = _resolve_script_directory(script) if script else None
        unsafe = contains_gateway_lifecycle_command_or_referenced_script(
            combined,
            cwd=script_dir,
        )
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
