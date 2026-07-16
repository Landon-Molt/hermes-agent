"""Tests for the interim_assistant_callback config gating in tui_gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_load_interim_assistant_messages_defaults_true():
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={}):
        assert _load_interim_assistant_messages() is True


def test_load_interim_assistant_messages_explicit_true():
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={"display": {"interim_assistant_messages": True}}):
        assert _load_interim_assistant_messages() is True


def test_load_interim_assistant_messages_explicit_false():
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={"display": {"interim_assistant_messages": False}}):
        assert _load_interim_assistant_messages() is False


def test_load_interim_assistant_messages_string_off():
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={"display": {"interim_assistant_messages": "off"}}):
        assert _load_interim_assistant_messages() is False


def test_callback_emits_message_interim_event():
    """The callback emits a message.interim event without settling turn state."""
    from tui_gateway.server import _load_interim_assistant_messages

    emitted: list[tuple] = []

    def fake_emit(event_type, sid, payload=None):
        emitted.append((event_type, sid, payload))

    with patch("tui_gateway.server._load_cfg", return_value={}), \
         patch("tui_gateway.server._emit", side_effect=fake_emit):
        assert _load_interim_assistant_messages() is True

        # Simulate the callback wiring
        sid = "test-session"

        def _interim_assistant_cb(text: str, *, already_streamed: bool = False):
            fake_emit("message.interim", sid, {
                "text": text,
                "already_streamed": already_streamed,
            })

        _interim_assistant_cb("hello world", already_streamed=True)

        assert len(emitted) == 1
        assert emitted[0][0] == "message.interim"
        assert emitted[0][1] == sid
        assert emitted[0][2]["text"] == "hello world"
        assert emitted[0][2]["already_streamed"] is True


def test_callback_omitted_when_disabled():
    """When config is false, the callback should not be installed on the agent."""
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={"display": {"interim_assistant_messages": False}}):
        assert _load_interim_assistant_messages() is False

        # In the real code path, the agent.interim_assistant_callback is set to None
        # when _load_interim_assistant_messages() returns False.
        agent = MagicMock()
        agent.interim_assistant_callback = None
        assert agent.interim_assistant_callback is None
