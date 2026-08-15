"""Post-delivery Gateway event contract for user hooks."""

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    RESPONSE_DELIVERY_RECEIPT_KEY,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner, _prepare_gateway_response_for_delivery
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="out-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _setup():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="100000001",
        chat_type="dm",
        user_id="test-user",
        thread_id="9001",
    )
    event = MessageEvent(
        text="请完成这项工作。" * 80,
        source=source,
        message_id="in-42",
    )
    adapter = _Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.hooks = AsyncMock()
    runner._background_tasks = set()
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda _source: adapter
    return runner, adapter, event, source


def _register(runner, event, source):
    runner._register_response_delivered_hook(
        event=event,
        source=source,
        session_key="agent:main:telegram:dm:100000001:9001",
        session_id="session-abc",
        run_generation=7,
        request_text=event.text,
        response_text="完整最终回复。" * 100,
    )


async def _invoke(callback):
    result = callback()
    if inspect.isawaitable(result):
        await result
    await asyncio.sleep(0)


async def _drain_hook_tasks(runner):
    tasks = list(runner._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_response_delivered_hook_requires_confirmed_success_receipt():
    runner, adapter, event, source = _setup()
    _register(runner, event, source)

    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await _invoke(callback)

    runner.hooks.emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_delivered_hook_emits_full_correlated_payload_once():
    runner, adapter, event, source = _setup()
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-99",
        "message_ids": ["out-97", "out-98", "out-99"],
        "mode": "text",
    }

    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await _invoke(callback)
    await _invoke(callback)  # receipt is consumed; duplicate invocation is a no-op
    await _drain_hook_tasks(runner)

    runner.hooks.emit.assert_awaited_once()
    event_type, context = runner.hooks.emit.await_args.args
    assert event_type == "response:delivered"
    assert context["schema_version"] == 1
    assert context["turn_id"] == "session-abc:telegram:in-42"
    assert context["session_id"] == "session-abc"
    assert context["session_key"] == "agent:main:telegram:dm:100000001:9001"
    assert context["platform"] == "telegram"
    assert context["chat_id"] == "100000001"
    assert context["thread_id"] == "9001"
    assert context["user_id"] == "test-user"
    assert context["inbound_message_id"] == "in-42"
    assert context["delivery_message_id"] == "out-99"
    assert context["platform_message_ids"] == ["out-97", "out-98", "out-99"]
    assert context["delivery_mode"] == "text"
    assert context["delivery_metadata"]["thread_id"] == "9001"
    assert context["delivery_metadata"]["direct_messages_topic_id"] == "9001"
    assert context["request"] == event.text
    assert context["response"] == "完整最终回复。" * 100
    assert context["run_generation"] == 7


@pytest.mark.asyncio
async def test_base_adapter_successful_text_send_produces_receipt_and_event():
    runner, adapter, event, source = _setup()
    session_key = "agent:main:telegram:dm:100000001:9001"
    interrupt_event = asyncio.Event()
    setattr(interrupt_event, "_hermes_run_generation", 7)
    adapter._active_sessions[session_key] = interrupt_event

    async def _handler(_event):
        return "完整最终回复。" * 100

    adapter.set_message_handler(_handler)
    _register(runner, event, source)

    await adapter._process_message_background(event, session_key)
    await _drain_hook_tasks(runner)

    runner.hooks.emit.assert_awaited_once()
    event_type, context = runner.hooks.emit.await_args.args
    assert event_type == "response:delivered"
    assert context["delivery_message_id"] == "out-1"
    assert context["platform_message_ids"] == ["out-1"]
    assert context["delivery_mode"] == "text"
    assert RESPONSE_DELIVERY_RECEIPT_KEY not in event.metadata


@pytest.mark.asyncio
async def test_normal_final_send_merges_prior_partial_stream_message_ids():
    runner, adapter, event, source = _setup()
    session_key = "agent:main:telegram:dm:100000001:9001"
    interrupt_event = asyncio.Event()
    setattr(interrupt_event, "_hermes_run_generation", 7)
    adapter._active_sessions[session_key] = interrupt_event
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": False,
        "message_id": "partial-2",
        "message_ids": ["partial-1", "partial-2"],
        "mode": "stream_partial",
    }

    async def _handler(_event):
        return "最终完整回复。"

    adapter.set_message_handler(_handler)
    _register(runner, event, source)

    await adapter._process_message_background(event, session_key)
    await _drain_hook_tasks(runner)

    _, context = runner.hooks.emit.await_args.args
    assert context["delivery_message_id"] == "out-1"
    assert context["platform_message_ids"] == [
        "partial-1",
        "partial-2",
        "out-1",
    ]


@pytest.mark.asyncio
async def test_response_delivered_dispatch_does_not_block_ordered_callbacks():
    runner, adapter, event, source = _setup()
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    callback_order = []

    async def _slow_emit(*_args):
        hook_started.set()
        await release_hook.wait()

    runner.hooks.emit.side_effect = _slow_emit
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-async",
        "message_ids": ["out-async"],
        "mode": "text",
    }
    adapter.register_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001",
        lambda: callback_order.append("after-response-hook"),
        generation=7,
    )

    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await asyncio.wait_for(_invoke(callback), timeout=0.1)

    assert callback_order == ["after-response-hook"]
    assert hook_started.is_set()
    assert runner._background_tasks

    release_hook.set()
    await _drain_hook_tasks(runner)
    assert not runner._background_tasks


@pytest.mark.asyncio
async def test_hook_task_is_owned_and_cleans_up_on_cancellation():
    runner, adapter, event, source = _setup()
    started = asyncio.Event()

    async def _slow_emit(*_args):
        started.set()
        await asyncio.Event().wait()

    runner.hooks.emit.side_effect = _slow_emit
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-owned",
        "message_ids": ["out-owned"],
    }
    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None

    await _invoke(callback)
    await started.wait()
    tasks = list(runner._response_delivery_hook_tasks)
    assert tasks
    assert set(tasks).issubset(runner._background_tasks)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert not runner._response_delivery_hook_tasks
    assert not runner._background_tasks


@pytest.mark.asyncio
async def test_failed_send_does_not_emit_response_delivered():
    runner, adapter, event, source = _setup()
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    session_key = "agent:main:telegram:dm:100000001:9001"
    interrupt_event = asyncio.Event()
    setattr(interrupt_event, "_hermes_run_generation", 7)
    adapter._active_sessions[session_key] = interrupt_event
    adapter.set_message_handler(AsyncMock(return_value="not delivered"))
    _register(runner, event, source)

    await adapter._process_message_background(event, session_key)
    await asyncio.sleep(0)

    runner.hooks.emit.assert_not_awaited()
    assert RESPONSE_DELIVERY_RECEIPT_KEY not in event.metadata


@pytest.mark.asyncio
async def test_audio_success_does_not_confirm_failed_authoritative_text(
    monkeypatch, tmp_path
):
    runner, adapter, event, source = _setup()
    session_key = "agent:main:telegram:dm:100000001:9001"
    interrupt_event = asyncio.Event()
    setattr(interrupt_event, "_hermes_run_generation", 7)
    adapter._active_sessions[session_key] = interrupt_event
    event.message_type = MessageType.VOICE
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="audio-1")
    )
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    audio_path = tmp_path / "reply.ogg"
    audio_path.write_bytes(b"audio")

    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda **_kwargs: json.dumps(
            {"success": True, "file_path": str(audio_path)}
        ),
    )
    adapter.set_message_handler(AsyncMock(return_value="最终权威文本。" * 300))
    _register(runner, event, source)

    await adapter._process_message_background(event, session_key)
    await _drain_hook_tasks(runner)

    adapter.play_tts.assert_awaited_once()
    assert adapter.send.await_count >= 1
    runner.hooks.emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_multi_file_tts_keeps_authoritative_caption_receipt(monkeypatch, tmp_path):
    runner, adapter, event, source = _setup()
    session_key = "agent:main:telegram:dm:100000001:9001"
    interrupt_event = asyncio.Event()
    setattr(interrupt_event, "_hermes_run_generation", 7)
    adapter._active_sessions[session_key] = interrupt_event
    event.message_type = MessageType.VOICE
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id="caption-audio-1"),
            SendResult(success=True, message_id="voice-only-2"),
        ]
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-3"))
    first_audio = tmp_path / "reply-1.ogg"
    second_audio = tmp_path / "reply-2.ogg"
    first_audio.write_bytes(b"audio-1")
    second_audio.write_bytes(b"audio-2")

    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda **_kwargs: json.dumps(
            {
                "success": True,
                "file_paths": [str(first_audio), str(second_audio)],
            }
        ),
    )
    adapter.set_message_handler(AsyncMock(return_value="短回复，使用首段语音 caption。"))
    _register(runner, event, source)

    await adapter._process_message_background(event, session_key)
    await _drain_hook_tasks(runner)

    assert adapter.play_tts.await_count == 2
    adapter.send.assert_not_awaited()
    runner.hooks.emit.assert_awaited_once()
    context = runner.hooks.emit.await_args.args[1]
    assert context["delivery_message_id"] == "caption-audio-1"
    assert context["platform_message_ids"] == ["caption-audio-1"]
    assert context["delivery_mode"] == "tts_caption"


@pytest.mark.asyncio
async def test_internal_delivery_reports_explicit_default_profile():
    runner, adapter, event, source = _setup()
    event.internal = True
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-internal",
        "message_ids": ["out-internal"],
        "mode": "text",
    }

    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await _invoke(callback)
    await _drain_hook_tasks(runner)

    context = runner.hooks.emit.await_args.args[1]
    assert context["internal"] is True
    assert context["profile"] == "default"


@pytest.mark.asyncio
async def test_multiplexed_delivery_reports_source_profile():
    runner, adapter, event, source = _setup()
    runner.config.multiplex_profiles = True
    source.profile = "research"
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-profile",
        "message_ids": ["out-profile"],
        "mode": "text",
    }

    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await _invoke(callback)
    await _drain_hook_tasks(runner)

    assert runner.hooks.emit.await_args.args[1]["profile"] == "research"


@pytest.mark.asyncio
async def test_generation_mismatch_does_not_consume_or_emit_current_callback():
    runner, adapter, event, source = _setup()
    _register(runner, event, source)
    event.metadata[RESPONSE_DELIVERY_RECEIPT_KEY] = {
        "success": True,
        "message_id": "out-current",
        "message_ids": ["out-current"],
        "mode": "text",
    }

    assert adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=6
    ) is None
    callback = adapter.pop_post_delivery_callback(
        "agent:main:telegram:dm:100000001:9001", generation=7
    )
    assert callback is not None
    await _invoke(callback)
    await _drain_hook_tasks(runner)

    runner.hooks.emit.assert_awaited_once()


def test_delivery_response_is_normalized_and_sanitized_before_snapshot():
    secret = "sk-" + "live-" + "abcdef1234567890"
    prepared = _prepare_gateway_response_for_delivery(
        Platform.TELEGRAM,
        {"failed": True, "error": f"HTTP 401 api_key={secret}"},
        "",
        intentional_silence=False,
    )

    assert secret not in prepared
    assert prepared != f"The request failed: HTTP 401 api_key={secret}"


def test_empty_sentinel_becomes_visible_delivery_text():
    prepared = _prepare_gateway_response_for_delivery(
        Platform.TELEGRAM,
        {"api_calls": 1},
        "(empty)",
        intentional_silence=False,
    )

    assert prepared
    assert prepared != "(empty)"
    assert "model returned no response" in prepared
