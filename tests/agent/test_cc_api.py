from __future__ import annotations

import json
from unittest.mock import patch

from agent.cc_api import CCAPIAdapter


class _FakeJsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeStreamResponse:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for event_name, payload in self._events:
            if event_name is not None:
                yield f"event: {event_name}"
            yield f"data: {json.dumps(payload)}"
            yield ""


class _FakeHttpxClient:
    calls = []
    stream_events = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append(("POST", url, json))
        return _FakeJsonResponse({"id": "sess_gateway_1"})

    def stream(self, method, url, headers=None, json=None):
        self.calls.append((method, url, json))
        events = self.stream_events.pop(0)
        return _FakeStreamResponse(events)


def test_gateway_adapter_user_turn(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("turn.started", {"type": "turn.started", "data": {"input": "Hi"}}),
        ("assistant.completed", {"type": "assistant.completed", "data": {"output_text": "hello"}}),
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "hello",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    response = adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Hi"},
            ],
            "tools": [],
        }
    )

    assert response.choices[0].message.content == "hello"
    assert _FakeHttpxClient.calls[0][1].endswith("/api/sessions")
    assert _FakeHttpxClient.calls[1][1].endswith("/api/sessions/sess_gateway_1/turns")


def test_gateway_adapter_tool_result_continuation(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [
        [
            ("tool_call.requested", {"type": "tool_call.requested", "data": {"name": "get_weather"}}),
            ("turn.response", {
                "id": "turn_1",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"Vancouver"}'},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }),
            ("done", {"status": "completed"}),
        ],
        [
            ("tool_result.accepted", {"type": "tool_result.accepted", "data": {"tool_call_id": "call_weather_1"}}),
            ("turn.response", {
                "id": "turn_2",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "It is 12C.",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }),
            ("done", {"status": "completed"}),
        ],
    ]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )

    first = adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Weather?"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        }
    )
    assert first.choices[0].message.tool_calls[0].function.name == "get_weather"

    second = adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"Vancouver"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_weather_1", "content": "Vancouver is 12C."},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        }
    )

    assert second.choices[0].message.content == "It is 12C."
    assert any(call[1].endswith("/tool-results") for call in _FakeHttpxClient.calls)


def test_gateway_adapter_tool_result_continuation_ignores_finish_reason_metadata(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [
        [
            ("turn.response", {
                "id": "turn_1",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "saved",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_memory_1",
                        "type": "function",
                        "function": {"name": "memory", "arguments": '{"action":"add","target":"user","content":"Prefers Helix."}'},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }),
            ("done", {"status": "completed"}),
        ],
        [
            ("turn.response", {
                "id": "turn_2",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "saved",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }),
            ("done", {"status": "completed"}),
        ],
    ]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )

    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Remember I prefer Helix as my editor."},
            ],
            "tools": [{"type": "function", "function": {"name": "memory", "parameters": {"type": "object"}}}],
        }
    )

    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Remember I prefer Helix as my editor."},
                {
                    "role": "assistant",
                    "content": "saved",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_memory_1",
                            "type": "function",
                            "function": {"name": "memory", "arguments": '{"action":"add","target":"user","content":"Prefers Helix."}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_memory_1", "content": '{"success":true}'},
            ],
            "tools": [{"type": "function", "function": {"name": "memory", "parameters": {"type": "object"}}}],
        }
    )

    assert any(call[1].endswith("/tool-results") for call in _FakeHttpxClient.calls)


def test_gateway_adapter_tool_result_continuation_ignores_tool_call_metadata(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [
        [
            ("tool_call.requested", {"type": "tool_call.requested", "data": {"name": "terminal"}}),
            ("turn.response", {
                "id": "turn_1",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_terminal_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }),
            ("done", {"status": "completed"}),
        ],
        [
            ("tool_result.accepted", {"type": "tool_result.accepted", "data": {"tool_call_id": "call_terminal_1"}}),
            ("turn.response", {
                "id": "turn_2",
                "model": "anthropic/claude-opus-4.6",
                "output_text": "/Users/austin/Desktop/hermes-agent-main",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }),
            ("done", {"status": "completed"}),
        ],
    ]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )

    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Use terminal."},
            ],
            "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}],
        }
    )

    second = adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Use terminal."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_terminal_1",
                            "call_id": "call_terminal_1",
                            "response_item_id": "fc_call_terminal_1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_terminal_1",
                    "content": '{"output":"/Users/austin/Desktop/hermes-agent-main","exit_code":0,"error":null}',
                },
            ],
            "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}],
        }
    )

    assert second.choices[0].message.content == "/Users/austin/Desktop/hermes-agent-main"
    assert any(call[1].endswith("/tool-results") for call in _FakeHttpxClient.calls)


def test_gateway_adapter_falls_back_to_assistant_completed_text(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("assistant.completed", {"type": "assistant.completed", "data": {"output_text": "Got it! I'll remember that."}}),
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    response = adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Remember your name is Koda"},
            ],
            "tools": [],
        }
    )

    assert response.choices[0].message.content == "Got it! I'll remember that."


def test_gateway_adapter_preserves_requested_tool_choice(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_memory_1",
                    "type": "function",
                    "function": {"name": "memory", "arguments": '{"action":"add","content":"Koda"}'},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Remember your name is Koda"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "memory", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "memory"}},
            "parallel_tool_calls": False,
        }
    )

    payload = _FakeHttpxClient.calls[1][2]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "memory"}}
    assert payload["parallel_tool_calls"] is False


def test_gateway_adapter_interrupt_posts_interrupt_endpoint(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "hello",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Hi"},
            ],
            "tools": [],
        }
    )

    adapter.interrupt()

    assert _FakeHttpxClient.calls[-1][1].endswith("/api/sessions/sess_gateway_1/interrupt")


def test_gateway_adapter_compact_posts_compact_endpoint(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "hello",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": "Hi"},
            ],
            "tools": [],
        }
    )

    compact_messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "hello"},
    ]
    adapter.compact(keep_recent_messages=2, messages=compact_messages)

    assert _FakeHttpxClient.calls[-1][1].endswith("/api/sessions/sess_gateway_1/compact")
    assert _FakeHttpxClient.calls[-1][2] == {"keep_recent_messages": 2}
    assert adapter.synced_messages == compact_messages


def test_gateway_adapter_converts_multimodal_user_turn(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("turn.response", {
            "id": "turn_1",
            "model": "anthropic/claude-opus-4.6",
            "output_text": "I can see it.",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
        ("done", {"status": "completed"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )
    adapter.create(
        {
            "messages": [
                {"role": "system", "content": "You are Hermes."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,QUFBQQ==",
                        },
                    ],
                },
            ],
            "tools": [],
        }
    )

    payload = _FakeHttpxClient.calls[1][2]
    assert isinstance(payload["input"], list)
    assert payload["input"][0] == {"type": "text", "text": "Describe this"}
    assert payload["input"][1]["type"] == "image"
    assert payload["input"][1]["source"]["type"] == "base64"
    assert payload["input"][1]["source"]["data"] == "QUFBQQ=="


def test_gateway_adapter_raises_gateway_error_event(monkeypatch):
    monkeypatch.setattr("agent.cc_api.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.calls = []
    _FakeHttpxClient.stream_events = [[
        ("error", {"message": "Gateway exploded"}),
    ]]

    adapter = CCAPIAdapter(
        base_url="http://127.0.0.1:8000",
        api_key="secret",
        model="anthropic/claude-opus-4.6",
        hermes_session_id="hermes_1",
    )

    with patch("agent.cc_api._json_response", return_value={"id": "sess_gateway_1"}):
        try:
            adapter.create(
                {
                    "messages": [
                        {"role": "system", "content": "You are Hermes."},
                        {"role": "user", "content": "Hi"},
                    ],
                    "tools": [],
                }
            )
        except RuntimeError as exc:
            assert "Gateway exploded" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for gateway error event")
