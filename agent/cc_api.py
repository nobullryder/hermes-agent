from __future__ import annotations

import base64
import copy
import json
import mimetypes
import threading
import urllib.request
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx


def _build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _sse_events(response: httpx.Response):
    event_name: str | None = None
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _json_response(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    raise ValueError("Gateway returned a non-object JSON payload.")


def _tool_choice_value(
    tools: list[dict[str, Any]],
    requested: str | dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    if not tools:
        return "none"
    if requested is None:
        return "auto"
    if isinstance(requested, str):
        requested_lower = requested.strip().lower()
        if requested_lower in {"auto", "none", "required"}:
            return requested_lower
        return requested
    if isinstance(requested, dict):
        return copy.deepcopy(requested)
    return "auto"


def _render_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    parts.append(str(item.get("text") or ""))
                elif item_type in {"image_url", "input_image"}:
                    parts.append("[image]")
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return json.dumps(content)
    return "" if content is None else str(content)


def _content_blocks_to_gateway(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _render_text_content(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part.strip():
                blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            text = str(part).strip()
            if text:
                blocks.append({"type": "text", "text": text})
            continue

        ptype = str(part.get("type") or "").strip()
        if ptype in {"text", "input_text"}:
            text = str(part.get("text") or "")
            if text.strip():
                blocks.append({"type": "text", "text": text})
            continue

        if ptype in {"image_url", "input_image"}:
            source = _image_part_to_gateway(part)
            if source is not None:
                blocks.append(source)
            continue

        if ptype == "document":
            blocks.append({"type": "document", "source": dict(part.get("source") or {})})
            continue

        text = _render_text_content(part)
        if text:
            blocks.append({"type": "text", "text": text})

    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text") or "")
    return blocks


def _image_part_to_gateway(part: dict[str, Any]) -> dict[str, Any] | None:
    image_value = part.get("image_url")
    url = ""
    if isinstance(image_value, dict):
        url = str(image_value.get("url") or "")
    elif isinstance(image_value, str):
        url = image_value
    elif isinstance(part.get("image_url"), str):
        url = str(part.get("image_url") or "")

    if not url and isinstance(part.get("image"), dict):
        image_value = part.get("image") or {}
        url = str(image_value.get("url") or "")

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else "image/png"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type or "image/png",
                "data": data,
            },
        }

    if url.startswith("http://") or url.startswith("https://"):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = resp.read()
                media_type = resp.headers.get_content_type() or mimetypes.guess_type(url)[0] or "image/png"
        except Exception:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(raw).decode("ascii"),
            },
        }

    return None


def _assistant_dict_from_response(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    payload = {
        "role": "assistant",
        "content": message.content or "",
    }
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _normalize_message_for_sync(message: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(message)
    normalized.pop("finish_reason", None)
    normalized.pop("reasoning", None)
    normalized.pop("reasoning_content", None)
    normalized.pop("reasoning_details", None)
    if isinstance(normalized.get("tool_calls"), list):
        canonical_tool_calls: list[dict[str, Any]] = []
        for tool_call in normalized.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            canonical_tool_calls.append(
                {
                    "id": str(tool_call.get("id") or tool_call.get("call_id") or ""),
                    "type": str(tool_call.get("type") or "function"),
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or ""),
                    },
                }
            )
        normalized["tool_calls"] = canonical_tool_calls
    return normalized


def _messages_prefix_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    normalized_left = [_normalize_message_for_sync(message) for message in left]
    normalized_right = [_normalize_message_for_sync(message) for message in right]
    return json.dumps(normalized_left, sort_keys=True, ensure_ascii=True) == json.dumps(
        normalized_right, sort_keys=True, ensure_ascii=True
    )


class CCAPIAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        hermes_session_id: str,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = model
        self.hermes_session_id = hermes_session_id
        self.gateway_session_id: str | None = None
        self.synced_messages: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def create(self, api_kwargs: dict[str, Any]) -> Any:
        return self._execute(api_kwargs, stream=False)

    def stream(
        self,
        api_kwargs: dict[str, Any],
        *,
        on_first_delta: Callable[[], None] | None = None,
        stream_delta_callback: Callable[[str], None] | None = None,
        tool_gen_callback: Callable[[str], None] | None = None,
    ) -> Any:
        return self._execute(
            api_kwargs,
            stream=True,
            on_first_delta=on_first_delta,
            stream_delta_callback=stream_delta_callback,
            tool_gen_callback=tool_gen_callback,
        )

    def interrupt(self) -> None:
        with self._lock:
            if not self.gateway_session_id:
                return
            with httpx.Client(timeout=10.0) as client:
                client.post(
                    f"{self.base_url}/api/sessions/{self.gateway_session_id}/interrupt",
                    headers=_build_headers(self.api_key),
                )

    def compact(self, *, keep_recent_messages: int, messages: list[dict[str, Any]]) -> None:
        with self._lock:
            if not self.gateway_session_id:
                return
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/api/sessions/{self.gateway_session_id}/compact",
                    headers=_build_headers(self.api_key),
                    json={"keep_recent_messages": keep_recent_messages},
                )
                response.raise_for_status()
            self.synced_messages = copy.deepcopy(messages)

    def _execute(
        self,
        api_kwargs: dict[str, Any],
        *,
        stream: bool,
        on_first_delta: Callable[[], None] | None = None,
        stream_delta_callback: Callable[[str], None] | None = None,
        tool_gen_callback: Callable[[str], None] | None = None,
    ) -> Any:
        messages = copy.deepcopy(list(api_kwargs.get("messages") or []))
        tools = copy.deepcopy(list(api_kwargs.get("tools") or []))
        with self._lock:
            self._api_state = {
                "tool_choice": copy.deepcopy(api_kwargs.get("tool_choice")),
                "parallel_tool_calls": api_kwargs.get("parallel_tool_calls"),
            }
            system_prompt = self._extract_system_prompt(messages)
            try:
                payload_builder = self._build_turn_payload(messages, tools, system_prompt)

                timeout = float(api_kwargs.get("timeout") or 1800.0)
                with httpx.Client(timeout=httpx.Timeout(connect=30.0, read=timeout, write=timeout, pool=30.0)) as client:
                    if stream:
                        response = self._execute_stream_request(
                            client=client,
                            payload_builder=payload_builder,
                            on_first_delta=on_first_delta,
                            stream_delta_callback=stream_delta_callback,
                            tool_gen_callback=tool_gen_callback,
                        )
                    else:
                        response = self._execute_stream_request(
                            client=client,
                            payload_builder=payload_builder,
                        )
            finally:
                self._api_state = {}

            self.synced_messages = [_normalize_message_for_sync(message) for message in messages]
            self.synced_messages.append(_normalize_message_for_sync(_assistant_dict_from_response(response)))
            return response

    def _ensure_session(self, client: httpx.Client, system_prompt: str) -> None:
        if self.gateway_session_id:
            return
        response = client.post(
            f"{self.base_url}/api/sessions",
            headers=_build_headers(self.api_key),
            json={
                "model": self.model,
                "system_prompt": system_prompt or None,
                "metadata": {"hermes_session_id": self.hermes_session_id},
            },
        )
        payload = _json_response(response)
        self.gateway_session_id = str(payload.get("id") or "")
        self.synced_messages = []

    def _build_turn_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        requested_tool_choice = api_tool_choice = None
        requested_parallel_tool_calls = True
        # api_kwargs are threaded in via _execute() and attached temporarily.
        api_state = getattr(self, "_api_state", None) or {}
        requested_tool_choice = api_state.get("tool_choice")
        if api_state.get("parallel_tool_calls") is False:
            requested_parallel_tool_calls = False

        non_system = [msg for msg in messages if msg.get("role") != "system"]
        with httpx.Client(timeout=30.0) as session_client:
            self._ensure_session(session_client, system_prompt)

        if not self.synced_messages:
            return self._bootstrap_payload(
                non_system,
                tools,
                requested_tool_choice=requested_tool_choice,
                parallel_tool_calls=requested_parallel_tool_calls,
            )

        if len(messages) < len(self.synced_messages) or not _messages_prefix_equal(
            messages[: len(self.synced_messages)], self.synced_messages
        ):
            return self._bootstrap_payload(
                non_system,
                tools,
                requested_tool_choice=requested_tool_choice,
                parallel_tool_calls=requested_parallel_tool_calls,
            )

        delta = messages[len(self.synced_messages) :]
        if not delta:
            raise ValueError("Gateway adapter received no new messages to send.")

        if all(msg.get("role") == "tool" for msg in delta):
            return self._tool_results_payload(
                delta,
                tools,
                requested_tool_choice=requested_tool_choice,
                parallel_tool_calls=requested_parallel_tool_calls,
            )

        if len(delta) == 1 and delta[0].get("role") == "user":
            return self._user_turn_payload(
                delta[0],
                tools,
                requested_tool_choice=requested_tool_choice,
                parallel_tool_calls=requested_parallel_tool_calls,
            )

        return self._bootstrap_payload(
            non_system,
            tools,
            requested_tool_choice=requested_tool_choice,
            parallel_tool_calls=requested_parallel_tool_calls,
        )

    def _bootstrap_payload(
        self,
        non_system_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        requested_tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        if len(non_system_messages) == 1 and non_system_messages[0].get("role") == "user":
            return self._user_turn_payload(
                non_system_messages[0],
                tools,
                requested_tool_choice=requested_tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )

        transcript: list[str] = [
            "Resume this Hermes conversation state faithfully.",
            "Previous transcript:",
        ]
        for msg in non_system_messages:
            role = str(msg.get("role") or "unknown").upper()
            content = _render_text_content(msg.get("content"))
            transcript.append(f"{role}: {content}")
            if msg.get("tool_calls"):
                transcript.append(
                    f"ASSISTANT_TOOL_CALLS: {json.dumps(msg.get('tool_calls'), ensure_ascii=True)}"
                )
        transcript.append("Continue from the latest state.")
        return (
            f"/api/sessions/{self.gateway_session_id}/turns",
            {
                "input": "\n".join(transcript).strip(),
                "tools": tools or None,
                "tool_choice": _tool_choice_value(tools, requested_tool_choice),
                "parallel_tool_calls": parallel_tool_calls,
                "stream": True,
            },
        )

    def _user_turn_payload(
        self,
        message: dict[str, Any],
        tools: list[dict[str, Any]],
        *,
        requested_tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        return (
            f"/api/sessions/{self.gateway_session_id}/turns",
            {
                "input": _content_blocks_to_gateway(message.get("content")),
                "tools": tools or None,
                "tool_choice": _tool_choice_value(tools, requested_tool_choice),
                "parallel_tool_calls": parallel_tool_calls,
                "stream": True,
            },
        )

    def _tool_results_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        requested_tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        results = []
        for msg in messages:
            tool_call_id = str(msg.get("tool_call_id") or "")
            if not tool_call_id:
                continue
            results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": msg.get("content"),
                    "name": msg.get("name"),
                    "is_error": False,
                }
            )
        return (
            f"/api/sessions/{self.gateway_session_id}/tool-results",
            {
                "results": results,
                "tools": tools or None,
                "tool_choice": _tool_choice_value(tools, requested_tool_choice),
                "parallel_tool_calls": parallel_tool_calls,
                "stream": True,
            },
        )

    def _execute_stream_request(
        self,
        *,
        client: httpx.Client,
        payload_builder: tuple[str, dict[str, Any]],
        on_first_delta: Callable[[], None] | None = None,
        stream_delta_callback: Callable[[str], None] | None = None,
        tool_gen_callback: Callable[[str], None] | None = None,
    ) -> Any:
        path, payload = payload_builder
        first_delta_fired = False
        turn_response: dict[str, Any] | None = None
        assistant_text_parts: list[str] = []

        with client.stream(
            "POST",
            f"{self.base_url}{path}",
            headers=_build_headers(self.api_key),
            json=payload,
        ) as response:
            response.raise_for_status()
            for event_name, raw_data in _sse_events(response):
                if not raw_data:
                    continue
                data = json.loads(raw_data)
                if event_name == "assistant.completed":
                    text = str((data.get("data") or {}).get("output_text") or "")
                    if text:
                        assistant_text_parts.append(text)
                    if text and stream_delta_callback:
                        if not first_delta_fired and on_first_delta:
                            on_first_delta()
                            first_delta_fired = True
                        stream_delta_callback(text)
                elif event_name == "tool_call.requested":
                    tool_name = str((data.get("data") or {}).get("name") or "")
                    if tool_name and tool_gen_callback:
                        if not first_delta_fired and on_first_delta:
                            on_first_delta()
                            first_delta_fired = True
                        tool_gen_callback(tool_name)
                elif event_name == "turn.response":
                    turn_response = data
                elif event_name == "error":
                    message = str(data.get("message") or "Gateway request failed.")
                    raise RuntimeError(message)
                elif event_name == "done" and data.get("status") == "interrupted":
                    raise InterruptedError("Gateway turn interrupted.")

        if turn_response is None:
            raise ValueError("Gateway stream ended without a turn.response event.")
        if not (turn_response.get("output_text") or "").strip():
            streamed_text = "".join(assistant_text_parts).strip()
            if streamed_text:
                turn_response = dict(turn_response)
                turn_response["output_text"] = streamed_text
        return self._wrap_turn_response(turn_response)

    def _extract_system_prompt(self, messages: list[dict[str, Any]]) -> str:
        if messages and messages[0].get("role") == "system":
            return _render_text_content(messages[0].get("content"))
        return ""

    def _wrap_turn_response(self, payload: dict[str, Any]) -> Any:
        tool_calls = []
        for item in payload.get("tool_calls") or []:
            function = item.get("function") or {}
            tool_calls.append(
                SimpleNamespace(
                    id=item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    type=item.get("type", "function"),
                    function=SimpleNamespace(
                        name=function.get("name", ""),
                        arguments=function.get("arguments", "{}"),
                    ),
                )
            )

        message = SimpleNamespace(
            role="assistant",
            content=payload.get("output_text") or "",
            tool_calls=tool_calls or None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason=payload.get("finish_reason") or "stop",
        )
        usage_payload = payload.get("usage") or {}
        usage = SimpleNamespace(
            prompt_tokens=int(usage_payload.get("prompt_tokens") or 0),
            completion_tokens=int(usage_payload.get("completion_tokens") or 0),
            total_tokens=int(usage_payload.get("total_tokens") or 0),
        )
        return SimpleNamespace(
            id=payload.get("id") or f"gateway-{uuid.uuid4()}",
            model=payload.get("model") or self.model,
            choices=[choice],
            usage=usage,
        )
