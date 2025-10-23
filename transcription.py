from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence


MODEL_CONFIG = {
    "whisper": {
        "api_model": "whisper-1",
        "response_format": "text",
        "use_prompt": True,
        "chunking_strategy": None,
        "supports_stream": False,
    },
    "gpt-4o-transcribe-diarize": {
        "api_model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "use_prompt": False,
        "chunking_strategy": "auto",
        "supports_stream": True,
    },
}


@dataclass
class TranscriptionUpdate:
    text: str
    is_final: bool
    prompt_append: str | None


def stream_transcriptions(
    client: Any,
    model_key: str,
    message: Any,
    history: Iterable[Any],
    system_prompt: str,
) -> Iterator[str]:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unsupported transcription model: {model_key}")

    config = MODEL_CONFIG[model_key]
    prompt = _build_prompt(history, system_prompt)
    message_text, files = _message_fields(message)
    if config["use_prompt"] and message_text:
        prompt += message_text

    if not files:
        return

    completed: list[tuple[str, str]] = []
    last_payload: str | None = None

    for file in files:
        audio_path = _field(file, "path")
        if not audio_path:
            if isinstance(file, str):
                audio_path = file
            if not audio_path:
                continue

        filename = os.path.basename(audio_path)
        builder = _TranscriptBuilder(model_key)

        for update in _stream_single_file(client, config, prompt, audio_path, builder):
            if not update.text:
                continue
            current_blocks = completed + [(filename, update.text)]
            payload = _assemble_transcript(current_blocks)
            if payload != last_payload:
                last_payload = payload
                yield payload
            if update.is_final:
                completed.append((filename, update.text))
                if config["use_prompt"] and update.prompt_append:
                    prompt = prompt + f"\n{update.prompt_append}"
                break
        else:
            final_text = builder.formatted_text()
            if final_text:
                completed.append((filename, final_text))
                payload = _assemble_transcript(completed)
                if payload != last_payload:
                    yield payload
                if config["use_prompt"]:
                    prompt = prompt + f"\n{final_text}"


def _stream_single_file(
    client: Any,
    config: dict[str, Any],
    prompt: str,
    audio_path: str,
    builder: "_TranscriptBuilder",
) -> Iterator[TranscriptionUpdate]:
    request_kwargs = {
        "model": config["api_model"],
        "response_format": config["response_format"],
    }
    if config["use_prompt"]:
        request_kwargs["prompt"] = prompt
    if config["chunking_strategy"]:
        request_kwargs["chunking_strategy"] = config["chunking_strategy"]

    with open(audio_path, "rb") as fh:
        request_kwargs["file"] = fh
        response = client.audio.transcriptions.create(
            stream=config["supports_stream"], **request_kwargs
        )

    if config["supports_stream"]:
        yield from builder.consume_iter(response)
    else:
        yield builder.consume_snapshot(response)


def _assemble_transcript(blocks: Sequence[tuple[str, str]]) -> str:
    parts = []
    for filename, text in blocks:
        body = text.rstrip()
        parts.append(f"``` transcript {filename}\n{body}\n```")
    return "\n".join(parts)


def _build_prompt(history: Iterable[Any], system_prompt: str) -> str:
    prompt = system_prompt or ""
    for msg in history or []:
        role = _field(msg, "role")
        if role not in ("user", "assistant"):
            continue
        content = _field(msg, "content")
        if isinstance(content, tuple) or content is None:
            continue
        prompt += f"\n{content}"
    return prompt


def _format_transcription_text(
    model_key: str, text: str | None, segments: Sequence[dict[str, Any]] | None
) -> str:
    if model_key == "whisper":
        return text or ""

    if model_key == "gpt-4o-transcribe-diarize":
        if segments:
            turns: list[str] = []
            prev_speaker: str | None = None
            for seg in segments:
                speaker = (seg.get("speaker") or "Speaker").strip()
                seg_text = (seg.get("text") or "").strip()
                if seg_text:
                    if speaker != prev_speaker or not turns:
                        turns.append(f"{speaker}: {seg_text}")
                    else:
                        turns[-1] = f"{turns[-1]} {seg_text}".strip()
                    prev_speaker = speaker
            if turns:
                return "\n\n".join(turns)
        return text or ""

    raise ValueError(f"Unhandled transcription model formatting: {model_key}")


class _TranscriptBuilder:
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self._text_chunks: list[str] = []
        self._segments: list[dict[str, Any]] = []
        self._segment_ids: set[Any] = set()
        self._final_text: str | None = None
        self._final_segments: list[dict[str, Any]] | None = None
        self._finalized = False

    def consume_iter(self, stream: Any) -> Iterator[TranscriptionUpdate]:
        if not hasattr(stream, "__iter__"):
            yield self.consume_snapshot(stream)
            return

        for event in stream:
            update = self._ingest(event, assume_final=False)
            if update is None:
                continue
            yield update

        if not self._finalized:
            yield self._finalize_update()

    def consume_snapshot(self, snapshot: Any) -> TranscriptionUpdate:
        update = self._ingest(snapshot, assume_final=True)
        if update is not None:
            return update
        return self._finalize_update()

    def formatted_text(self) -> str:
        text = self._final_text
        if text is None and self._text_chunks:
            text = "".join(self._text_chunks)
        segments = self._final_segments or self._segments
        return _format_transcription_text(self.model_key, text, segments)

    def _ingest(self, obj: Any, assume_final: bool) -> TranscriptionUpdate | None:
        data = _to_dict(obj)
        changed, is_final = self._apply_event(data, assume_final=assume_final)
        if not changed:
            return None
        formatted = self.formatted_text()
        append = formatted if (is_final or assume_final) and formatted else None
        return TranscriptionUpdate(text=formatted, is_final=is_final or assume_final, prompt_append=append)

    def _apply_event(self, data: dict[str, Any], assume_final: bool) -> tuple[bool, bool]:
        event_type = data.get("type")
        changed = False
        is_final = False

        if event_type == "transcript.text.delta":
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                self._text_chunks.append(delta)
                changed = True

        if event_type == "transcript.text.segment":
            segment_payload = data.get("segment") or data
            segment = _normalize_segment(segment_payload)
            seg_id = segment.get("id")
            if seg_id is None or seg_id not in self._segment_ids:
                if seg_id is not None:
                    self._segment_ids.add(seg_id)
                self._segments.append(segment)
                changed = True

        if event_type == "transcript.text.done":
            self._capture_final(data)
            changed = True
            is_final = True

        if not changed:
            text_value = data.get("text")
            segments_value = data.get("segments")
            if isinstance(text_value, str) and text_value:
                self._final_text = text_value
                changed = True
            if isinstance(segments_value, list) and segments_value:
                self._final_segments = [_normalize_segment(seg) for seg in segments_value]
                changed = True
            if changed:
                is_final = True

        if assume_final:
            is_final = True
        if is_final:
            self._finalized = True

        return changed, is_final

    def _capture_final(self, data: dict[str, Any]) -> None:
        text_value = data.get("text")
        if isinstance(text_value, str) and text_value:
            self._final_text = text_value
        segments_value = data.get("segments")
        if isinstance(segments_value, list) and segments_value:
            self._final_segments = [_normalize_segment(seg) for seg in segments_value]

    def _finalize_update(self) -> TranscriptionUpdate:
        formatted = self.formatted_text()
        self._finalized = True
        append = formatted if formatted else None
        return TranscriptionUpdate(text=formatted, is_final=True, prompt_append=append)


def _normalize_segment(segment: Any) -> dict[str, Any]:
    data = _to_dict(segment)
    speaker = data.get("speaker")
    if isinstance(speaker, str):
        data["speaker"] = speaker.strip()
    text = data.get("text")
    if isinstance(text, str):
        data["text"] = text.strip()
    return data


def _message_fields(message: Any) -> tuple[str | None, Any]:
    return _field(message, "text"), _field(message, "files")


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        return {"text": obj}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}
