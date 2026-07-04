from __future__ import annotations

from datetime import datetime
from pathlib import Path

from speech2md.core.models import TranscriptionResult


def write_markdown(
    result: TranscriptionResult,
    output_path: str | Path,
    source: str = "",
    language: str = "",
    model: str = "",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = ["---"]
    parts.append(f"created: {datetime.now().isoformat()}")
    if source:
        parts.append(f"source: {source}")
    if language:
        parts.append(f"language: {language}")
    if model:
        parts.append(f"model: {model}")
    if result.duration:
        parts.append(f"duration_sec: {result.duration:.1f}")
    parts.append("---")
    parts.append("")
    parts.append(result.full_text)
    parts.append("")

    path.write_text("\n".join(parts), encoding="utf-8")
    return path
