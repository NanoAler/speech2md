from __future__ import annotations

from speech2md.core.postprocess.llm_client import LLMClient
from speech2md.core.postprocess.prompts import get_prompt

CHUNK_SIZE = 2000
OVERLAP = 200


def format_text(text: str, llm: LLMClient, language: str = "", prompt_template: str = "") -> str:
    if not text.strip():
        return text

    prompt = get_prompt(language, prompt_template).format(text=text)
    return llm.complete(prompt)


def format_long_text(
    text: str,
    llm: LLMClient,
    language: str = "",
    prompt_template: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> str:
    if len(text) <= chunk_size:
        return format_text(text, llm, language, prompt_template)

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    formatted: list[str] = []
    for i, chunk in enumerate(chunks):
        result = format_text(chunk, llm, language, prompt_template)
        if i > 0 and formatted:
            prev = formatted[-1]
            common = _find_overlap(prev, result, window=overlap)
            formatted[-1] = prev[: -len(common)] if common else prev
        formatted.append(result)

    return "\n\n".join(formatted)


def _find_overlap(a: str, b: str, window: int = 200) -> str:
    a_end = a[-window:].strip()
    b_start = b[:window].strip()
    for i in range(min(len(a_end), len(b_start)), 0, -1):
        if a_end[-i:].strip() == b_start[:i].strip():
            return a_end[-i:]
    return ""
