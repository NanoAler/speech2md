from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from speech2md.core.config import get_output_dir
from speech2md.core.config import load as load_config
from speech2md.core.export import write_markdown
from speech2md.core.models import Job, JobMode, JobStatus
from speech2md.core.postprocess.formatter import format_text
from speech2md.core.postprocess.llm_client import create_client
from speech2md.core.stt.whisper_engine import WhisperTranscriber

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


def run_file_job(
    audio_path: str | Path,
    *,
    language: str = "",
    use_llm: bool = True,
    on_progress: ProgressCallback | None = None,
) -> Job:
    settings = load_config()
    audio_path = Path(audio_path)

    if on_progress:
        on_progress("init", 0.0)

    job = Job(
        source=str(audio_path),
        mode=JobMode.file,
        language=language or settings.language,
        status=JobStatus.running,
    )

    try:
        if on_progress:
            on_progress("stt", 0.1)

        transcriber = WhisperTranscriber(
            model_size=settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        result = transcriber.transcribe(audio_path, language=job.language)

        if on_progress:
            on_progress("stt", 0.6)

        if use_llm:
            if on_progress:
                on_progress("llm", 0.7)

            try:
                llm = create_client(
                    backend=settings.llm_backend,
                    url=settings.llm_url,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    timeout=settings.llm_timeout,
                )
                prompt_template = (
                    settings.prompt_template_ru
                    if job.language == "ru"
                    else settings.prompt_template_en
                )
                formatted = format_text(result.full_text, llm, job.language, prompt_template)
                result.full_text = formatted
            except Exception as exc:
                logger.warning("LLM post-processing skipped: %s", exc)

        if on_progress:
            on_progress("export", 0.9)

        output_dir = get_output_dir(settings)
        output_path = output_dir / f"{audio_path.stem}.md"
        write_markdown(
            result,
            output_path,
            source=str(audio_path),
            language=result.language or job.language,
            model=settings.whisper_model,
        )

        job.result = result
        job.status = JobStatus.completed
        job.finished_at = __import__("datetime").datetime.now()

        if on_progress:
            on_progress("done", 1.0)

    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)
        if on_progress:
            on_progress("error", 0.0)

    return job
