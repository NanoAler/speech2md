from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from speech2md.core.models import TranscriptionResult


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str = "") -> TranscriptionResult:
        raise NotImplementedError
