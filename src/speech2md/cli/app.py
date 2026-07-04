from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from speech2md.cli.commands import config_cmd
from speech2md.cli.commands.listen import listen_cmd
from speech2md.core.pipeline import run_file_job

app = typer.Typer(name="speech2md")
console = Console()


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., help="Path to audio file (mp3, wav, etc.)", exists=True),
    language: str = typer.Option("", "--lang", "-l", help="Language code (ru, en, etc.)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM post-processing"),
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        def on_progress(stage: str, percent: float) -> None:
            labels = {
                "init": "Initializing...",
                "stt": "Transcribing audio...",
                "llm": "Post-processing with LLM...",
                "export": "Exporting...",
                "done": "Done!",
                "error": "Error!",
            }
            task = next(iter(progress.tasks), None)
            if task is None:
                progress.add_task("Working...")
            elif stage in labels:
                progress.update(task.id, description=labels[stage])

        job = run_file_job(
            audio_path,
            language=language,
            use_llm=not no_llm,
            on_progress=on_progress,
        )

    if job.status.value == "completed":
        assert job.result is not None
        console.print("\n[green]✓ Transcription saved[/green]")
        console.print(f"  Duration: {job.result.duration:.1f}s")
        console.print(f"  Language: {job.result.language or 'detected'}")
        console.print(f"  Words:    ~{len(job.result.full_text.split())}")
    else:
        console.print(f"\n[red]✗ Failed: {job.error}[/red]")
        raise typer.Exit(code=1)


app.add_typer(config_cmd.app, name="config", help="View or change configuration")
app.command(name="listen")(listen_cmd)
