import typer
from rich.console import Console

console = Console()


def listen_cmd(
    language: str = typer.Option("", "--lang", "-l", help="Language code (ru, en, etc.)"),
) -> None:
    console.print("[yellow]Live transcription not yet implemented.[/yellow]")
    console.print("This will stream from microphone to VAD to whisper to live output.")
    console.print("Coming in a future update.")
    raise typer.Exit(code=0)
