import typer
from rich.console import Console
from rich.table import Table

from speech2md.core.config import CONFIG_FILE
from speech2md.core.config import load as load_config
from speech2md.core.config import save as save_config

app = typer.Typer()
console = Console()


@app.command()
def show() -> None:
    settings = load_config()
    table = Table("Key", "Value")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def set(
    key: str = typer.Argument(..., help="Setting key"),
    value: str = typer.Argument(..., help="Setting value"),
) -> None:
    settings = load_config()
    if not hasattr(settings, key):
        console.print(f"[red]Unknown setting: {key}[/red]")
        raise typer.Exit(code=1)
    setattr(settings, key, value)
    save_config(settings)
    console.print(f"[green]{key}[/green] set to [yellow]{value}[/yellow]")
    console.print(f"Saved to {CONFIG_FILE}")


@app.command()
def path() -> None:
    console.print(str(CONFIG_FILE))
