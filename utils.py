import questionary
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

def display_welcome(console):
    # Title panel - clean and centered
    title_panel= Panel(
            Text("APPLICATION OF VISION LANGUAGE MODELS IN ROBOTICS", justify="center", style="bold"),
            box=box.DOUBLE,
            border_style="blue",
            width=console.width,
            padding=(1, 0)
        )

    # Mode panels with minimal styling
    mode_columns = Columns([
        Panel(
            Text("Chat Mode\n\nRuns a chat-like GUI to provide\nmanual prompts to the model", justify="center"),
            border_style="dim"
        ),
        Panel(
            Text("Evaluation - R2R\n\nRuns the Room-to-Room\nevaluation",
                 justify="center"),
            border_style="dim"
        ),
        Panel(
            Text("Evaluation - EQA\n\nRuns the Embodied Question\nAnswering evaluation", justify="center"),
            border_style="dim"
        )
    ], expand=True)

    console.print(title_panel)
    console.print("\n")
    console.print(Text("Available Modes:", style="bold", justify="center"))
    console.print(mode_columns)
    console.print("\n")


def select_mode():
    choice = questionary.select(
        "Select script mode:",
        choices=[
            "Chat Mode",
            "Evaluation - R2R",
            "Evaluation - EQA"
        ],
        default="Chat Mode"
    ).ask()

    # Return numerical value or the mode name
    modes = {
        "Chat Mode": 1,
        "Evaluation - R2R": 2,
        "Evaluation - EQA": 3
    }
    return modes.get(choice, 1)