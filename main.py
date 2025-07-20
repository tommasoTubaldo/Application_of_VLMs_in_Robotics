import asyncio, eval, utils
from colorama import init
from gemini import GeminiAPI
from ai2_thor_routines import AI2_Thor
init(autoreset=True)
from rich.console import Console

# Init
init(autoreset=True)
console = Console()
robot = AI2_Thor()
vlm = GeminiAPI(
    model="gemini-2.0-flash",
    temperature=0.0,
    max_tokens=8192
)
initial_distance_agent_obj = 4


async def main(mode):
    tasks = []
    tasks.append(robot.sim_loop())

    if mode == 1:
        tasks.append(vlm.chat_loop(robot))
    elif mode == 2:
        tasks.append(eval.vln(robot, vlm, initial_distance_agent_obj))
    elif mode == 3:
        tasks.append(eval.eqa(robot, vlm, initial_distance_agent_obj))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    utils.display_welcome(console)
    mode = utils.select_mode()

    try:
        asyncio.run(main(mode))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        console.print(f"Error: {str(e)}", style="bold red")
        console.print_exception(show_locals=True)