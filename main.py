import asyncio, eval
from colorama import init, Fore
from gemini import GeminiAPI
from ai2_thor_routines import AI2_Thor
init(autoreset=True)

# Init
robot = AI2_Thor()
vlm = GeminiAPI(
    model="gemini-2.0-flash",
    #model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=8192
)
initial_distance_agent_obj = 5

# Main: executes robot sim and vlm as parallel processes
async def main():
    await asyncio.gather(
        robot.sim_loop(),
        vlm.chat_loop(robot),
        #eval.eqa(robot, vlm, initial_distance_agent_obj)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass