import os, time, io, asyncio, base64, textwrap, cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from colorama import Fore, Style, init
init(autoreset=True)

class GeminiAPI():
    def __init__(self, model, temperature:float = 0.7, max_tokens:int = 1e5, generate_with_tools:bool = True):
        self.model = model

        # Load system behavior instructions
        prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLMs_on_AI2-Thor/prompts'))
        with open(os.path.join(prompt_dir, "system_instruction_val.txt"), "r") as f:
            self.system_instruction = f.read()

        self.conversation_history = []

        # Function declarations for the model
        self.set_pos_function = {
            "name": "get_position",
            "description": "Returns the current position and orientation of the robot as a 3-element vector: [x, y, yaw], where x and y are the robot coordinates in meters and yaw is the orientation angle in radians.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }

        self.set_image_function = {
            "name": "get_image",
            "description": "Returns the latest image from the robot's onboard camera.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }

        self.set_move_forward_function = {
            "name": "move_forward",
            "description": "Moves the robot forward by a specified distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {
                        "type": "number",
                        "description": "The forward distance (in meters) the robot should travel. Only positive values are supported.",
                    },
                },
                "required": ["distance"],
            },
        }

        self.set_move_backward_function = {
            "name": "move_backward",
            "description": "Moves the robot backward by a specified distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {
                        "type": "number",
                        "description": "The backward distance (in meters) the robot should travel. Only positive values are supported.",
                    },
                },
                "required": ["distance"],
            },
        }

        self.set_rotate_right_function = {
            "name": "rotate_right",
            "description": "Rotates the robot to the right by a specified angle. The angle must be a positive value and is measured in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "The angle to rotate the robot, in degrees. Only positive values are allowed.",
                    },
                },
                "required": ["angle"],
            },
        }

        self.set_rotate_left_function = {
            "name": "rotate_left",
            "description": "Rotates the robot to the left by a specified angle. The angle must be a positive value and is measured in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "The angle to rotate the robot, in degrees. Only positive values are allowed.",
                    },
                },
                "required": ["angle"],
            },
        }

        self.set_response_completed = {
            "name": "response_completed",
            "description": "Execute this function when the robot response is completed or when asking for clarifications in order to let the user provide a new prompt or a response.",
            "parameters": {
              "type": "object",
              "properties": {}
            },
        }

        # Configure the model with system instructions and tools
        self.tools = types.Tool(function_declarations=[
            self.set_pos_function,
            self.set_image_function,
            self.set_move_forward_function,
            self.set_move_backward_function,
            self.set_rotate_right_function,
            self.set_rotate_left_function,
            self.set_response_completed
        ])

        # Configure function calling mode
        if generate_with_tools:
            mode = "AUTO"   # AUTO: The model decides whether to generate a natural language response or suggest a function call based on the prompt and context.
                            # ANY: The model is constrained to always predict a function call and guarantee function schema adherence.
                            #      If 'allowed_function_names' is not specified in tool_config, the model can choose from any of the provided function declarations.
                            #      If 'allowed_function_names' is provided as a list, the model can only choose from the functions in that list.
                            #      Use this mode when you require a function call in response to every prompt (if applicable)
        else:
            mode = "NONE"   # NONE: The model is prohibited from making function calls

        self.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode #, allowed_function_names=[]
            )
        )

        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            max_output_tokens=max_tokens,
            temperature=temperature,
            tools=[self.tools],
            tool_config=self.tool_config
        )

        # Configure the client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def image_to_base64(self, image: Union[np.ndarray, "np.uint8"]) -> str:
        """Converts an image in format numpy.uint8 in base64"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape [H, W, 3].")

        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise ValueError("Could not encode image to JPEG.")

        # Convert to base64 string
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str

    def display_base64_image(self, base64_string: str):
        """
        Decodes and displays a base64-encoded image.

        Parameters:
        - base64_string (str): The base64-encoded image string.
        """
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)

        # Convert bytes to a PIL Image
        image = Image.open(BytesIO(image_data))

        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()


    def chat(self, robot):
        """
        Generates a response using the current conversation history and tools.

        :param robot: Robot instance
        :return: Prediction from Gemini
        """
        contents = list(self.conversation_history)

        while True:
            # Binary exponential backoff parameters
            max_retries = 10
            retry_delay = 2  # seconds (initial delay)
            retries = 0

            # Handle 429 and 503 exceptions adopting a binary exponential backoff algorithm
            while True:
                try:
                    start_time = time.time()
                    response = self.client.models.generate_content(
                        model=self.model,
                        config=self.config,
                        contents=contents
                    )
                    end_time = time.time()
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if retries < max_retries:
                            print(Fore.RED + f"[429 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")

                    elif "503" in str(e) or "502" in str(e) or "UNAVAILABLE" in str(e):
                        if retries < max_retries:
                            print(Fore.RED + f"[503 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")
                    else:
                        raise e

            # Process response
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(Fore.BLUE + "\nAssistant:" + Style.RESET_ALL)
                    print(textwrap.fill(part.text, width=100))
                    print(Fore.YELLOW + f"[Inference time: {end_time - start_time:.2f} s]" + Style.RESET_ALL)

                    contents.append(types.Content(role="user", parts=[types.Part(text=part.text)]))

                if part.function_call:
                    tool_call = part.function_call

                    if tool_call.name == "get_position":
                        result = robot.get_position()
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} get_position()")

                        function_response_part = types.Part.from_function_response(name=tool_call.name,response={"result": result},)

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "get_image":
                        result = self.image_to_base64(robot.get_image())
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} get_image()")

                        function_response_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=result))

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "move_forward":
                        robot.move_forward(**tool_call.args)
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} move_forward({tool_call.args})")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has moved forward by {tool_call.args} meters")]))

                    elif tool_call.name == "move_backward":
                        robot.move_backward(**tool_call.args)
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} move_backward({tool_call.args})")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has moved backward by {tool_call.args} meters")]))

                    elif tool_call.name == "rotate_right":
                        robot.rotate_right(**tool_call.args)
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} rotate_right({tool_call.args})")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has rotate to the right by {tool_call.args} degrees")]))

                    elif tool_call.name == "rotate_left":
                        robot.rotate_left(**tool_call.args)
                        print(f"\n{Fore.MAGENTA}[Tool]{Style.RESET_ALL} rotate_left({tool_call.args})")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has has rotate to the left by {tool_call.args} degrees")]))

                    elif tool_call.name == "response_completed":
                        robot.controller.step("Done")
                        self.conversation_history = contents
                        return response


    def chat_no_prints(self, robot):
        """
        Generates a response using the current conversation history and tools.

        :param robot: Robot instance
        :return: Prediction from Gemini
        """
        contents = list(self.conversation_history)
        path = []

        while True:
            # Binary exponential backoff parameters
            max_retries = 10
            retry_delay = 2  # seconds (initial delay)
            retries = 0

            # Handle 429 and 503 exceptions adopting a binary exponential backoff algorithm
            while True:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        config=self.config,
                        contents=contents
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if retries < max_retries:
                            #print(Fore.RED + f"[429 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")

                    elif "503" in str(e) or "UNAVAILABLE" in str(e):
                        if retries < max_retries:
                            #print(Fore.RED + f"[503 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")
                    else:
                        raise e

            # Save agent position
            path.append(robot.controller.last_event.metadata["agent"]["position"])

            # Check if tool call is made
            for part in response.candidates[0].content.parts:
                if part.text:
                    contents.append(types.Content(role="user", parts=[types.Part(text=part.text)]))

                if part.function_call:
                    tool_call = part.function_call

                    if tool_call.name == "get_position":
                        result = robot.get_position()

                        function_response_part = types.Part.from_function_response(name=tool_call.name,response={"result": result},)

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "get_image":
                        result = self.image_to_base64(robot.get_image())

                        function_response_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=result))

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "move_forward":
                        robot.move_forward(**tool_call.args)

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has moved forward by {tool_call.args} meters")]))

                    elif tool_call.name == "move_backward":
                        robot.move_backward(**tool_call.args)

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has moved backward by {tool_call.args} meters")]))

                    elif tool_call.name == "rotate_right":
                        robot.rotate_right(**tool_call.args)

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has rotate to the right by {tool_call.args} degrees")]))

                    elif tool_call.name == "rotate_left":
                        robot.rotate_left(**tool_call.args)

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"The robot has has rotate to the left by {tool_call.args} degrees")]))

                    elif tool_call.name == "response_completed":
                        event = robot.controller.step("Done")
                        self.conversation_history = contents
                        return response, event, path


    async def chat_loop(self, robot):
        """Handle Gemini chat interaction in a single loop.
        """
        print(Fore.CYAN + "\n----------   TurtleBot3 VLM Chat Interface   ----------" + Style.RESET_ALL)
        print("Type 'exit' or 'quit' to end the session.")

        while True:
            user_input = await asyncio.to_thread(input, Fore.GREEN + "\n\nUser: " + Style.RESET_ALL)
            user_input = user_input.strip()

            # Exit condition
            if user_input.lower() in ["exit", "quit"]:
                print(Fore.CYAN + "\nSession ended.")

                if self.conversation_history:
                    self.conversation_history.append(types.Content(role="user", parts=[types.Part(text=self.system_instruction)]))

                    # Count input tokens from conversation history
                    total_tokens = self.client.models.count_tokens(
                        model=self.model, contents=self.conversation_history
                    )
                    print(Fore.YELLOW + f"[Total input tokens: {total_tokens.total_tokens}]")

                for task in asyncio.all_tasks():
                    task.cancel()
                break

            # Add to the conversation history the current user prompt
            self.conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

            # Call Gemini in non-blocking way
            await asyncio.to_thread(self.chat, robot)