import asyncio, csv, random, math, os
import pandas as pd
from tqdm import tqdm
from google.genai import types
from ai2thor.util.metrics import get_shortest_path_to_object, path_distance, compute_single_spl
from colorama import Fore, Style, init
init(autoreset=True)

def extract_vln_data():
    object_task = []

    # Read the CSV file
    with open("data/vln.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question_entry = {
                "scene_id": row["scene_id"],
                "prompt": row["prompt"],
                "object_id": row["object_id"],
                "object_type": row["object_type"]
            }

            q_type = row["task_type"]
            if q_type == "object":
                object_task.append(question_entry)

    return object_task

def extract_eqa_questions():
    questions = []
    count_questions = []
    existence_questions = []
    preposition_questions = []

    # Read the CSV file
    with open("data/eqa.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question_entry = {
                "scene_id": row["scene_id"],
                "prompt": row["prompt"],
                "ground_truth": row["ground_truth"],
                "object_id": row["object_id"]
            }

            q_type = row["task_type"]
            if q_type == "color":
                questions.append(question_entry)
            elif q_type == "preposition":
                preposition_questions.append(question_entry)
            elif q_type == "existence":
                existence_questions.append(question_entry)
            elif q_type == "count":
                count_questions.append(question_entry)

    return questions, preposition_questions, existence_questions, count_questions

def compute_distance(event, objectId, position):
    """
    Computes the Euclidian distance between agent camera and target object
    or between position and target object if position is provided.

    :param event: event metadata
    :param objectId: object identification code
    :param position: tuple with x, y and z coordinates. Provide 'None' to compute distance between agent camera and target object
    :return: Euclidian distance between objects
    """
    for obj in event.metadata["objects"]:
        if obj["objectId"] == objectId:
            if position:
                obj_position = [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
                return math.dist(position, obj_position)
            else:
                return obj["distance"]

    raise ValueError(f"Object with ID '{objectId}' not found in event metadata.")

def compute_minimum_distance_from_obj(event, objectId, path):
    # Convert path from list[dict] to list[tuple]
    path_vec = []
    for position in path:
        path_vec.append([position["x"], position["y"], position["z"]])

    # Compute minimum distance between path positions and target object
    for obj in event.metadata["objects"]:
        if obj["objectId"] == objectId:
            min_distance = float("inf")

            for position in path_vec:
                distance = compute_distance(event, objectId, position)

                if distance < min_distance:
                    min_distance = distance

            return min_distance

    raise ValueError(f"Object with ID '{objectId}' not found in event metadata.")

def check_eqa_question(eqa_question, last_response):
    if eqa_question["ground_truth"] in last_response:
        return True
    else:
        return False

def check_visibility(event, objectId):
    for obj in event.metadata["objects"]:
        if obj["objectId"] == objectId:
            return obj["visible"]
    raise Exception("Object not found for visibility check")


async def vln(robot, model, initial_distance_agent_obj):
    print(Fore.CYAN + "\n\n------------------        Vision Language Navigation        ------------------\n" + Style.RESET_ALL)

    # Extract VLN data
    object_task_data = extract_vln_data()

    # Initialize dataframe for metrics
    task_types = ["object"]
    metrics = ["SR", "SPL", "dist_termination", "dist_delta", "dist_min"]
    results = pd.DataFrame(index=task_types, columns=metrics)

    # Set visibility distance of the agent
    robot.controller.reset(visibilityDistance=100)
    distance_threshold = 0.2

    # Process object task
    previous_scene = ""
    best_path_length = float("inf")
    acc_success = 0
    acc_spl = 0

    for task in tqdm(object_task_data, desc="Processing OBJECT tasks"):
        # Set the current scene if changed
        if task["scene_id"] != previous_scene:
            robot.controller.reset(scene=task["scene_id"])
            previous_scene = task["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Randomize agent position at fixed distance from the target object
        feasible_positions = robot.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        fixed_distance_positions = []
        for pos in feasible_positions:
            position_vec = [pos["x"], pos["y"], pos["z"]]
            if abs(compute_distance(init_event, task["object_id"], position_vec) - initial_distance_agent_obj) < distance_threshold:
                fixed_distance_positions.append(pos)

        try:
            initial_position = random.choice(fixed_distance_positions)
        except IndexError:
            raise RuntimeError(f"Feasible initial position not available, increment distance threshold.")
        robot.controller.step(action="Teleport", position=initial_position)

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=task["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Compute metrics information
        distance_from_obj = compute_distance(event, task["object_id"], None)
        path_length = path_distance(path)

        # Save distance information of shortest path
        if path_length < best_path_length:
            dist_termination = distance_from_obj
            initial_position_vec = [initial_position["x"], initial_position["y"], initial_position["z"]]
            dist_delta = compute_distance(event, task["object_id"], initial_position_vec) - dist_termination
            dist_min = compute_minimum_distance_from_obj(event, task["object_id"], path)

        # SR and SPL information
        if distance_from_obj < 1:
            acc_success += 1
            acc_spl = compute_single_spl(path,get_shortest_path_to_object(robot.controller, task["object_type"],initial_position),True)

    # Save metrics about object task
    results.loc["object","SR"] = acc_success / len(object_task_data)
    results.loc["object","SPL"] = acc_spl / len(object_task_data)
    results.loc["object","dist_termination"] = dist_termination
    results.loc["object","dist_delta"] = dist_delta
    results.loc["object","dist_min"] = dist_min

    # Show and save overall results as csv file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(Fore.GREEN + "\n\n------------------------        VLN - OBJECT results        ------------------------\n")
    print(results)
    results.to_csv("results/vln_results.csv")

    # End session
    for task in asyncio.all_tasks():
        task.cancel()


async def eqa(robot, model, initial_distance_agent_obj):
    print(Fore.BLUE + "\n\n------------------        Embodied Question Answering        ------------------\n" + Style.RESET_ALL)

    # Extract EQA data
    color_questions, preposition_questions, existence_questions, count_questions = extract_eqa_questions()

    # Initialize dataframe for metrics
    question_types = ["color", "preposition", "existence", "count"]
    metrics = ["answer_accuracy", "vision_accuracy", "position_accuracy", "dist_termination", "dist_delta", "dist_min"]
    results = pd.DataFrame(index=question_types, columns=metrics)

    # Set visibility distance of the agent
    robot.controller.reset(visibilityDistance=100)
    distance_threshold = 0.2


    # Process color questions
    previous_scene = ""
    best_path_length = float("inf")
    correct_answers = 0
    wrong_but_seen_answers = 0
    wrong_but_on_goal_answers = 0

    for question in tqdm(color_questions, desc="Processing COLOR questions"):
        # Set the current scene if changed
        if question["scene_id"] != previous_scene:
            robot.controller.reset(scene=question["scene_id"])
            previous_scene = question["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Randomize agent position at fixed distance from the target object
        feasible_positions = robot.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        fixed_distance_positions = []
        for pos in feasible_positions:
            position_vec = [pos["x"], pos["y"], pos["z"]]
            if abs(compute_distance(init_event, question["object_id"], position_vec) - initial_distance_agent_obj) < distance_threshold:
                fixed_distance_positions.append(pos)

        try:
            initial_position = random.choice(fixed_distance_positions)
        except IndexError:
            raise RuntimeError(f"Feasible initial position not available, increment distance threshold.")
        robot.controller.step(action="Teleport", position=initial_position)

        # Reset model memory
        model.conversation_history = []

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=question["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Compute metrics information
        distance_from_obj = compute_distance(event, question["object_id"],None)
        path_length = path_distance(path)

        # Save distance information of shortest path
        if path_length < best_path_length:
            dist_termination = distance_from_obj
            initial_position_vec = [initial_position["x"], initial_position["y"], initial_position["z"]]
            dist_delta = compute_distance(event, question["object_id"], initial_position_vec) - dist_termination
            dist_min = compute_minimum_distance_from_obj(event, question["object_id"], path)

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1
        elif check_visibility(event, question["object_id"]):
            wrong_but_seen_answers +=1
        elif distance_from_obj < 1:
            wrong_but_on_goal_answers += 1

    # Save metrics about color questions
    results.loc["color", "answer_accuracy"] = correct_answers / len(color_questions)
    results.loc["color", "vision_accuracy"] = wrong_but_seen_answers / len(color_questions)
    results.loc["color", "position_accuracy"] = wrong_but_on_goal_answers / len(color_questions)
    results.loc["color", "dist_termination"] = dist_termination
    results.loc["color", "dist_delta"] = dist_delta
    results.loc["color", "dist_min"] = dist_min

    # Print color results
    print(Fore.GREEN + "\n\n-----------------------        EQA - COLOR  results        -----------------------\n")
    print(results.loc["color"])
    print("\n\n")


    # Process preposition questions
    previous_scene = ""
    best_path_length = float("inf")
    correct_answers = 0
    wrong_but_seen_answers = 0
    wrong_but_on_goal_answers = 0

    for question in tqdm(preposition_questions, desc="Processing PREPOSITION questions"):
        # Set the current scene if changed
        if question["scene_id"] != previous_scene:
            robot.controller.reset(scene=question["scene_id"])
            previous_scene = question["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Randomize agent position at fixed distance from the target object
        feasible_positions = robot.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        fixed_distance_positions = []
        for pos in feasible_positions:
            position_vec = [pos["x"], pos["y"], pos["z"]]
            if abs(compute_distance(init_event, question["object_id"], position_vec) - initial_distance_agent_obj) < distance_threshold:
                fixed_distance_positions.append(pos)

        try:
            initial_position = random.choice(fixed_distance_positions)
        except IndexError:
            raise RuntimeError(f"Feasible initial position not available, increment distance threshold.")
        robot.controller.step(action="Teleport", position=initial_position)
        robot.controller.step(action="Done")

        # Reset model memory
        model.conversation_history = []

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=question["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Compute metrics information
        distance_from_obj = compute_distance(event, question["object_id"], None)
        path_length = path_distance(path)

        # Save distance information for shortest path
        if path_length < best_path_length:
            dist_termination = distance_from_obj
            initial_position_vec = [initial_position["x"], initial_position["y"], initial_position["z"]]
            dist_delta = compute_distance(event, question["object_id"], initial_position_vec) - dist_termination
            dist_min = compute_minimum_distance_from_obj(event, question["object_id"], path)

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1
        elif check_visibility(event, question["object_id"]):
            wrong_but_seen_answers += 1
        elif distance_from_obj < 1:
            wrong_but_on_goal_answers += 1

    # Save metrics about preposition questions
    results.loc["preposition", "answer_accuracy"] = correct_answers / len(preposition_questions)
    results.loc["preposition", "vision_accuracy"] = wrong_but_seen_answers / len(preposition_questions)
    results.loc["preposition", "position_accuracy"] = wrong_but_on_goal_answers / len(preposition_questions)
    results.loc["preposition", "dist_termination"] = dist_termination
    results.loc["preposition", "dist_delta"] = dist_delta
    results.loc["preposition", "dist_min"] = dist_min

    # Print preposition results
    print(Fore.GREEN + "\n\n---------------------        EQA - PREPOSITION  results        ---------------------\n")
    print(results.loc["preposition"])
    print("\n\n")


    # Process existence questions
    previous_scene = ""
    correct_answers = 0

    for question in tqdm(existence_questions, desc="Processing EXISTENCE questions"):
        # Set the current scene if changed
        if question["scene_id"] != previous_scene:
            robot.controller.reset(scene=question["scene_id"])
            previous_scene = question["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Randomize agent position
        feasible_positions = robot.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        initial_position = random.choice(feasible_positions)
        robot.controller.step(action="Teleport", position=initial_position)

        # Reset model memory
        model.conversation_history = []

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=question["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1

    # Save metrics about existence questions
    results.loc["existence", "answer_accuracy"] = correct_answers / len(existence_questions)


    # Process count questions
    previous_scene = ""
    correct_answers = 0

    for question in tqdm(count_questions, desc="Processing COUNT questions"):
        # Set the current scene if changed
        if question["scene_id"] != previous_scene:
            robot.controller.reset(scene=question["scene_id"])
            previous_scene = question["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Randomize agent position
        feasible_positions = robot.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        initial_position = random.choice(feasible_positions)
        robot.controller.step(action="Teleport", position=initial_position)

        # Reset model memory
        model.conversation_history = []

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=question["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1

    # Save metrics about count questions
    results.loc["count", "answer_accuracy"] = correct_answers / len(count_questions)


    # Show and save overall results as csv file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(Fore.GREEN + "\n\n-------------------------        EQA  results        -------------------------\n")
    print(results)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/eqa_results.csv")

    # End session
    for task in asyncio.all_tasks():
        task.cancel()