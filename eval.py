import asyncio, csv, random, math, os, ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from google.genai import types
from ai2thor.util.metrics import path_distance, compute_single_spl, vector_distance
from thortils.navigation import get_shortest_path_to_object
from colorama import Fore, Style, init
init(autoreset=True)

def extract_vln_data():
    route_task = []

    # Read the CSV files
    with open("data/vln_route.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question_entry = {
                "scene_id": row["scene_id"],
                "prompt": row["prompt"],
                "init_position": ast.literal_eval(row["init_position"]),
                "init_orientation": ast.literal_eval(row["init_orientation"]),
                "final_position": ast.literal_eval(row["final_position"]),
            }

            q_type = row["task_type"]
            if q_type == "route":
                route_task.append(question_entry)

    return route_task

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

def get_shortest_path_to_point(controller, initial_position, target_position, allowed_error=0.0001):
    event = controller.step(
        action="GetShortestPathToPoint",
        position=initial_position,
        target=target_position,
        allowedError=allowed_error,
    )

    if not event.metadata["lastActionSuccess"]:
        raise ValueError(f"Failed to get path: {event.metadata['errorMessage']}")

    return event.metadata["actionReturn"]["corners"]

def compute_minimum_distance_from_pos(controller, position, path):
    minimum_distance = float("inf")

    for path_pos in path:
        distance = path_distance(get_shortest_path_to_point(controller, path_pos, position))

        if distance < minimum_distance:
            minimum_distance = distance

    return minimum_distance

def compute_minimum_distance_from_obj(event, controller, objectId, path):
    # Compute minimum distance between path positions and target object
    for obj in event.metadata["objects"]:
        if obj["objectId"] == objectId:
            min_distance = float("inf")

            for position in path:
                try:
                    distance = path_distance(get_shortest_path_to_object(
                        controller,
                        objectId,
                        position,
                        (0.0, 0.0, 0.0)
                    ))
                    if distance < min_distance:
                        min_distance = distance
                except Exception as e:
                    continue

            return min_distance if min_distance != float("inf") else None

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
    # Extract R2R data
    route_task_data = extract_vln_data()

    # Initialize dataframe for metrics
    task_types = ["route"]
    metrics = ["SR", "SPL", "dist_termination", "dist_delta", "dist_min"]
    results = pd.DataFrame(index=task_types, columns=metrics)

    # Set visibility distance of the agent
    robot.controller.reset(visibilityDistance=100)

    # Process R2R task
    previous_scene = ""
    acc_success = 0
    acc_spl = 0
    dist_termination = []
    dist_delta = []
    dist_min = []

    for task in tqdm(route_task_data, desc="Processing ROUTE tasks"):
        # Set the current scene if changed
        if task["scene_id"] != previous_scene:
            robot.controller.reset(scene=task["scene_id"])
            previous_scene = task["scene_id"]

            # Extract objects metadata
            init_event = robot.controller.last_event

        # Initialize initial position of the agent
        robot.controller.step(action="Done")
        try:
            robot.controller.step(action="Teleport", position=task["init_position"], rotation=task["init_orientation"])
        except IndexError:
            raise RuntimeError(f"Feasible initial position not available.")

        # Provide the question to the model
        model.conversation_history.append(types.Content(role="user", parts=[types.Part(text=task["prompt"])]))

        # Call Gemini in non-blocking way
        last_response, event, path = await asyncio.to_thread(model.chat_no_prints, robot)

        # Compute metrics information
        distance_from_final_pos_euclidian = vector_distance(event.metadata["agent"]["position"], task["final_position"])

        try:
            shortest_path_start = get_shortest_path_to_point(robot.controller, task["init_position"],
                                                             task["final_position"])
            distance_from_final_pos_at_start = path_distance(shortest_path_start)

            shortest_path_termination = get_shortest_path_to_point(robot.controller,
                                                                   event.metadata["agent"]["position"],
                                                                   task["final_position"])
            distance_from_final_pos_at_termination = path_distance(shortest_path_termination)

            dist_termination.append(distance_from_final_pos_at_termination)
            dist_delta.append(distance_from_final_pos_at_start - distance_from_final_pos_at_termination)
            dist_min.append(compute_minimum_distance_from_pos(robot.controller, task["final_position"], path))
        except Exception:
            continue

        # SR and SPL information
        try:
            if distance_from_final_pos_euclidian < 2:
                acc_success += 1
                acc_spl = compute_single_spl(path, get_shortest_path_to_point(controller=robot.controller,
                                                                              initial_position=task["init_position"],
                                                                              target_position=task["final_position"]),
                                             True)
        except Exception:
            continue

    # Save metrics about object task
    results.loc["route", "SR"] = acc_success / len(route_task_data)
    results.loc["route", "SPL"] = acc_spl / len(route_task_data)
    results.loc["route", "dist_termination"] = np.mean(dist_termination)
    results.loc["route", "dist_delta"] = np.mean(dist_delta)
    results.loc["route", "dist_min"] = np.mean(dist_min)

    # Show and save overall results as csv file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(Fore.GREEN + "\n\n------------------------        R2R results        ------------------------\n")
    print(results)
    print("\n")

    # End session
    for task in asyncio.all_tasks():
        task.cancel()


async def eqa(robot, model, initial_distance_agent_obj):
    # Extract EQA data
    color_questions, preposition_questions, existence_questions, count_questions = extract_eqa_questions()

    # Initialize dataframe for metrics
    question_types = ["color", "preposition", "existence", "count"]
    metrics = ["answer_accuracy", "vision_accuracy", "dist_termination", "dist_delta", "dist_min"]
    results = pd.DataFrame(index=question_types, columns=metrics)

    # Set visibility distance of the agent
    robot.controller.reset(visibilityDistance=100)
    distance_threshold = 0.7

    # Process color questions
    previous_scene = ""
    correct_answers = 0
    wrong_but_seen_answers = 0
    dist_termination = []
    dist_delta = []
    dist_min = []

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
            if abs(compute_distance(init_event, question["object_id"],
                                    position_vec) - initial_distance_agent_obj) < distance_threshold:
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
        try:
            distance_from_obj_at_start = path_distance(
                get_shortest_path_to_object(robot.controller, question["object_id"],
                                            initial_position, (0.0, 0.0, 0.0)))
            distance_from_obj_at_termination = path_distance(
                get_shortest_path_to_object(robot.controller, question["object_id"],
                                            event.metadata["agent"]["position"], (0.0, 0.0, 0.0)))
            min_distance_from_obj_along_path = compute_minimum_distance_from_obj(event, robot.controller,
                                                                                 question["object_id"], path)

            dist_termination.append(distance_from_obj_at_termination)
            dist_delta.append(distance_from_obj_at_start - distance_from_obj_at_termination)
            if min_distance_from_obj_along_path is not None:
                dist_min.append(min_distance_from_obj_along_path)
        except Exception:
            continue

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1
        elif check_visibility(event, question["object_id"]):
            wrong_but_seen_answers += 1

    # Save metrics about color questions
    results.loc["color", "answer_accuracy"] = correct_answers / len(color_questions)
    results.loc["color", "vision_accuracy"] = wrong_but_seen_answers / len(color_questions)
    results.loc["color", "dist_termination"] = np.mean(dist_termination)
    results.loc["color", "dist_delta"] = np.mean(dist_delta)
    results.loc["color", "dist_min"] = np.mean(dist_min)

    # Print color results
    print(Fore.GREEN + "\n\n-----------------------        EQA - COLOR  results        -----------------------\n")
    print(results.loc["color"])
    print("\n\n")

    # Process preposition questions
    previous_scene = ""
    correct_answers = 0
    wrong_but_seen_answers = 0
    dist_termination = []
    dist_delta = []
    dist_min = []

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
            if abs(compute_distance(init_event, question["object_id"],
                                    position_vec) - initial_distance_agent_obj) < distance_threshold:
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
        try:
            distance_from_obj_at_start = path_distance(
                get_shortest_path_to_object(robot.controller, question["object_id"],
                                            initial_position, (0.0, 0.0, 0.0)))
            distance_from_obj_at_termination = path_distance(
                get_shortest_path_to_object(robot.controller, question["object_id"],
                                            event.metadata["agent"]["position"], (0.0, 0.0, 0.0)))
            min_distance_from_obj_along_path = compute_minimum_distance_from_obj(event, robot.controller,
                                                                                 question["object_id"], path)

            dist_termination.append(distance_from_obj_at_termination)
            dist_delta.append(distance_from_obj_at_start - distance_from_obj_at_termination)
            if min_distance_from_obj_along_path is not None:
                dist_min.append(min_distance_from_obj_along_path)
        except Exception:
            continue

        # Accuracy information
        if check_eqa_question(question, last_response):
            correct_answers += 1
        elif check_visibility(event, question["object_id"]):
            wrong_but_seen_answers += 1

    # Save metrics about preposition questions
    results.loc["preposition", "answer_accuracy"] = correct_answers / len(color_questions)
    results.loc["preposition", "vision_accuracy"] = wrong_but_seen_answers / len(color_questions)
    results.loc["preposition", "dist_termination"] = np.mean(dist_termination)
    results.loc["preposition", "dist_delta"] = np.mean(dist_delta)
    results.loc["preposition", "dist_min"] = np.mean(dist_min)

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
    print("\n")
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/eqa_results.csv")

    # End session
    for task in asyncio.all_tasks():
        task.cancel()