import asyncio
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class AI2_Thor():
    def __init__(self):
        self.controller = Controller(
            # Agent & Scene type
            agentMode="locobot",
            visibilityDistance=1.5,
            scene="FloorPlan_Val1_1",   # Available rooms in RoboTHOR: FloorPlan_Train{1:12}_{1:5} or FloorPlan_Val{1:3]_{1:5}, such as FloorPlan_Train3_5 or FloorPlan_Val1_2

            # Step size and properties
            gridSize=0.25,
            movementGaussianSigma=0.005,
            rotateStepDegrees=90,
            rotateGaussianSigma=0.5,

            # Image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=False,

            # Camera properties (854 x 480)
            width=896,
            height=896,
            fieldOfView=90,

            # Headless setup
            #platform=CloudRendering
        )

        self.controller.step("Done")

    def get_position(self):
        """Returns the current position of the agent as x,y coordinates and yaw angle."""
        position = self.controller.last_event.metadata["agent"]["position"]
        rotation = self.controller.last_event.metadata["agent"]["rotation"]
        return {'x': float(position['x']), 'y': float(position['z']), 'yaw': float(rotation['y'])}

    def get_image(self):
        """Returns the current image as a numpy array."""
        return self.controller.last_event.frame

    def move_forward(self, distance):
        """Moves the agent forward of a certain distance."""
        self.controller.step(action="MoveAhead",moveMagnitude=distance)

    def move_backward(self, distance):
        """Moves the agent backward of a certain distance."""
        self.controller.step(action="MoveBack",moveMagnitude=distance)

    def rotate_right(self, angle):
        """Rotate the agent to the right of a certain angle."""
        self.controller.step(action="RotateRight",degrees=angle)

    def rotate_left(self, angle):
        """Rotate the agent to the left of a certain angle"""
        self.controller.step(action="RotateLeft",degrees=angle)

    async def sim_loop(self):
        """Continuously step through the simulation."""
        while True:
            await asyncio.sleep(0)