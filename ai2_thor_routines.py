import asyncio
from ai2thor.controller import Controller

class AI2_Thor():
    def __init__(self):
        self.controller = Controller(
            # Agent & Scene type
            agentMode="locobot",
            visibilityDistance=1.5,
            scene="FloorPlan_Val1_3",

            # Step size and properties
            gridSize=0.25,
            movementGaussianSigma=0.005,
            rotateStepDegrees=90,
            rotateGaussianSigma=0.5,

            # Image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=False,

            # Camera properties
            width=896,
            height=896,
            fieldOfView=90
        )

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