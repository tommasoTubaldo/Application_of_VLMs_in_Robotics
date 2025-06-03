from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, OSXIntel64

controller = Controller(
            # Agent & Scene type
            agentMode="locobot",
            visibilityDistance=1.5,
            scene="FloorPlan_Train1_3",

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

print(controller.last_event.metadata["agent"]["rotation"])