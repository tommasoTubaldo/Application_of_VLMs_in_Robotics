import json
import pandas as pd
from pandas import json_normalize
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, OSXIntel64

controller = Controller(
            # Agent & Scene type
            agentMode="locobot",
            visibilityDistance=1.5,
            scene="FloorPlan_Val1_1",

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

data = controller.last_event.metadata["objects"]

#print(controller.last_event.metadata["objects"])

# Flatten the structure
df = json_normalize(data, sep='_')

# Show all rows
pd.set_option('display.max_rows', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Prevent truncation of wide columns
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Show a few columns
print(df[['name', 'position_x','position_y','position_z', 'distance', 'visible', 'objectType', 'objectId']])