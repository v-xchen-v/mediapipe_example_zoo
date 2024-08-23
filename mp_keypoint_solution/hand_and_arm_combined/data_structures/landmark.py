# Ref: https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
from dataclasses import dataclass

@dataclass
class Landmark:
    """
    x:  The x coordinate. 
    y:  The y coordinate. 
    z:  The z coordinate. 
    # visibility: Landmark visibility. Should stay unset if not supported. Float score of whether landmark is visible or 
    # occluded by other objects.
    """
    x: float
    y: float
    z: float
    # visibility: float
    