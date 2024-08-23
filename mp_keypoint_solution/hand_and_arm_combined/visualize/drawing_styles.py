# Ref: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py?ref=assemblyai.com

from typing import Mapping, Tuple
from .drawing_common import DrawingSpec
from ..mp_landmark_index import SINGLEHANDARM_LANDMARK
from . import single_hand_arm_connections

_RADIUS = 3

# Colors
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

# Hands
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1
_THICKNESS_ARM = 4

# Single hand arm landmarks
_ARM_LANDMARKS = (SINGLEHANDARM_LANDMARK.SHOULDER, SINGLEHANDARM_LANDMARK.ELBOW)
_PALM_LANDMARKS = (SINGLEHANDARM_LANDMARK.WRIST, SINGLEHANDARM_LANDMARK.THUMB_CMC,
                   SINGLEHANDARM_LANDMARK.INDEX_FINGER_MCP,
                   SINGLEHANDARM_LANDMARK.MIDDLE_FINGER_MCP, SINGLEHANDARM_LANDMARK.RING_FINGER_MCP,
                   SINGLEHANDARM_LANDMARK.PINKY_MCP)
_THUMP_LANDMARKS = (SINGLEHANDARM_LANDMARK.THUMB_MCP, SINGLEHANDARM_LANDMARK.THUMB_IP,
                    SINGLEHANDARM_LANDMARK.THUMB_TIP)
_INDEX_FINGER_LANDMARKS = (SINGLEHANDARM_LANDMARK.INDEX_FINGER_PIP,
                           SINGLEHANDARM_LANDMARK.INDEX_FINGER_DIP,
                           SINGLEHANDARM_LANDMARK.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (SINGLEHANDARM_LANDMARK.MIDDLE_FINGER_PIP,
                            SINGLEHANDARM_LANDMARK.MIDDLE_FINGER_DIP,
                            SINGLEHANDARM_LANDMARK.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (SINGLEHANDARM_LANDMARK.RING_FINGER_PIP,
                          SINGLEHANDARM_LANDMARK.RING_FINGER_DIP,
                          SINGLEHANDARM_LANDMARK.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (SINGLEHANDARM_LANDMARK.PINKY_PIP, SINGLEHANDARM_LANDMARK.PINKY_DIP,
                           SINGLEHANDARM_LANDMARK.PINKY_TIP)

_HAND_LANDMARK_STYLE = {
    _ARM_LANDMARKS: 
        DrawingSpec(
            color=_CYAN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PALM_LANDMARKS:
        DrawingSpec(
            color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Hands connections
_HAND_CONNECTION_STYLE = {
    single_hand_arm_connections.ARM_CONNECTION:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_ARM),
    single_hand_arm_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    single_hand_arm_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    single_hand_arm_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
    single_hand_arm_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
    single_hand_arm_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
    single_hand_arm_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER)
}

def get_default_hand_2d_landmarks_style() -> Mapping[int, DrawingSpec]:
  """Returns the default hand landmarks drawing style.

  Returns:
      A mapping from each hand landmark to its default drawing spec.
  """
  hand_landmark_style = {}
  for k, v in _HAND_LANDMARK_STYLE.items():
    for landmark in k:
      hand_landmark_style[landmark] = v
  return hand_landmark_style

def get_default_hand_2d_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default hand connections drawing style.

  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
  hand_connection_style = {}
  for k, v in _HAND_CONNECTION_STYLE.items():
    for connection in k:
      hand_connection_style[connection] = v
  return hand_connection_style