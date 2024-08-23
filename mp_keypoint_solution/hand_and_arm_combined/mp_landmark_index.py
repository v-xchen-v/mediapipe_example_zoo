# Ref: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
import enum

# 21 hand landmarks
MP_HAND_LANDMARK_NAME2INDEX = {
    'wrist': 0, 
    'thumb_cmc':1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4, 
    'index_finger_mcp': 5, 'index_finger_pip': 6, 'index_finger_dip': 7, 'index_finger_tip': 8,
    'middle_finger_mcp': 9, 'middle_finger_pip': 10, 'middle_finger_dip': 11, 'middle_finger_tip': 12,
    'ring_finger_mcp': 13, 'ring_finger_pip': 14, 'ring_finger_dip': 15, 'ring_finger_tip': 16,
    'pinky_finger_mcp': 17, 'pinky_finger_pip': 18, 'pinky_finger_dip': 19, 'pinky_finger_tip': 20,
}
MP_HAND_LANDMARK_NAMES = list(MP_HAND_LANDMARK_NAME2INDEX.keys())
MP_HAND_LANDMARK_INDEX2NAME = {v: k for k, v in MP_HAND_LANDMARK_NAME2INDEX.items()}

# Ref: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# 33 body landmarks
# MP_POSE_LANDMARK_NAME2INDEX = {
#     "nose": 0,
#     "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
#     "right_eye_inner": 4, "right_eye": 5, "right eye_outer": 6,
#     "left_ear": 7,
#     "right_eye": 8,
#     "mouth_left": 9, "mouth_right": 10,
#     "left_shoulder": 11, "right_shoulder": 12,
#     "left_elbow": 13, "right_elbow": 14,
#     "left_wrist": 15, "right_wrist": 16,
#     "left_pinky": 17, "right_pinky": 18,
#     "left_index": 18, "right_index": 20,
#     "left_thumb": 21, "right_thumb": 22,
#     "left_hip": 23, "right_hip": 24,
#     "left_knee": 25, "right_knee": 26,
#     "left_ankle": 27, "right_ankle": 28,
#     "left_heel": 29, "right_heel": 30,
#     "left_foot_index": 31, "right_foot_index": 32
# }
# 33 body landmarks of fliped image
MP_POSE_LANDMARK_NAME2INDEX = {
    "nose": 0,
    "right_eye_inner": 1, "right_eye": 2, "right_eye_outer": 3,
    "left_eye_inner": 4, "left_eye": 5, "left_eye_outer": 6,
    "right_ear": 7,
    "left_eye": 8,
    "mouth_right": 9, "mouth_left": 10,
    "right_shoulder": 11, "left_shoulder": 12,
    "right_elbow": 13, "left__elbow": 14,
    "right_wrist": 15, "left__wrist": 16,
    "right_pinky": 17, "left_pinky": 18,
    "right_index": 18, "left_index": 20,
    "right_thumb": 21, "left_thumb": 22,
    "right_hip": 23, "left_hip": 24,
    "right_knee": 25, "left_knee": 26,
    "right_ankle": 27, "left_ankle": 28,
    "right_heel": 29, "left_heel": 30,
    "right_foot_index": 31, "left_foot_index": 32
}

MP_POSE_LANDMARK_NAMES = list(MP_POSE_LANDMARK_NAME2INDEX.keys())
MP_POSE_LANDMARK_INDEX2NAME = {v: k for k, v in MP_POSE_LANDMARK_NAME2INDEX.items()}

# 2x23 for two arm with hand
HAND_ARM_LANDMARK_NAME2INDEX = {
    "left_shoulder": 0, 
    "left_elbow": 1,    
    "left_wrist": 2,    
    "left_thumb_cmc":3, "left_thumb_mcp": 4, "left_thumb_ip": 5, "left_thumb_tip": 6, 
    "left_sindex_finger_mcp": 7, "left_index_finger_pip": 8, "left_index_finger_dip": 9, "left_index_finger_tip": 10,
    "left_middle_finger_mcp": 11, "left_middle_finger_pip": 12, "left_middle_finger_dip": 13, "left_middle_finger_tip": 14,
    "left_ring_finger_mcp": 15, "left_ring_finger_pip": 16, "left_ring_finger_dip": 17, "left_ring_finger_tip": 18,
    "left_pinky_finger_mcp": 19, "left_pinky_finger_pip": 20, "left_pinky_finger_dip": 21, "left_pinky_finger_tip": 22,
    "right_shoulder": 23,
    "right_elbow": 24,
    "right_write": 25,
    "right_thumb_cmc":26, "right_thumb_mcp": 27, "right_thumb_ip": 28, "right_thumb_tip": 29, 
    "right_sindex_finger_mcp": 30, "right_index_finger_pip": 31, "right_index_finger_dip": 32, "right_index_finger_tip": 33,
    "right_middle_finger_mcp": 34, "right_middle_finger_pip": 35, "right_middle_finger_dip": 36, "right_middle_finger_tip": 37,
    "right_ring_finger_mcp": 38, "right_ring_finger_pip": 39, "right_ring_finger_dip": 40, "right_ring_finger_tip": 41,
    "right_pinky_finger_mcp": 42, "right_pinky_finger_pip": 43, "right_pinky_finger_dip": 44, "right_pinky_finger_tip": 45,
}
HAND_ARM_LANDMARK_NAMES = list(HAND_ARM_LANDMARK_NAME2INDEX.keys())
HAND_ARM_LANDMARK_INDEX2NAME = {v: k for k, v in HAND_ARM_LANDMARK_NAME2INDEX.items()}

SINGLE_HAND_ARM_LANDMARK_NAME2INDEX = {
    "shoulder": 0,
    "elbow": 1,
    "wrist": 2,
    "thumb_cmc":3, "thumb_mcp": 4, "thumb_ip": 5, "thumb_tip": 6, 
    "index_finger_mcp": 7, "index_finger_pip": 8, "index_finger_dip": 9, "index_finger_tip": 10,
    "middle_finger_mcp": 11, "middle_finger_pip": 12, "middle_finger_dip": 13, "middle_finger_tip": 14,
    "ring_finger_mcp": 15, "ring_finger_pip": 16, "ring_finger_dip": 17, "ring_finger_tip": 18,
    "pinky_finger_mcp": 19, "pinky_finger_pip": 20, "pinky_finger_dip": 21, "pinky_finger_tip": 22,
}

class SINGLEHANDARM_LANDMARK(enum.IntEnum):
  """The 23 single arm and hand landmarks."""
  SHOULDER = 0
  ELBOW = 1
  WRIST = 2
  THUMB_CMC = 3
  THUMB_MCP = 4
  THUMB_IP = 5
  THUMB_TIP = 6
  INDEX_FINGER_MCP = 7
  INDEX_FINGER_PIP = 8
  INDEX_FINGER_DIP = 9
  INDEX_FINGER_TIP = 10
  MIDDLE_FINGER_MCP = 11
  MIDDLE_FINGER_PIP = 12
  MIDDLE_FINGER_DIP = 13
  MIDDLE_FINGER_TIP = 14
  RING_FINGER_MCP = 15
  RING_FINGER_PIP = 16
  RING_FINGER_DIP = 17
  RING_FINGER_TIP = 18
  PINKY_MCP = 19
  PINKY_PIP = 20
  PINKY_DIP = 21
  PINKY_TIP = 22
  
ONESIDE_HAND_ARM_LANDMARK_NAMES = list(SINGLE_HAND_ARM_LANDMARK_NAME2INDEX.keys())
SINGLE_HAND_ARM_LANDMARK_INDEX2NAME = {v: k for k, v in SINGLE_HAND_ARM_LANDMARK_NAME2INDEX.items()}