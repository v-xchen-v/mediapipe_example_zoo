# Ref: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py?ref=assemblyai.com

"""MediaPipe Single Arm Hand connections."""

ARM_CONNECTION = ((0, 1), (1, 2))

HAND_PALM_CONNECTIONS = ((2, 3), (2, 7), (11, 15), (15, 19), (7, 11), (2, 19))

HAND_THUMB_CONNECTIONS = ((3, 4), (4, 5), (5, 6))

HAND_INDEX_FINGER_CONNECTIONS = ((7, 8), (8, 9), (9, 10))

HAND_MIDDLE_FINGER_CONNECTIONS = ((11, 12), (12, 13), (13, 14))

HAND_RING_FINGER_CONNECTIONS = ((15, 16), (16, 17), (17, 18))

HAND_PINKY_FINGER_CONNECTIONS = ((19, 20), (20, 21), (21, 22))

SINGLE_ARM_HAND_CONNECTIONS = frozenset().union(*[
    ARM_CONNECTION,
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])