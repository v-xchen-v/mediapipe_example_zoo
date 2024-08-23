"""
Refs:
    - https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
    - https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

- difference between hand/pose_landmarks and hand/pose_world_landmarks:
    - pose_landmarks:
        A list of pose landmarks. Each landmark consists of the following:

            x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
            z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
            visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

    - pose_world_landmarks
        Another list of pose landmarks in world coordinates. Each landmark consists of the following:

            x, y and z: Real-world 3D coordinates in meters with the origin at the center between hips.
            visibility: Identical to that defined in the corresponding pose_landmarks.
"""
import numpy as np
import numpy.typing as npt

from typing import List, Optional, Tuple
from .data_structures.landmark import Landmark
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
from .mp_landmark_index import MP_POSE_LANDMARK_NAME2INDEX, ONESIDE_HAND_ARM_LANDMARK_NAMES, MP_HAND_LANDMARK_NAME2INDEX

def center(nested_array_list):
    a = np.array(nested_array_list)
    mean = np.mean(a, axis=0)
    return mean[0], mean[1], mean[2]  # x, y, z

VISIBLE_THRESHOLD =0.5

class MPKeyPointSolution:
    """Combine mediepipe human pose and left/right hand together."""
    def __init__(self, static_image_mode = False):
        self.static_image_mode = static_image_mode
        
        self.hands = mp_hands.Hands(
            static_image_mode = self.static_image_mode,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.pose = mp_pose.Pose(
                static_image_mode= self.static_image_mode,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0)
    
    def process(self, image: npt.ArrayLike, type="BOTH_SIDE") -> Optional[Tuple[List[Landmark], List[Landmark]]]:
        """ Process a image to find out a single body with at least one hand

        Args:
            image (np.array): numpy array image
            type (str, optional): "BOTH_SIDE", "LEFT_SIDE", "RIGHT_SIDE". Defaults to "BOTH_SIDE".

        Returns:
            List[Landmark]: return a list of 3d keypoints by the type of single body in the image.
            List[Landmark]: return a list of world 3d keypoints by the type of single body in the image.
        """
        image = cv2.flip(image, 1)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        
        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(image)
        pose_results = self.pose.process(image)
        image.flags.writeable = True
        
        vis_landmarks_in_win = False
        if vis_landmarks_in_win:
            # print(pose_results.pose_world_landmarks)
            annotate_image = image.copy()
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            # Draw the hand annotations on the image.
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotate_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                annotate_image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(annotate_image, 1))
            # cv2.waitKey(0)
        
        left_hand_world_landmarks, right_hand_world_landmarks = None, None
        left_hand_normaized_landmarks, right_hand_normalized_landmarks = None, None
        if hand_results.multi_hand_world_landmarks != None and pose_results.pose_world_landmarks != None:
            # Note that handedness is determined assuming the input image is mirrored, i.e., taken with a 
            # front-facing/selfie camera with images flipped horizontally. If it is not the case, please swap the 
            # handedness output in the application.
            # for handness, hand_landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_world_landmarks):
            for handness, hand_landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_world_landmarks):
                hand_type = handness.classification[0].label
                if hand_type == "Left":
                    left_hand_world_landmarks = hand_landmarks
                else:
                    right_hand_world_landmarks = hand_landmarks
                    h, w, c = image.shape
                    # origin = center(np.array([np.array([item.x, item.y, item.z]) for item in hand_landmarks.landmark]))
                # print(hand_type)

            
            for handness, hand_landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
                hand_type = handness.classification[0].label
                if hand_type == "Left":
                    left_hand_normaized_landmarks = hand_landmarks
                else:
                    right_hand_normalized_landmarks = hand_landmarks
            #     if hand_type=='Right':
            #         handColor=(255,0,0) # Blue
            #     if hand_type=='Left':
            #         handColor=(0,0,255) # Red
            #     for ind in [0,5,6,7,8]:
            #         cv2.circle(image, 
            #                 (int(hand_landmarks.landmark[ind].x*image.shape[1]), int(hand_landmarks.landmark[ind].y*image.shape[0]))
            #                 ,15,handColor,5)
            #         # cv2.circle(image, 
            #         #         (int(hand_landmarks.landmark[ind].x), int(hand_landmarks.landmark[ind].y))
            #         #         ,15,handColor,5)
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
        else:
            return None, None
        
        def format_landmark(landmark):
            return Landmark(landmark.x, landmark.y, landmark.z)
            
        if type == "RIGHT_SIDE":
            if right_hand_world_landmarks is None:
                return None, None
            # Got a array of landmarks if not visible set None
            arm_hand_uv_landmarks = []
            arm_hand_world_landmarks = []
            
            # first two are ['shoulder' 'elbow'] from body pose, followed by ['wrist', 'thumb_xxx', ...] from hand pose
            
            for name in ONESIDE_HAND_ARM_LANDMARK_NAMES[:2]: 
                # fill the 'shoulder', 'elbow' keypoint
                current_world_landmark = pose_results.pose_world_landmarks.landmark[MP_POSE_LANDMARK_NAME2INDEX[f'right_{name}']]
                if current_world_landmark.visibility < VISIBLE_THRESHOLD:
                    arm_hand_world_landmarks.append(None)
                    return None, None
                else:
                    arm_hand_world_landmarks.append(format_landmark(current_world_landmark))
                    
            for name in ONESIDE_HAND_ARM_LANDMARK_NAMES[:2]: 
                # fill the 'shoulder', 'elbow' normalized keypoint
                current_uv_landmark = pose_results.pose_landmarks.landmark[MP_POSE_LANDMARK_NAME2INDEX[f'right_{name}']]
                if current_uv_landmark.visibility < VISIBLE_THRESHOLD:
                    arm_hand_uv_landmarks.append(None)
                    return None, None
                else:
                    arm_hand_uv_landmarks.append(format_landmark(current_uv_landmark))
            
            
            # fill the keypoint and normalized keypoint from hand pose
            for name in ONESIDE_HAND_ARM_LANDMARK_NAMES[2:]:
                if right_hand_world_landmarks != None:
                    arm_hand_uv_landmarks.append(format_landmark(right_hand_normalized_landmarks.landmark[MP_HAND_LANDMARK_NAME2INDEX[name]]))
                else:
                    arm_hand_uv_landmarks.append(None)
                
            # fill the keypoint and normalized keypoint from hand pose
            for name in ONESIDE_HAND_ARM_LANDMARK_NAMES[2:]:
                # connect the hand and body in different coordinated(have same axes with different origin) with 
                # assuming that the wrist is the same position and x, y, z 
                
                if name == 'wrist':
                    pose_twist_world_landmark = pose_results.pose_world_landmarks.landmark[MP_POSE_LANDMARK_NAME2INDEX[f'right_{name}']]
                    hand_twist_world_landmark = right_hand_world_landmarks.landmark[MP_HAND_LANDMARK_NAME2INDEX[name]]
                    # dxyz of origin of hand and pose cooridate
                    twist_hand2pose_dxyz = [
                                            hand_twist_world_landmark.x - pose_twist_world_landmark.x, 
                                            hand_twist_world_landmark.y - pose_twist_world_landmark.y,
                                            hand_twist_world_landmark.z - pose_twist_world_landmark.z
                                        ]
                    
                if right_hand_world_landmarks != None:
                    formatted_landmark = format_landmark(right_hand_world_landmarks.landmark[MP_HAND_LANDMARK_NAME2INDEX[name]])
                    formatted_landmark.x -= twist_hand2pose_dxyz[0]
                    formatted_landmark.y -= twist_hand2pose_dxyz[1]
                    formatted_landmark.z -= twist_hand2pose_dxyz[2]
                    arm_hand_world_landmarks.append(formatted_landmark)
                else:
                    arm_hand_world_landmarks.append(None)
                    
            return arm_hand_world_landmarks, arm_hand_uv_landmarks
                
        else:
            raise NotImplementedError()