import cv2
import sys
import os
import tensorflow as tf
import mediapipe as mp
from read_input_files import read_video


#obtain PoseNet model from tensorflow hub
# model = tf.saved_model.load('/Users/jamesschiavo/Downloads/Software Engineering Projects/fitness/movenet-tensorflow2-singlepose-lightning-v4/')
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



# Open the video file
video_path = "/Users/jamesschiavo/Downloads/Software Engineering Projects/fitness/workout_videos/bench_press/bench_press_19.mp4"

#read input video
read_video(mp_pose, mp_drawing, mp_drawing_styles, video_path)