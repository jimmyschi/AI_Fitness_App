import cv2
import numpy as np
import tensorflow as tf
import os
from time_under_tension import calculate_force_vector
from time_under_tension import draw_force_vectors
from time_under_tension import plot_reps


def read_video(mp_pose, mp_drawing, mp_drawing_styles, video_path):
    # if not os.path.exists(video_path):
    #     print(f"Error: File not found at {video_path}")
    #     exit()
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    # Check if the file is opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
        
    timestamps = []
    landmarks = []
    frame_count = 0
    positions_count = 0
    joint_idx = 14 #get from UI (ex for bench)
    
    joint_position = []
    
    pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence=.5, model_complexity=2)
        
    # Read and display frames
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:  # Break the loop if no more frames
            break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        
        if results.pose_landmarks:
            positions_count += 1
            joint_position.append(results.pose_landmarks.landmark[joint_idx])
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 #convert to seconds
            timestamps.append(timestamp)
            
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        cv2.imshow("Pose Estimation", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print("Frames: ", frame_count)
    print("POSITIONS COUNT: ", positions_count)
    
    timestamps = np.array(timestamps)
    joint_positions = np.array([[lm.x, lm.y, lm.z] for lm in joint_position])
    # forces, angles = calculate_force_vector(joint_position, 100, timestamps)

    # plot_reps(forces=forces, angles=angles, timestamps=timestamps, exercise="bench_press")
    return timestamps, joint_positions

        
        




