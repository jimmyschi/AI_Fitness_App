import time
import os
import sys
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

#TODO: get body_mass from UI as float
def calculate_force_vector(joint_position, body_mass, timestamps):
    #body mass is in kg
    #Joint positions should be shape: (#frames, 3)
    #timestamps shape: (# frames, 1)
    #Need to calculate net direction = sqrt(x^2 + y^2 + z^2)
    
    # print("JOINT POSITION SHAPE: ", joint_position.shape)
    # print("JOINT_POSITION: ", joint_position)
    
    # Validate data
    if len(joint_position) < 2 or len(timestamps) < 2:
        raise ValueError("Not enough data points to calculate velocity or acceleration.")
    
    #velocity => m/s
    position_differences = np.diff(joint_position, axis=0)
    
    time_differences = np.diff(timestamps)[:, None]
    print("Time Differences Shape: ", time_differences.shape)
    print("Time Differences: ", time_differences)
    velocities = position_differences / time_differences
    print("Velocities Shape: ", velocities.shape)
    print("Velocities: ", velocities)
    #acceleration => m/s^2
    accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]
    print("Accelerations Shape: ", accelerations.shape)
    print("Accelerations: ", accelerations)
    segment_mass = body_mass * 0.03 #Eblow mass ~ 3% body mass
    weight_lifted = 60 #weight is in kg
    forces = segment_mass * accelerations
    magnitudes = np.linalg.norm(forces, axis=1)
    print("Magnitudes: ", magnitudes)
    angles = {
        "theta_x": np.arccos(forces[:, 0] / magnitudes),
        "theta_y": np.arccos(forces[:, 1] / magnitudes),
        "theta_z": np.arccos(forces[:, 2] / magnitudes),
    }
    return magnitudes, angles

def draw_force_vectors(frame, forces, joint_positions, scale_factor=1):
    #TODO: make sure width and height are correct
    #TODO: look into scale factor
    image_height, image_width, _ = frame.shape
    for force, position in zip(forces, joint_positions):
        start_point = (int(position[0] * image_width), int(position[1] * image_height))
        end_point = (
            int(start_point[0] + force[0] * scale_factor),
            int(start_point[1] - force[1] * scale_factor)
        )
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2) #green arrow for force vector
    

#TODO: get exercise as a string from UI
def plot_reps(forces, angles, timestamps, exercise):
    print("Forces Shape: ", forces.shape)
    print("Forces: ", forces)

    print("Angles: ", angles)

    # Detect peaks (highest points in eccentric phase) and troughs (lowest points in concentric phase)
    theta_y = angles['theta_y']
    #Applpy Guassian filtering to reduce noise
    smoothed_theta_y = gaussian_filter1d(theta_y, sigma=2)
    
    peaks, _ = find_peaks(smoothed_theta_y, prominence=0.2, distance=5)  # Peaks correspond to eccentric-to-concentric transitions
    troughs, _ = find_peaks(-smoothed_theta_y, prominence=0.2, distance=5)  # Troughs correspond to concentric-to-eccentric transitions

    print("Peaks (eccentric to concentric):", peaks)
    print("Troughs (concentric to eccentric):", troughs)

    # Ensure we alternate peaks and troughs correctly to count valid repetitions
    valid_reps = []
    eccentric_times = []
    concentric_times = []

    for i in range(len(troughs) - 1):
        peak_candidates = peaks[(peaks > troughs[i]) & (peaks < troughs[i + 1])]
        if len(peak_candidates) > 0:
            peak = peak_candidates[0]
            valid_reps.append((troughs[i], peak, troughs[i + 1]))
            eccentric_times.append(timestamps[peak] - timestamps[troughs[i]])
            concentric_times.append(timestamps[troughs[i + 1]] - timestamps[peak])

    print("Valid reps (trough -> peak -> trough):", valid_reps)
    print("Eccentric Times: ", eccentric_times)
    print("Concentric Times: ", concentric_times)

    # Plotting the reps with time (eccentric vs concentric)
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(eccentric_times) + 1)
    plt.bar(x - 0.2, eccentric_times, width=0.4, color='r', label='Eccentric time')
    plt.bar(x + 0.2, concentric_times, width=0.4, color='b', label='Concentric time')
    plt.xticks(x)
    plt.ylabel('Time (s)')
    plt.xlabel('Reps')
    plt.title(f'{exercise} - Eccentric vs Concentric Portions')
    plt.legend()
    plt.show()
    
    reps = len(eccentric_times)
    
    return concentric_times, eccentric_times

        
             
             