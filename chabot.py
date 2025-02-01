#Use a pipelione as high-level helper
import torch
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import json 
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from read_input_files import process_video

print(torch.__version__)

def label_dataset(exercise_type, joint_positions, timestamps):
    #Flatten into a feature vector for each frame
    joint_positions_flattened = np.array([np.array(frame).flatten() for frame in joint_positions])
    
    #Normalize the data
    scaler = StandardScaler()
    joint_positions_normalized = scaler.fit_transform(joint_positions_flattened)
    
    
    #k=Start with 2 clusters representing good/bad form
    k=2
    
    kmeans = KMeans(n_clusters=k,random_state=42)
    labels = kmeans.fit_predict(joint_positions_normalized)
    
    #Add cluster labels to the data
    joint_positions_with_labels = np.columnstack([joint_positions_flattened, labels])
    
    #Analyze the clusters
    for i in range(k):
        cluster_data = joint_positions_flattened[labels == i]
        print(f"Cluster {i}: {len(cluster_data)} samples")
        print(f"Example data point: {cluster_data[0]}")
        
    psuedo_labels = ["Good form" if label==0 else "Bad form" for label in labels]
    labeled_data = [
        {"exercise": exercise_type,
         "joint_positions": joint_positions_flattened,
         "timestamps": timestamps,
         "label": psuedo_labels[i]} for i in range(len(joint_positions_flattened))
    ]
    return labeled_data
def create_dataset():
    exercises = {}
    base_path = Path(__file__).resolve().parent
    print(f"base_path: {base_path}")
    input_path = os.path.join(base_path, "training_input_videos")
    output_path = os.path.join(base_path, "training_output_videos")
    
    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return None

    os.makedirs(output_path, exist_ok=True)
    
    exercise_names = os.listdir(input_path)
    exercise_directory = [exercise_name for exercise_name in exercise_names if os.path.isdir(os.path.join(input_path, exercise_name))]
    print(f"exercise_directory: {exercise_directory}")

    for exercise_type in exercise_directory:
        exercise_input_dir = os.path.join(input_path, exercise_type)
        exercise_output_dir = os.path.join(output_path, exercise_type)
      
        os.makedirs(exercise_output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(exercise_input_dir) if f.endswith(('.mp4', '.MOV'))]
        for video_file in video_files:
            input_video_path = os.path.join(exercise_input_dir, video_file)
            output_video_path = os.path.join(exercise_output_dir, video_file)
            joint_positions, timestamps = process_video(input_video_path, output_video_path)
            labeled_data = label_dataset(exercise_type, joint_positions, timestamps)
            if exercise_type not in exercises:
                exercises[exercise_type] = labeled_data
            else:
                exercises[exercise_type].extend(labeled_data)
    # print(f"exercises: {exercises}")
    joint_positions_dataset = os.path.join(base_path, "labeled_dataset.txt")
    with open(joint_positions_dataset, "w") as f:
        json.dump(exercises, f, indent=4)
    print(f"Dataset saved to {joint_positions_dataset}")
    return joint_positions_dataset          

def load_model():
    base_path = Path(__file__).resolve().parent.parent
    model_name= "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    labeled_dataset = os.path.join(base_path, "labeled_dataset.text")
    if(os.path.exists(labeled_dataset) == False):
        labeled_dataset = create_dataset()
        
    #Load the JSON data from the file
    with open(labeled_dataset, "r") as f:
        dataset_content = json.load(f)
    
    input_text = json.dumps(dataset_content)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}")
    return model
  

model = load_model()