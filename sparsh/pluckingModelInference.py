import numpy as np
import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import pandas as pd
import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
import PIL
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError
import torchvision.transforms as T
import hydra
from omegaconf import OmegaConf, DictConfig
import csv
import joblib

from proprioLSTM import LSTMModel # Lstm proprio model
from lstmimages import LSTMImagePredictor # for vision only model
from lstmTactile import LSTMTactilePredictor # for tactile only model
from lstmTactileVision import LSTMTactileImagePredictor # For vision+tactile model

from proprioACT import ACTModel, input_indices, action_indices # Act proprio model
from ACTImages import ACTImagesModel # for vision only model
from actTactile import ACTTactileModel # for tactile only model
from ACTTactileVision import ActTactileImagesModel # For vision+tactile model

from proprioTransformer import TransformerModel #Transformr proprio model
from transformerImages import TransformerImagesModel # for vision only model
from transTactile import TransformerTactileModel # for tactile only model
from transTactileVision import TransformerTactileVisionModel # For vision+tactile model

import sys
sys.path.append('/home/bonggeeun/anaconda3/envs/rosallegro/bin/python')

target_image_size = 64

input_columns = ['x_pos', 'y_pos', 'z_pos', 'roll', 'pitch', 'yaw',
                'xarm_joint_1_position', 'xarm_joint_2_position',
                'xarm_joint_3_position', 'xarm_joint_4_position',
                'xarm_joint_5_position', 'xarm_joint_6_position',
                'xarm_joint_7_position', 'xarm_joint_1_torque',
                'xarm_joint_2_torque', 'xarm_joint_3_torque', 'xarm_joint_4_torque',
                'xarm_joint_5_torque', 'xarm_joint_6_torque', 'xarm_joint_7_torque',
                'thumb_1_position', 'thumb_2_position', 'thumb_3_position',
                'thumb_4_position', 'index_1_position', 'index_2_position',
                'index_3_position', 'index_4_position', 'thumb_1_velocity',
                'thumb_2_velocity', 'thumb_3_velocity', 'thumb_4_velocity',
                'index_1_velocity', 'index_2_velocity', 'index_3_velocity',
                'index_4_velocity', 'thumb_1_torque', 'thumb_2_torque', 'thumb_3_torque',
                'thumb_4_torque', 'index_1_torque', 'index_2_torque', 'index_3_torque',
                'index_4_torque']
output_columns = ["roll", "pitch", "yaw", "thumb_1_position", "thumb_2_position", "thumb_3_position",
                "thumb_4_position", "index_1_position", "index_2_position", "index_3_position",
                "index_4_position", "thumb_1_torque", "thumb_2_torque", "thumb_3_torque", 
                "thumb_4_torque", "index_1_torque", "index_2_torque", "index_3_torque", 
                "index_4_torque", "task_status"]
exclude_columns = [ "timestamp", "Stage", "Fx", "Fy", "Fz", "xarm_joint_1_velocity", "xarm_joint_2_velocity",
                    "xarm_joint_3_velocity", "xarm_joint_4_velocity", "xarm_joint_5_velocity", "xarm_joint_6_velocity",
                    'xarm_joint_7_velocity', "middle_1_position", "middle_2_position", "middle_3_position",
                    "middle_4_position", "middle_1_torque", "middle_2_torque", "middle_3_torque", "middle_4_torque",
                    'middle_1_velocity', 'middle_2_velocity', 'middle_3_velocity', 'middle_4_velocity', 'ring_1_velocity',
                    'ring_2_velocity', 'ring_3_velocity', 'ring_4_velocity', "ring_1_position", "ring_2_position",
                    "ring_3_position", "ring_4_position", "ring_1_torque", "ring_2_torque", "ring_3_torque", "ring_4_torque"
                ]
#Input scaler
input_scaler = joblib.load("/home/bonggeeun/Ishrath/sparsh/Model_scripts/input_successScaler.pkl")

# Initialize CvBridge to convert ROS images to OpenCV
bridge = CvBridge()

# Variables to store received messages
eof_color_images, eof_depth_images = [], []
world_color_images, world_depth_images = [], []
thumb_images, index_images = [], []
xArm_joint_angles, hand_joint_angles = [],[]

# ROS Publisher for publishing predictions
predictions_pub = rospy.Publisher('/predicted_states', Float32MultiArray, queue_size=10)

# Callback function for camera image
def eof_color_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = PIL.Image.fromarray(np.uint8(cv_image)) 
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    cv_image = transform(cv_image)
    eof_color_images.append(cv_image)
    if len(eof_color_images) > 10:
        eof_color_images.pop(0)  # Keep only the last 10 images

# Callback function for camera image
def eof_depth_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = PIL.Image.fromarray(np.uint8(cv_image)) 
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    cv_image = transform(cv_image)
    eof_depth_images.append(cv_image)
    if len(eof_depth_images) > 10:
        eof_depth_images.pop(0)  # Keep only the last 10 images

# Callback function for camera image
def world_color_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = PIL.Image.fromarray(np.uint8(cv_image)) 
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    cv_image = transform(cv_image)
    world_color_images.append(cv_image)
    if len(world_color_images) > 10:
        world_color_images.pop(0)  # Keep only the last 10 images

def world_depth_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = PIL.Image.fromarray(np.uint8(cv_image)) 
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    cv_image = transform(cv_image)
    world_depth_images.append(cv_image)
    if len(world_depth_images) > 10:
        world_depth_images.pop(0)  # Keep only the last 10 images

# Callback function for tactile image
def thumb_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = torch.tensor(np.array(cv_image), dtype=torch.float32).permute(2, 0, 1) / 255
    thumb_images.append(cv_image)
    if len(thumb_images) > 10:
        thumb_images.pop(0)  # Keep only the last 10 tactile images

# Callback function for tactile image
def index_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_image = torch.tensor(np.array(cv_image), dtype=torch.float32).permute(2, 0, 1) / 255
    index_images.append(cv_image)
    if len(index_images) > 10:
        index_images.pop(0)  # Keep only the last 10 tactile images

# Callback function for allegro hand joint angles
def hand_callback(msg):
    positions = msg.position
    velocities = msg.velocity
    efforts = msg.effort
    # Create a row with the timestamp, positions, velocities, and efforts
    row = list(positions)[12:16] + list(positions)[:4] + list(velocities)[12:16] + list(velocities)[:4] + list(efforts)[12:16] + list(efforts)[:4] #NEEDS FILTERING, APPENDING AND ORDERING!!
    hand_joint_angles.append(row)
    if len(hand_joint_angles) > 10:
        hand_joint_angles.pop(0)  # Keep only the last 10 joint angles

# Callback function for xArm states
def xArm_callback(msg):
    xArm_states = list(msg.data)
    xArm_joint_angles.append(xArm_states)
    if len(xArm_joint_angles) > 10:
        xArm_joint_angles.pop(0)

# Function to save predictions to CSV
def save_predictions_to_csv(predictions):
    # Open the CSV file in append mode
    with open('/home/bonggeeun/Ishrath/sparsh/predictions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the first 20 values of predictions[0] to the CSV
        writer.writerow(predictions[:20].cpu().numpy())  # Extract first 20 values

# Training
@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg:DictConfig):
    # Initialize model
    csv_feature_dim = 44 
    image_feature_dim = 32

    # Load model
    device = torch.device("cuda")# if torch.cuda.is_available() else "cpu"

    # Initialize model
    # model = LSTMModel(csv_feature_dim=csv_feature_dim, output_dim=20, hidden_size=128)#LSTMImagePredictor(cfg, csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128).to(device)
    # model = LSTMImagePredictor(csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128) # Vision lstm model
    # model = LSTMTactilePredictor(cfg, csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128) # Tactile lstm model
    # model = LSTMTactileImagePredictor(cfg, csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128) # Tactile+Vision model

    model = ACTModel(input_dim=29,  # 2048 features from ResNet50
                         output_dim=len(output_columns),
                         input_timesteps=10,
                         output_timesteps=5) # ACT proprio model
    # model = ACTImagesModel(input_dim=len(input_columns) + 32*4 -15,  # 2048 features from ResNet50
    #                      output_dim=len(output_columns),
    #                      input_timesteps=10,
    #                      output_timesteps=5) # Vision ACT model
    # model = ACTTactileModel(cfg, input_dim=44 + 64 - 15,  # 2048 features from ResNet50
    #                         output_dim=20,
    #                         input_timesteps=10,
    #                         output_timesteps=5) # ACT tactile only model
    # model = ActTactileImagesModel(cfg, input_dim=44 + 32*4 + 64 - 15,  # 2048 features from ResNet50
    #                         output_dim=20,
    #                         input_timesteps=10,
    #                         output_timesteps=5) # Tactile+vision model
    
    # model = TransformerModel(input_dim=44, output_dim=20, input_timesteps=10, output_timesteps=5) #  Trans Proprio model
    # model = TransformerImagesModel(input_dim=44 + 32*4,  # 2048 features from ResNet50
    #                      output_dim=20,
    #                      input_timesteps=10,
    #                      output_timesteps=5) # Vision model
    # model = TransformerTactileModel(cfg, input_dim=44 +64,  # 2048 features from ResNet50
    #                         output_dim=20,
    #                         input_timesteps=10,
    #                         output_timesteps=5) # Tactile model
    # model = TransformerTactileVisionModel(cfg, input_dim=44 + 32*4 +64,  # 2048 features from ResNet50
    #                         output_dim=20,
    #                         input_timesteps=10,
    #                         output_timesteps=5) # Tactile+Vision model

    # Load full checkpoint
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/proprioLstm_model.ckpt", map_location="cuda") # LSTM model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/visionPropriolstm_model.ckpt", map_location="cuda") # Vision model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactilePropriolstm_model.ckpt", map_location="cuda") # Tactile model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactileVisionPropriolstm_model.ckpt", map_location="cuda") # Tactile+vision model

    checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/proprioACT_model.ckpt", map_location="cuda") # ACT model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/visionProprioACT_model.ckpt", map_location="cuda") # Vision model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactileProprioACT_model.ckpt", map_location="cuda") # Tactile model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactileVisionProprioACT_model.ckpt", map_location="cuda") # Tactile+Vision model

    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/proprioTrans_model.ckpt", map_location="cuda") # Trans model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/visionProprioTrans_model.ckpt", map_location="cuda") # Vision model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactileProprioTrans_model.ckpt", map_location="cuda") # Tactile model
    # checkpoint = torch.load("/home/bonggeeun/Ishrath/Modelweights_logs/tactileVisionProprioTrans_model.ckpt", map_location="cuda") # Tactile+Vision model

    # Extract only the model state_dict
    model_state_dict = checkpoint["state_dict"]  # Extract actual weights
    model.to(device) #remove for only proprioception

    # Load into model
    model.load_state_dict(model_state_dict)  # strict=False allows partial loading
    model.eval()

    # For realtime evaluation
    # Initialize ROS node
    rospy.init_node('prediction_inference_node', anonymous=False)

    # Initialize ROS Subscribers
    # camera_sub = rospy.Subscriber('/camera_image', Image, camera_callback, queue_size=10) /cameraWorld/color_image, /cameraWorld/depth_image, /cameraEOF/color_image, /cameraEOF/depth_image
    thumb_sub = rospy.Subscriber('/DigitThumb/color_image', Image, thumb_callback, queue_size=1)
    index_sub = rospy.Subscriber('/DigitIndex/color_image', Image, index_callback, queue_size=1)
    eof_color_sub = rospy.Subscriber("/cameraEOF/color_image", Image, eof_color_callback, queue_size=1)
    eof_depth_sub = rospy.Subscriber("/cameraEOF/depth_image", Image, eof_depth_callback, queue_size=1)
    world_color_sub = rospy.Subscriber("/cameraWorld/color_image", Image, world_color_callback, queue_size=1)
    world_depth_sub = rospy.Subscriber("/cameraWorld/depth_image", Image, world_depth_callback, queue_size=1)
    xArm_sub = rospy.Subscriber("/xArm/states", Float32MultiArray, xArm_callback, queue_size=1)
    allegro_sub = rospy.Subscriber("/allegroHand_0/joint_states", JointState, hand_callback, queue_size=1)

    # ROS Publisher for publishing predictions
    predictions_pub = rospy.Publisher('/predicted_states', Float32MultiArray, queue_size=5)
    scaler = joblib.load("/home/bonggeeun/Ishrath/sparsh/Model_scripts/output_successScaler.pkl")

    # Start the ROS event loop
    rospy.loginfo("Ros publisher and subscriber initialised")
    # Periodic check for inference (every 0.02 seconds)
    rate = rospy.Rate(50)  # 50Hz (0.02 seconds)
    preds=[]
    while not rospy.is_shutdown():
        if len(thumb_images) == 10 and len(index_images) == 10 and len(xArm_joint_angles) == 10 and len(hand_joint_angles) == 10 and len(eof_color_images) == 10 and len(world_color_images) == 10:
            joint_angles = np.hstack((xArm_joint_angles, hand_joint_angles))

            # Normalize joint angles using the fitted scaler
            normalized_joint_angles = input_scaler.transform(joint_angles)

            # Convert the data to tensors
            joint_angles_tensor = torch.tensor(normalized_joint_angles).float().unsqueeze(0).to(device) # Remove to device for only proprioception models
            index_tensor = torch.tensor(np.array(index_images)).float().unsqueeze(0).to(device)
            thumb_tensor = torch.tensor(np.array(thumb_images)).float().unsqueeze(0).to(device)
            eof_color_tensor = torch.tensor(np.array(eof_color_images)).float().unsqueeze(0).to(device)
            eof_depth_tensor = torch.tensor(np.array(eof_color_images)).float().unsqueeze(0).to(device)
            world_color_tensor = torch.tensor(np.array(world_color_images)).float().unsqueeze(0).to(device)
            world_depth_tensor = torch.tensor(np.array(world_depth_images)).float().unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                # predictions = model(joint_angles_tensor) # LSTM model
                # predictions = model(joint_angles_tensor, eof_color_tensor, eof_depth_tensor, world_color_tensor, world_depth_tensor) # Vision only
                # predictions = model(joint_angles_tensor, index_tensor, thumb_tensor) # Tactile only
                # predictions = model(joint_angles_tensor, world_color_tensor, world_depth_tensor, eof_color_tensor, eof_depth_tensor, index_tensor, thumb_tensor) # Tactile+Vision model
                
                predictions = model(joint_angles_tensor[:,:, input_indices],joint_angles_tensor[:,5:,action_indices]) # ACT model
                # predictions = model(joint_angles_tensor[:,:, input_indices],joint_angles_tensor[:,5:,action_indices], world_color_tensor, world_depth_tensor, eof_color_tensor, eof_depth_tensor) # Vision model
                # predictions = model(joint_angles_tensor[:,:, input_indices],joint_angles_tensor[:,5:,action_indices], index_tensor, thumb_tensor) # Tactile model
                # predictions = model(joint_angles_tensor[:,:, input_indices],joint_angles_tensor[:,5:,action_indices], world_color_tensor, world_depth_tensor, eof_color_tensor, eof_depth_tensor, index_tensor, thumb_tensor) # Tactile+Vision model
                
                # predictions = model(joint_angles_tensor)
                # predictions = model(joint_angles_tensor, world_color_tensor, world_depth_tensor, eof_color_tensor, eof_depth_tensor) # Vision only
                # predictions = model(joint_angles_tensor, index_tensor, thumb_tensor) # Tactile only
                # predictions = model(joint_angles_tensor, world_color_tensor, world_depth_tensor, eof_color_tensor, eof_depth_tensor, index_tensor, thumb_tensor)
                print(predictions)
            preds_np = predictions.cpu().numpy().squeeze(0)
            # # Cal avg task status
            avgStatus = np.mean(np.abs(preds_np[:,-1]))

            # Extract the first 20 values from the predictions
            # first_prediction = preds_np[np.argmax(preds_np[:, -1])]
            first_prediction = preds_np[0]
            first_prediction[-1] = avgStatus # setting avg tast status
            first_prediction = scaler.inverse_transform(first_prediction.reshape(1, -1))
            # print(first_prediction)

            y_pred_unnormalized = np.array(first_prediction, dtype=np.float32)
            # Publish the first 20 values on the /predicted_states topic
            msg = Float32MultiArray()
            msg.data = y_pred_unnormalized.flatten().tolist()
            predictions_pub.publish(msg)
            preds.append(first_prediction)

        rate.sleep()  # Sleep for 0.02 seconds (50Hz)

if __name__ == "__main__":
    main()
