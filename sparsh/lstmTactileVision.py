import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError
import torchmetrics
import torchvision.transforms as T
import hydra
from omegaconf import OmegaConf, DictConfig
import PIL
import sys
import joblib
from pytorch_lightning.loggers import CSVLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Setup CSV logger
logger = CSVLogger("logs", name="my_model")

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

class SparshFeatureExtractor(nn.Module):
    """
    A class to wrap the SPARSH model and extract features from image pairs.
    """
    def __init__(self, cfg: DictConfig, checkpoint_path= "/home/bonggeeun/Ishrath/sparsh/checkpoints/mae_vitbase.ckpt"):
        checkpoint_path = "/home/bonggeeun/Ishrath/sparsh/checkpoints/mae_vitbase.ckpt"
        super(SparshFeatureExtractor, self).__init__()

        # Instantiate model from the Hydra config
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.cuda()
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        # Freeze model parameters
        self.model.requires_grad_(False)
        print("Model loaded successfully!")

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        
        # Concatenate the two images along the channel dimension
        combined_tensor = torch.cat([img1, img2], dim=1)  # Shape: (B, 6, H, W)

        # Extract features using the SPARSH model
        with torch.no_grad():
            pred, mask = self.model(combined_tensor)

        return pred
    
def assign_uniform_values(df, start, end):
    """Assign values to process stage"""
    num_rows = len(df)
    df["task_status"] = np.linspace(start, end, num=num_rows)  # Uniformly distributed values
    return df

class ResNetFeatureExtractor(nn.Module):
    """A class to extract features from images using pretrained ResNet-50."""
    def __init__(self, input_channels=3, pretrained=True, output_dim=32):
        super(ResNetFeatureExtractor, self).__init__()
        # Use a pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer since it's not needed for feature extraction
        self.resnet.fc = nn.Identity()
        
        # Set the ResNet to evaluation mode to prevent BatchNorm and Dropout layers from updating
        self.resnet.eval()
        
        # Disable gradient computation for all ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False  

    def forward(self, x):
        # Ensure no gradients are computed during the forward pass
        with torch.no_grad():
            features = self.resnet(x)  # Extract features using ResNet
        
        # Apply the feature reduction layer
        reduced_features = features#self.feature_reduction(features)
        
        return reduced_features

def load_images(img_folder):
    """Load images from path and perform transformations"""
    images = []
    image_files = sorted(os.listdir(img_folder))  # Ensure the files are sorted
    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path).convert('RGB')  # Assuming images are RGB
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
        img = transform(img)
        images.append(img)
    return torch.stack(images)

def load_image_paths(img_folder):
    """Load images' path from data collected"""
    image_paths = []
    image_files = sorted(os.listdir(img_folder))  # Ensure the files are sorted
    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)  # Full path to the image
        image_paths.append(img_path)  # Append the image path to the list
    return image_paths

def load_tactile_images(img_folder):
    """Load images from path and perform preprocessing for SPARSH"""
    images = []
    image_files = sorted(os.listdir(img_folder))  # Ensure the files are sorted
    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path).convert('RGB')  # Assuming images are RGB
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255
        images.append(img)
    return torch.stack(images)

def create_sequences_with_images(data, img_folder_World_color, img_folder_World_depth, img_folder_EOF_color, img_folder_EOF_depth,img_folder_digit_index, img_folder_digit_thumb, input_timesteps, output_timesteps):
    """Create a sequence of proprioception, camera and digit images"""
    image_data_world_color = load_image_paths(img_folder_World_color)
    image_data_world_depth = load_image_paths(img_folder_World_depth)
    image_data_eof_color = load_image_paths(img_folder_EOF_color)
    image_data_eof_depth = load_image_paths(img_folder_EOF_depth)
    img_data_digit_index = load_image_paths(img_folder_digit_index)
    img_data_digit_thumb = load_image_paths(img_folder_digit_thumb)

    X, y_states, images_world_color, images_world_depth, images_eof_color, images_eof_depth, images_digit_index, images_digit_thumb = [], [], [], [], [], [], [], []
    for i in range(len(data) - input_timesteps - output_timesteps + 1):
        X.append(data[input_columns].iloc[i : i + input_timesteps].values)
        y_states.append(data[output_columns].iloc[i + input_timesteps : i + input_timesteps + output_timesteps])

        images_world_color.append(image_data_world_color[i:i + input_timesteps])
        images_world_depth.append(image_data_world_depth[i:i + input_timesteps])
        images_eof_color.append(image_data_eof_color[i:i + input_timesteps])
        images_eof_depth.append(image_data_eof_depth[i:i + input_timesteps])
        images_digit_index.append(img_data_digit_index[i:i + input_timesteps])
        images_digit_thumb.append(img_data_digit_thumb[i:i + input_timesteps])

    del image_data_world_color, image_data_world_depth, image_data_eof_color, image_data_eof_depth, img_data_digit_index, img_data_digit_thumb
    return np.array(X), np.array(y_states), np.array(images_world_color), np.array(images_world_depth), np.array(images_eof_color),np.array(images_eof_depth), np.array(images_digit_index), np.array(images_digit_thumb)

def load_data(base_dir="/home/bonggeeun/Ishrath/sparsh/Model_scripts/test_data"):
    """Load collected data from csv file and images folder"""
    success_folder = os.path.join(base_dir, "Success")
    failed_folder = os.path.join(base_dir, "Failed")

    success_files = [os.path.join(success_folder, f) for f in os.listdir(success_folder) if f.endswith('.csv')]

    scaler = joblib.load("/home/bonggeeun/Ishrath/sparsh/Model_scripts/succesScaler.pkl")

    scaled_data_all = []
    y_states_all = []
    input_columns_all, image_data_all_wc, image_data_all_wd, image_data_all_ec, image_data_all_ed, digit_index_all, digit_thumb_all = [], [], [], [], [], [], []

    # Process Success Files
    for file in success_files:
        print(file)
        df = pd.read_csv(file)
        df = df.drop(columns=exclude_columns)
        df = assign_uniform_values(df, 0, 100)
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)

        csv_prefix = os.path.basename(file).split('.')[0]

        if csv_prefix.startswith("filtered_") or csv_prefix.startswith("Rfiltered_"):
            img_folder_World_color = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                  .replace("Rfiltered_", "Rfiltered_images"), "World_color")
            img_folder_World_depth = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                  .replace("Rfiltered_", "Rfiltered_images"), "World_depth")
            img_folder_EOF_color = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                .replace("Rfiltered_", "Rfiltered_images"), "EOF_color")
            img_folder_EOF_depth = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                .replace("Rfiltered_", "Rfiltered_images"), "EOF_depth")
            img_folder_digit_index = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                .replace("Rfiltered_", "Rfiltered_images"), "Digit_index")
            img_folder_digit_thumb = os.path.join(base_dir, "Success", csv_prefix.replace("filtered_", "filtered_images")
                                                .replace("Rfiltered_", "Rfiltered_images"), "Digit_thumb")

            X, y_states, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb = create_sequences_with_images(df, img_folder_World_color, img_folder_World_depth,
                                                                 img_folder_EOF_color, img_folder_EOF_depth, img_folder_digit_index, img_folder_digit_thumb, input_timesteps=10, output_timesteps=5)

            scaled_data_all.append(X)
            y_states_all.append(y_states)
            input_columns_all.append(df.columns)
            image_data_all_wc.append(image_data_wc)
            image_data_all_wd.append(image_data_wd)
            image_data_all_ec.append(image_data_ec)
            image_data_all_ed.append(image_data_ed)
            digit_index_all.append(digit_data_index)
            digit_thumb_all.append(digit_data_thumb)
            del X, y_states, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb

    scaled_data = np.concatenate(scaled_data_all, axis=0)
    y_states = np.concatenate(y_states_all, axis=0)
    input_columns = input_columns_all[0]
    image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb = np.concatenate(image_data_all_wc, axis=0), np.concatenate(image_data_all_wd, axis=0), np.concatenate(image_data_all_ec, axis=0), np.concatenate(image_data_all_ed, axis=0), np.concatenate(digit_index_all, axis=0), np.concatenate(digit_thumb_all, axis=0)

    del df, scaled_data_all, image_data_all_wc, image_data_all_wd, image_data_all_ec, image_data_all_ed, digit_index_all, digit_thumb_all

    return scaled_data, y_states, input_columns, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb

    
class LSTMTactileImagePredictor(pl.LightningModule):
    """LSTM network to predict next states and process status"""
    def __init__(self,cfg, csv_feature_dim, image_feature_dim, hidden_size, input_timesteps=10, future_steps=5, learning_rate=1e-3):
        super(LSTMTactileImagePredictor, self).__init__()
        self.future_steps = future_steps
        self.input_timesteps = input_timesteps
        self.learning_rate = learning_rate
        # Feature extractors for camera and tactile images
        self.resnet_feature_extractor = ResNetFeatureExtractor()
        self.sparsh_feature_extractor = SparshFeatureExtractor(cfg)
        # Feature reduction layer to reduce the dimensionality to `output_dim`
        self.feature_reduction = nn.Linear(2048, 32) #2048 for resnet 8192
        self.tactile_reduction = nn.Linear(1536, 64) #2 tatile images already concatenated and fed
        combined_feature_dim = csv_feature_dim + 32*4 + 64
        self.lstm = nn.LSTM(combined_feature_dim, hidden_size, batch_first=True)

        # Fully connected layer for outputting predictions
        self.fc = nn.Linear(hidden_size, future_steps * 20)  # 20 values for each step

        # Define metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, sequence, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb):
        self.resnet_feature_extractor.eval()
        
        # Initialize lists to store features
        image_features_wc, image_features_wd, image_features_ec, image_features_ed, tactile_features = [], [], [], [], []
        
        # Iterate through timesteps
        for t in range(self.input_timesteps):
            # Process each set of image paths (image_data_* is now a list of paths)
            
            # Load and extract features for wc images
            image_t = image_data_wc[:, t, :, :, :] 
            feature_t = self.resnet_feature_extractor(image_t)
            feature_t = self.feature_reduction(feature_t)
            image_features_wc.append(feature_t.unsqueeze(1))
            
            # Load and extract features for wd images
            image_t = image_data_wd[:, t, :, :, :] 
            feature_t = self.resnet_feature_extractor(image_t)
            feature_t = self.feature_reduction(feature_t)
            image_features_wd.append(feature_t.unsqueeze(1))
            
            # Load and extract features for ec images
            image_t = image_data_ec[:, t, :, :, :] 
            feature_t = self.resnet_feature_extractor(image_t)
            feature_t = self.feature_reduction(feature_t)
            image_features_ec.append(feature_t.unsqueeze(1))
            
            # Load and extract features for ed images
            image_t = image_data_ed[:, t, :, :, :] 
            feature_t = self.resnet_feature_extractor(image_t)
            feature_t = self.feature_reduction(feature_t)
            image_features_ed.append(feature_t.unsqueeze(1))
            
            # Process tactile data (index and thumb images)
            image_index = digit_data_index[:, t, :, :, :] 
            # image_index = torch.tensor(np.array(image_t), dtype=torch.float32).permute(2, 0, 1) / 255
            image_thumb = digit_data_thumb[:, t, :, :, :] 
            # image_thumb = torch.tensor(np.array(image_t), dtype=torch.float32).permute(2, 0, 1) / 255
            
            feature_t = self.sparsh_feature_extractor(image_index, image_thumb)  # Extract tactile features
            feature_t = feature_t.mean(dim=1)  # Aggregate across the sequence
            feature_t = self.tactile_reduction(feature_t)
            tactile_features.append(feature_t.unsqueeze(1))
        
        # Concatenate features along the timesteps dimension
        image_features_wc = torch.cat(image_features_wc, dim=1)  # Shape: [batch_size, timesteps, feature_dim]
        image_features_wd = torch.cat(image_features_wd, dim=1)
        image_features_ec = torch.cat(image_features_ec, dim=1)
        image_features_ed = torch.cat(image_features_ed, dim=1)
        tactile_features = torch.cat(tactile_features, dim=1)

        # Concatenate sequence data with image and tactile features
        combined_sequence = torch.cat((sequence, image_features_wc, image_features_wd, image_features_ec, image_features_ed, tactile_features), dim=2)

        # Pass the combined sequence through the LSTM
        lstm_out, _ = self.lstm(combined_sequence)  # Shape: (batch_size, seq_len, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state

        # Predict future steps
        output = self.fc(lstm_out)
        output = output.view(-1, self.future_steps, 20)  # Reshape to (batch_size, future_steps, 53)

        return output

    def training_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb)#[:, input_indices][:,action_indices]

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"train_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        accuracy = self.train_mae(predicted_states, y_states)

        y_pred = predicted_states.view(-1, predicted_states.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = y_states.view(-1, y_states.shape[-1])
        self.log("train_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss
    
    def validation_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb)#[:, input_indices][:,action_indices]

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"val_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        accuracy = self.train_mae(predicted_states, y_states)

        y_pred = predicted_states.view(-1, predicted_states.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = y_states.view(-1, y_states.shape[-1])
        self.log("val_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss

    def test_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb)#[:, input_indices][:,action_indices]

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"test_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        accuracy = self.train_mae(predicted_states, y_states)

        y_pred = predicted_states.view(-1, predicted_states.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = y_states.view(-1, y_states.shape[-1])
        self.log("test_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class CustomDataset(Dataset):
    """Class to create dataset of data collected"""
    def __init__(self,x, y_states, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb):
        """
        Args:
            image_paths (list): List of file paths to the images.
            labels (list, optional): List of labels corresponding to each image. Default is None.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.x = x
        self.y_states = y_states
        self.image_data_wc = image_data_wc
        self.image_data_wd = image_data_wd
        self.image_data_ec = image_data_ec
        self.image_data_ed = image_data_ed
        self.digit_data_index = digit_data_index
        self.digit_data_thumb = digit_data_thumb
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
        self.tactile_transform = transforms.Compose([ 
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        image_sequence_len = len(self.image_data_wc[idx])  # A sequence of paths for each timestep
        image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb = [], [], [], [], [], []
        
        for i in range(0,image_sequence_len):
            img_wc = Image.open(self.image_data_wc[idx,i]).convert('RGB')  # Open each image
            img_wc = self.transform(img_wc)
            img_wd = Image.open(self.image_data_wd[idx,i]).convert('RGB')  # Open each image
            img_wd = self.transform(img_wd)
            img_ec = Image.open(self.image_data_ec[idx,i]).convert('RGB')  # Open each image
            img_ec = self.transform(img_ec)
            img_ed = Image.open(self.image_data_ed[idx,i]).convert('RGB')  # Open each image
            img_ed = self.transform(img_ed)

            img_index = Image.open(self.digit_data_index[idx,i]).convert('RGB')  # Open each image
            img_index = torch.tensor(np.array(img_index), dtype=torch.float32).permute(2, 0, 1) / 255
            img_thumb = Image.open(self.digit_data_thumb[idx,i]).convert('RGB')  # Open each image
            img_thumb = torch.tensor(np.array(img_thumb), dtype=torch.float32).permute(2, 0, 1) / 255

            image_data_wc.append(img_wc)
            image_data_wd.append(img_wd)
            image_data_ec.append(img_ec)
            image_data_ed.append(img_ed)
            digit_data_index.append(img_index)
            digit_data_thumb.append(img_thumb)
        
        return self.x[idx], self.y_states[idx], torch.stack(image_data_wc), torch.stack(image_data_wd), torch.stack(image_data_ec), torch.stack(image_data_ed), torch.stack(digit_data_index), torch.stack(digit_data_thumb)

# Training
@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg:DictConfig):
# def main():    
    # # Training and Testing
    # X, y_states, input_columns, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb = load_data(base_dir="/home/bonggeeun/Ishrath/sparsh/Model_scripts/data")
    # input_timesteps, output_timesteps = 10, 5
    # print("data loaded")
    # # Split the data into training and temp (temp will be further split into validation and test)
    # X_train, X_temp, y_states_train, y_states_temp, image_data_wc_train, image_data_wc_temp, image_data_wd_train, image_data_wd_temp , image_data_ec_train, image_data_ec_temp , image_data_ed_train, image_data_ed_temp, digit_data_index_train, digit_data_index_temp, digit_data_thumb_train, digit_data_thumb_temp  = train_test_split(
    #     X, y_states, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb, test_size=0.2, random_state=42
    # )
    # del X, y_states, input_columns, image_data_wc, image_data_wd, image_data_ec, image_data_ed, digit_data_index, digit_data_thumb
    # # Now, split the temp set into validation and test sets (e.g., 50% validation, 50% test)
    # X_val, X_test, y_states_val, y_states_test, image_data_wc_val, image_data_wc_test, image_data_wd_val, image_data_wd_test, image_data_ec_val, image_data_ec_test, image_data_ed_val, image_data_ed_test, digit_data_index_val, digit_data_index_test, digit_data_thumb_val, digit_data_thumb_test = train_test_split(
    #     X_temp, y_states_temp, image_data_wc_temp, image_data_wd_temp,image_data_ec_temp,image_data_ed_temp, digit_data_index_temp, digit_data_thumb_temp, test_size=0.5, random_state=42
    # )
    # del X_temp, y_states_temp, image_data_wc_temp, image_data_wd_temp,image_data_ec_temp,image_data_ed_temp, digit_data_index_temp, digit_data_thumb_temp
    # print("data split started")
    # train_dataset = CustomDataset(torch.tensor(X_train, dtype=torch.float32),
    #                                torch.tensor(y_states_train, dtype=torch.float32),
    #                                image_data_wc_train, image_data_wd_train, image_data_ec_train, image_data_ed_train, digit_data_index_train, digit_data_thumb_train)
    # val_dataset = CustomDataset(torch.tensor(X_val, dtype=torch.float32),
    #                              torch.tensor(y_states_val, dtype=torch.float32),
    #                              image_data_wc_val, image_data_wd_val, image_data_ec_val, image_data_ed_val, digit_data_index_val, digit_data_thumb_val)
    # test_dataset = CustomDataset(torch.tensor(X_test, dtype=torch.float32),
    #                               torch.tensor(y_states_test, dtype=torch.float32),
    #                               image_data_wc_test, image_data_wd_test, image_data_ec_test, image_data_ed_test, digit_data_index_test, digit_data_thumb_test)
   
    # # Save the train, val, and test datasets
    # torch.save(train_dataset, 'TactileVision_train_dataset.pth')
    # torch.save(val_dataset, 'TactileVision_val_dataset.pth')
    # torch.save(test_dataset, 'TactileVision_test_dataset.pth')
    # print("datasets saved")

    # Load the train, val, and test datasets
    train_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
    val_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
    test_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)

    # Create DataLoader instances for each split
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32,  num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32,  num_workers=4, shuffle=False)

    # Initialize model
    csv_feature_dim = 44  # The number of features from the CSV
    print(csv_feature_dim)
    image_feature_dim = 32#12288 # For dalle 8192
    model = LSTMTactileImagePredictor(cfg, csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128)

    # Train the model
    trainer = pl.Trainer(logger=logger,max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')#fast_dev_run=True 
    # Fit the model with the training data and validation data
    trainer.fit(model, train_loader, val_loader)

    # You can also evaluate the model on the test set after training
    trainer.test(model, test_loader)

    # Save model weights
    trainer.save_checkpoint("tactileVisionPropriolstm_model.ckpt")
    print("Model weights saved as 'tactileVisionPropriolstm_model.ckpt'")

# if __name__ == "__main__":
#     main()
