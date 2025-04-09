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
from pytorch_lightning.loggers import CSVLogger
from lstmTactileVision import CustomDataset
from pytorch_lightning.loggers import CSVLogger

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
       
class LSTMTactilePredictor(pl.LightningModule):
    # @hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
    def __init__(self,cfg, csv_feature_dim, image_feature_dim, hidden_size, input_timesteps=10, future_steps=5, learning_rate=1e-3):
        super(LSTMTactilePredictor, self).__init__()
        self.future_steps = future_steps
        self.input_timesteps = input_timesteps
        self.learning_rate = learning_rate

        self.sparsh_feature_extractor = SparshFeatureExtractor(cfg)
        # Add a feature reduction layer to reduce the dimensionality to `output_dim`
        self.tactile_reduction = nn.Linear(1536, 64) #2 tatile images already concatenated and fed
        # LSTM for combined sequence (CSV + reduced image features)
        combined_feature_dim = csv_feature_dim + 64
        self.lstm = nn.LSTM(combined_feature_dim, hidden_size, batch_first=True)

        # Fully connected layer for outputting predictions
        self.fc = nn.Linear(hidden_size, future_steps * 20)  # 20 values for each step

        # Define metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, sequence, digit_data_index, digit_data_thumb):
        # Initialize lists to store features
        tactile_features = []
        
        # Iterate through timesteps
        for t in range(self.input_timesteps):
            
            # Process tactile data (index and thumb images)
            image_index = digit_data_index[:, t, :, :, :] 
            image_thumb = digit_data_thumb[:, t, :, :, :] 
            
            feature_t = self.sparsh_feature_extractor(image_index, image_thumb)  # Extract tactile features
            feature_t = feature_t.mean(dim=1)  # Aggregate across the sequence
            feature_t = self.tactile_reduction(feature_t)
            tactile_features.append(feature_t.unsqueeze(1))
        
        # Concatenate features along the timesteps dimension
        tactile_features = torch.cat(tactile_features, dim=1)

        # Concatenate sequence data with image and tactile features
        combined_sequence = torch.cat((sequence, tactile_features), dim=2)

        # Pass the combined sequence through the LSTM
        lstm_out, _ = self.lstm(combined_sequence)  # Shape: (batch_size, seq_len, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state

        # Predict future steps
        output = self.fc(lstm_out)
        output = output.view(-1, self.future_steps, 20)  # Reshape to (batch_size, future_steps, 53)

        return output


    def training_step(self, batch, batch_idx):
        sequence, future_values, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predictions = self.forward(sequence, digit_index, digit_thumb)

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predictions, future_values)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"train_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predictions, future_values)
        accuracy = self.train_mae(predictions, future_values)

        y_pred = predictions.view(-1, predictions.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = future_values.view(-1, future_values.shape[-1])
        self.log("train_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss
    
    def validation_step(self, batch, batch_idx):
        sequence, future_values, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predictions = self.forward(sequence, digit_index, digit_thumb)
        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predictions, future_values)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"val_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predictions, future_values)
        accuracy = self.val_mae(predictions, future_values)

        y_pred = predictions.view(-1, predictions.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = future_values.view(-1, future_values.shape[-1])
        self.log("val_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss

    def test_step(self, batch, batch_idx):
        sequence, future_values, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predictions = self.forward(sequence, digit_index, digit_thumb)
        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predictions, future_values)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"test_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predictions, future_values)
        accuracy = self.val_mae(predictions, future_values)

        y_pred = predictions.view(-1, predictions.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = future_values.view(-1, future_values.shape[-1])
        self.log("test_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return state_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg:DictConfig):
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
    model = LSTMTactilePredictor(cfg, csv_feature_dim=csv_feature_dim, image_feature_dim=image_feature_dim, hidden_size=128)

    # Train the model
    trainer = pl.Trainer(logger=logger,max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')#fast_dev_run=True 
    # Fit the model with the training data and validation data
    trainer.fit(model, train_loader, val_loader)

    # You can also evaluate the model on the test set after training
    trainer.test(model, test_loader)

    # Save model weights
    trainer.save_checkpoint("tactilePropriolstm_model.ckpt")
    print("Model weights saved as 'tactilePropriolstm_model.ckpt'")

# if __name__ == "__main__":
#     main()
