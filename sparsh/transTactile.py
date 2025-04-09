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
from lstmTactileVision import CustomDataset
from pytorch_lightning.loggers import CSVLogger

# Setup CSV logger
logger = CSVLogger("logs", name="my_model")

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

class PositionalEncoding(nn.Module):
    """Positional Encoding for Temporal Awareness"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerTactileModel(pl.LightningModule):
    """Transformer Model to predict next states and process status"""
    def __init__(self, cfg, input_dim, output_dim, input_timesteps, output_timesteps, d_model=128, num_heads=8, num_layers=4):
        super(TransformerTactileModel, self).__init__()
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        self.state_output_layer = nn.Linear(d_model, output_dim)

        # Feature extraction from tactile images
        self.sparsh_feature_extractor = SparshFeatureExtractor(cfg)
        # Add a feature reduction layer to reduce the dimensionality to `output_dim`
        self.tactile_reduction = nn.Linear(1536, 64)

        # Define metrics
        self.mae = MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x, digit_index, digit_thumb):
        batch_size = x.shape[0]
        tactile_features = []
        for t in range(self.input_timesteps):
            # Process tactile data (index and thumb images)
            image_index = digit_index[:, t, :, :, :] 
            image_thumb = digit_thumb[:, t, :, :, :] 
            
            feature_t = self.sparsh_feature_extractor(image_index, image_thumb)  # Extract tactile features
            feature_t = feature_t.mean(dim=1)  # Aggregate across the sequence
            feature_t = self.tactile_reduction(feature_t)
            tactile_features.append(feature_t.unsqueeze(1))
        
        tactile_features = torch.cat(tactile_features, dim=1)
        # Concatenate the image features with tabular data (x)
        x = torch.cat((x,tactile_features), dim=-1)  # Concatenate along the feature dimension
        x = self.input_projection(x)  # Project the concatenated input to the model's input dimension
        x = self.positional_encoding(x)  # Apply positional encoding

        # Pass through the encoder
        encoder_output = self.encoder(x)
        
        # Prepare decoder input (zero initialization for simplicity here)
        decoder_input = torch.zeros((batch_size, self.output_timesteps, encoder_output.shape[2]), device=x.device)
        decoder_input = self.positional_encoding(decoder_input)

        # Decoder output
        decoded_output = self.decoder(decoder_input, encoder_output)
        predicted_states = self.state_output_layer(decoded_output)
        
        return predicted_states

    def compute_metrics(self, predicted_states, y_states):
        """ Compute MAE and R² Score """
        predicted_states_flat = predicted_states.view(-1, predicted_states.shape[-1])
        y_states_flat = y_states.view(-1, y_states.shape[-1])

        mae_value = self.mae(predicted_states_flat, y_states_flat)
        r2_value = self.r2(predicted_states_flat, y_states_flat)

        return mae_value, r2_value

    def training_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, digit_index, digit_thumb)

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"train_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        mae_value, r2_value = self.compute_metrics(predicted_states, y_states)

        self.log("train_loss", state_loss, prog_bar=True)
        self.log("train_mae", mae_value, prog_bar=True)
        self.log("train_r2", r2_value, prog_bar=True)
        
        return state_loss

    def validation_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, digit_index, digit_thumb)

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"val_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        mae_value, r2_value = self.compute_metrics(predicted_states, y_states)

        self.log("val_loss", state_loss, prog_bar=True)
        self.log("val_mae", mae_value, prog_bar=True)
        self.log("val_r2", r2_value, prog_bar=True)
        
        return state_loss

    def test_step(self, batch, batch_idx):
        x, y_states, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predicted_states = self(x, digit_index, digit_thumb)

        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predicted_states, y_states)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"test_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predicted_states, y_states)
        mae_value, r2_value = self.compute_metrics(predicted_states, y_states)

        self.log("test_loss", state_loss, prog_bar=True)
        self.log("test_mae", mae_value, prog_bar=True)
        self.log("test_r2", r2_value, prog_bar=True)

        return state_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg:DictConfig):
    # # Training and Testing
    # Load the train, val, and test datasets
    train_dataset = torch.load('/home/bonggeeun/Ishrath/sparsh/TactileVision_train_dataset.pth', weights_only=False)
    val_dataset = torch.load('/home/bonggeeun/Ishrath/sparsh/TactileVision_val_dataset.pth', weights_only=False)
    test_dataset = torch.load('/home/bonggeeun/Ishrath/sparsh/TactileVision_test_dataset.pth', weights_only=False)
    print("Dataset loaded")
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, shuffle=False)

    print("Train val test loaded")
    del train_dataset,val_dataset,test_dataset

    model = TransformerTactileModel(cfg, input_dim=44 +64,  # 2048 features from ResNet50
                            output_dim=20,
                            input_timesteps=10,
                            output_timesteps=5)

    trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    # Save model weights
    trainer.save_checkpoint("tactileProprioTrans_model.ckpt")
    print("Model weights saved as 'tactileProprioTrans_model.ckpt'")

# if __name__ == "__main__":
#     main()
