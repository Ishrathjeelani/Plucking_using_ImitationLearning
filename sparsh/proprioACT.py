import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, models
from PIL import Image
import joblib
from lstmTactileVision import CustomDataset
from torchmetrics import MeanAbsoluteError
import torchmetrics
from pytorch_lightning.loggers import CSVLogger

# Setup CSV logger
logger = CSVLogger("logs", name="my_model")

column_names=['x_pos', 'y_pos', 'z_pos', 'roll', 'pitch', 'yaw',
       'xarm_joint_1_position', 'xarm_joint_2_position',
       'xarm_joint_3_position', 'xarm_joint_4_position',
       'xarm_joint_5_position', 'xarm_joint_6_position',
       'xarm_joint_7_position', 'xarm_joint_1_torque', 'xarm_joint_2_torque',
       'xarm_joint_3_torque', 'xarm_joint_4_torque', 'xarm_joint_5_torque',
       'xarm_joint_6_torque', 'xarm_joint_7_torque', 'thumb_1_position',
       'thumb_2_position', 'thumb_3_position', 'thumb_4_position',
       'index_1_position', 'index_2_position', 'index_3_position',
       'index_4_position', 'thumb_1_velocity', 'thumb_2_velocity',
       'thumb_3_velocity', 'thumb_4_velocity', 'index_1_velocity',
       'index_2_velocity', 'index_3_velocity', 'index_4_velocity',
       'thumb_1_torque', 'thumb_2_torque', 'thumb_3_torque', 'thumb_4_torque',
       'index_1_torque', 'index_2_torque', 'index_3_torque', 'index_4_torque']

input_columns = ['x_pos', 'y_pos', 'z_pos', 'roll', 'pitch', 'yaw',
                'xarm_joint_1_position', 'xarm_joint_2_position',
                'xarm_joint_3_position', 'xarm_joint_4_position',
                'xarm_joint_5_position', 'xarm_joint_6_position',
                'xarm_joint_7_position', 
                'thumb_1_position', 'thumb_2_position', 'thumb_3_position',
                'thumb_4_position', 'index_1_position', 'index_2_position',
                'index_3_position', 'index_4_position', 'thumb_1_velocity',
                'thumb_2_velocity', 'thumb_3_velocity', 'thumb_4_velocity',
                'index_1_velocity', 'index_2_velocity', 'index_3_velocity',
                'index_4_velocity']

action_columns = ['xarm_joint_1_torque','xarm_joint_2_torque', 'xarm_joint_3_torque', 'xarm_joint_4_torque',
                'xarm_joint_5_torque', 'xarm_joint_6_torque', 'xarm_joint_7_torque', 'thumb_1_torque', 'thumb_2_torque', 'thumb_3_torque',
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

input_indices = [column_names.index(col) for col in input_columns]
action_indices = [column_names.index(col) for col in action_columns]

def assign_uniform_values(df, start, end):
    """Assign values to process stage"""
    num_rows = len(df)
    df["task_status"] = np.linspace(start, end, num=num_rows)  # Uniformly distributed values
    return df

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

class ACTModel(pl.LightningModule):
    """Transformer Model to predict next states and process status"""
    def __init__(self, input_dim, output_dim, input_timesteps, output_timesteps, d_model=128, num_heads=8, num_layers=4, action_dim=15):
        super(ACTModel, self).__init__()
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.action_dim = action_dim  # Action dimension

        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        # Action Conditioning: Linear projection for the concatenated input
        self.action_projection = nn.Linear(d_model + action_dim, d_model)

        # Output layers
        self.state_output_layer = nn.Linear(d_model, output_dim)

        # Metrics for evaluation
        self.mae = torchmetrics.MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x, actions):
        batch_size = x.shape[0]

        # Encoder: process the state input only (no action input)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoder_output = self.encoder(x)

        # Prepare the decoder input (usually zeros or some initial sequence)
        decoder_input = torch.zeros((batch_size, self.output_timesteps, encoder_output.shape[2]), device=x.device)
        decoder_input = self.positional_encoding(decoder_input)

        decoder_input = torch.cat((decoder_input, actions), dim=-1)

        decoder_input = self.action_projection(decoder_input)

        decoder_input = decoder_input.view(batch_size, self.output_timesteps, -1)

        # Decoder generates the predicted states and success probability
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
        predicted_states = self(x[:,:, input_indices],x[:,5:,action_indices])#[:, input_indices][:,action_indices]

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
        predicted_states = self(x[:,:, input_indices],x[:,5:,action_indices])#[:, input_indices][:,action_indices]

        # Calculate state loss (MSE)
        # state_loss = nn.MSELoss()(predicted_states, y_states)
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
        predicted_states = self(x[:,:, input_indices],x[:,5:,action_indices])#[:, input_indices][:,action_indices]

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

# # Training and Testing
# train_loader = DataLoader(train_dataset, batch_size=32, num_workers=3, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, num_workers=3, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, num_workers=3, shuffle=False)
# del train_dataset,val_dataset,test_dataset

# # Load the train, val, and test datasets
# train_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
# val_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
# test_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)

# train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
# model = ACTModel(input_dim=29,  # 2048 features from ResNet50
#                          output_dim=20,
#                          input_timesteps=10,
#                          output_timesteps=5)

# trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
# trainer.fit(model, train_loader, val_loader)
# trainer.test(model, test_loader)
# # Save model weights
# trainer.save_checkpoint("proprioACT_model.ckpt")
# print("Model weights saved as 'proprioACT_model.ckpt'")

