import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
from torchmetrics import MeanAbsoluteError
import torchmetrics
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

class LSTMModel(pl.LightningModule):
    """LSTM network to predict next states and process status"""
    def __init__(self, csv_feature_dim, output_dim,  hidden_size, input_timesteps=10, future_steps=5, learning_rate=1e-3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(csv_feature_dim, hidden_size, batch_first=True)

        # Fully connected layer for outputting predictions
        self.fc = nn.Linear(hidden_size, future_steps * 20)  # 20 values for each step

        self.future_steps = future_steps
        # Define metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()
    
    def forward(self, x):
        # LSTM encoder
         # Pass the combined sequence through the LSTM
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state

        # Predict future steps
        output = self.fc(lstm_out)
        output = output.view(-1, self.future_steps, 20)  # Reshape to (batch_size, future_steps, 53)

        return output
        
    def training_step(self, batch, batch_idx):
        sequence, future_values, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predictions = self(sequence)
        
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
        predictions = self(sequence)
        
        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predictions, future_values)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"val_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predictions, future_values)
        accuracy = self.train_mae(predictions, future_values)

        y_pred = predictions.view(-1, predictions.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = future_values.view(-1, future_values.shape[-1])
        self.log("val_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return state_loss

    def test_step(self, batch, batch_idx):  # **Newly Added**
        sequence, future_values, images_wc, images_wd, images_ec, images_ed, digit_index, digit_thumb = batch
        predictions = self(sequence)
        
        # Calculate state loss (MSE)
        state_loss = nn.MSELoss(reduction='none')(predictions, future_values)
        for i in range(state_loss.shape[2]):  # Loop over each state
            self.log(f"test_loss_state_{i+1}", state_loss[:, :, i].mean(), prog_bar=True)
        state_loss = nn.MSELoss()(predictions, future_values)
        accuracy = self.train_mae(predictions, future_values)

        y_pred = predictions.view(-1, predictions.shape[-1])  # Shape -> [batch_size * time_steps, num_features]
        y_true = future_values.view(-1, future_values.shape[-1])
        self.log("test_loss", state_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mae", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_r2", self.r2(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return state_loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# # Training and Testing
# # Load the train, val, and test datasets
# train_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
# val_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)
# test_dataset = torch.load('TactileVision_train_dataset.pth', weights_only=False)

# train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

# print("Data loaded")
# del train_dataset,val_dataset,test_dataset

# # Initialize model, trainer, and train
# csv_feature_dim = 44
# output_dim = 20
# output_timesteps = 5

# model = LSTMModel(csv_feature_dim=csv_feature_dim, output_dim=output_dim, hidden_size=128)

# trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
# trainer.fit(model, train_loader, val_loader)
# trainer.test(model, test_loader)
# # Save model weights
# trainer.save_checkpoint("proprioLstm_model.ckpt")
# print("Model weights saved as 'proprioLstm_model.ckpt'")