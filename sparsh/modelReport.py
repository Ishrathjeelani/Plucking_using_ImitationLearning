import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = "/home/bonggeeun/Ishrath/Modelweights_logs/tactileProprioLSTM.csv"
# Define file paths for saving the plots
loss_plot_path = "/home/bonggeeun/Ishrath/sparsh/Plots/tactileProprioLSTM_loss.png"
# mae_plot_path = "/home/bonggeeun/Ishrath/sparsh/Plots/proprioLSTM_mae.png"
r2_plot_path = "/home/bonggeeun/Ishrath/sparsh/Plots/tactileProprioLSTM_r2.png"

data = pd.read_csv(file_path)
df=data
output_columns = ["roll", "pitch", "yaw", "thumb_1_position", "thumb_2_position", "thumb_3_position",
                "thumb_4_position", "index_1_position", "index_2_position", "index_3_position",
                "index_4_position", "thumb_1_torque", "thumb_2_torque", "thumb_3_torque", 
                "thumb_4_torque", "index_1_torque", "index_2_torque", "index_3_torque", 
                "index_4_torque", "task_status"]

# Losses for each feature
train_losses = [f"train_loss_state_{i}" for i in range(1, 21)]
val_losses = [f"val_loss_state_{i}" for i in range(1, 21)]
test_losses = [f"test_loss_state_{i}" for i in range(1, 21)]  # Added test loss columns

# Extract R² columns
train_r2_col = "train_r2"
val_r2_col = "val_r2"
test_r2_col = "test_r2"

# Create figure for loss plots (20 subplots)
fig, axes = plt.subplots(4, 5, figsize=(24, 16))
fig.suptitle("Training vs Validation vs Test Loss for Each State", fontsize=18)

for i, ax in enumerate(axes.flatten()):
    state_name = output_columns[i]
    state = i + 1
    if state > 20:
        break  # Stop after 20 subplots
    
    # Plot train, val, and test loss for this state
    ax.plot(df["epoch"], df[train_losses[i]], label="Train Loss", color='blue', marker="o", markersize=5)
    ax.plot(df["epoch"], df[val_losses[i]], label="Val Loss", color='orange', marker="o", markersize=5)
    ax.plot(df["epoch"], df[test_losses[i]], label="Test Loss", color='green', marker="o", markersize=5)#.iloc[-1]

    ax.set_title(f"{state_name}", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
# plt.show()

# Save plots
plt.savefig(loss_plot_path)

# Create figure for R² scores (single plot with train, val, test)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["epoch"], df[train_r2_col], label="Train R²", color='blue', marker="o", markersize=5)
ax.plot(df["epoch"], df[val_r2_col], label="Val R²", color='orange', marker="o", markersize=5)
ax.plot(df["epoch"], df[test_r2_col], label="Test R²", color='green', marker="o", markersize=5)

ax.set_title("Training vs Validation vs Test R² Score", fontsize=14)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("R² Score", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
# plt.show()

# Save plots
plt.savefig(r2_plot_path)

