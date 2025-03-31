import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time  # For timing epochs

# ================================================================
# 0. Configuration and Device Setup
# ================================================================
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (can be adjusted)
T = 48  # Window size (timesteps)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
VALIDATION_SPLIT = 0.1 # Percentage of training data for validation
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6

# ================================================================
# 1. Data Loading and Preprocessing (Same as TensorFlow version)
# ================================================================
print("Loading and preprocessing data...")
# Load lights and AC CSV files and merge them on Timestamp.
lights_df = pd.read_csv("historical_data/lights_2025.csv", index_col=0, parse_dates=True)
ac_df = pd.read_csv("historical_data/ac_2025.csv", index_col=0, parse_dates=True)

# Merge using the index (Timestamp). We assume both files have the same ordering.
df = lights_df.copy()
df["AC_Set_Temp"] = ac_df["AC_Set_Temp"]

# Reset index to have Timestamp as a column.
df = df.reset_index()

# Ensure numeric columns are converted correctly.
for col in ["OutdoorTemp", "EnergyCost", "Brightness", "AC_Set_Temp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Drop rows with missing values.
df = df.dropna(subset=["OutdoorTemp", "EnergyCost", "Brightness", "AC_Set_Temp"]).reset_index(drop=True)

# Compute time-based features from Timestamp.
df["Hour"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# One-hot encode DayType (assume five categories: work, weekend, vacation, party, work-from-home)
daytype_dummies = pd.get_dummies(df["DayType"], prefix="DayType")
df = pd.concat([df, daytype_dummies], axis=1)

# Define feature columns.
# Features: OutdoorTemp, EnergyCost, Hour_sin, Hour_cos, one-hot DayType, plus past Brightness and AC_Set_Temp.
feature_cols = ["OutdoorTemp", "EnergyCost", "Hour_sin", "Hour_cos"] + list(daytype_dummies.columns) + ["Brightness", "AC_Set_Temp"]
target_cols = ["Brightness", "AC_Set_Temp"]
num_features = len(feature_cols)

# ================================================================
# 2. Create Sliding Windows for Sequence Data (Same as TensorFlow version)
# ================================================================
print("Creating sliding windows...")
X_windows = []
y_windows = []

# Convert to numpy array. This should now be numeric.
data = df[feature_cols].values  # shape (N, num_features)

# Create windows (forecast one step ahead from the window)
for i in range(len(data) - T):
    X_windows.append(data[i:i+T, :])
    y_windows.append(data[i+T, -2:])  # last two columns: [Brightness, AC_Set_Temp]

# Convert to numpy arrays with type float32.
X_windows = np.array(X_windows, dtype=np.float32)  # shape (num_samples, T, num_features)
y_windows = np.array(y_windows, dtype=np.float32)  # shape (num_samples, 2)

print("X_windows shape:", X_windows.shape)
print("y_windows shape:", y_windows.shape)

# ================================================================
# 3. PyTorch Dataset and DataLoader
# ================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create the full dataset
full_dataset = TimeSeriesDataset(X_windows, y_windows)

# Split into training and testing sets (time-based split)
num_samples = len(full_dataset)
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size

# Use Subset for splitting without shuffling indices
train_indices = list(range(train_size))
test_indices = list(range(train_size, num_samples))

train_subset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Further split training data into training and validation sets
num_train = len(train_subset)
val_size = int(VALIDATION_SPLIT * num_train)
train_size_final = num_train - val_size

# Use random_split for train/validation split (can shuffle here if desired)
# For time series, it's often better to split sequentially, but random_split is common
# If sequential split is needed:
# val_indices = list(range(train_size_final, num_train))
# train_indices_final = list(range(train_size_final))
# train_dataset = Subset(train_subset, train_indices_final)
# val_dataset = Subset(train_subset, val_indices)

train_dataset, val_dataset = random_split(train_subset, [train_size_final, val_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle training data
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================================================================
# 4. Build the High-Capacity Multi-Output LSTM Model (PyTorch)
# ================================================================
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout_prob=0.3):
        super(MultiOutputLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Note: PyTorch LSTM dropout applies between layers, not within recurrent steps like Keras recurrent_dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0) # Dropout only between layers

        # Shared dense layer
        self.shared_dense = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # Branch for brightness
        self.bright_dense = nn.Linear(64, 32)
        self.bright_output = nn.Linear(32, 1)

        # Branch for AC set temperature
        self.ac_dense = nn.Linear(64, 32)
        self.ac_output = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch_size, seq_length, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        # cn shape: (num_layers, batch_size, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the last time step
        out = out[:, -1, :] # shape: (batch_size, hidden_size)

        # Shared dense layers
        shared = self.dropout(self.relu(self.shared_dense(out))) # Apply dropout after activation

        # Brightness branch
        bright = self.relu(self.bright_dense(shared))
        brightness_pred = self.bright_output(bright) # Linear activation is default

        # AC branch
        ac = self.relu(self.ac_dense(shared))
        ac_pred = self.ac_output(ac) # Linear activation is default

        return brightness_pred, ac_pred

model = MultiOutputLSTM(input_size=num_features).to(device)
print(model)

# Count parameters (optional)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# ================================================================
# 5. Loss Function, Optimizer, and Scheduler
# ================================================================
criterion = nn.MSELoss() # Mean Squared Error for both outputs
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR, verbose=True)

# ================================================================
# 6. Training Loop with Validation and Early Stopping
# ================================================================
best_val_loss = float('inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': [], 'train_mae_b': [], 'train_mae_ac': [], 'val_mae_b': [], 'val_mae_ac': []}

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_mae_b = 0.0
    running_mae_ac = 0.0
    train_samples_count = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        target_b = targets[:, 0].unsqueeze(1) # Shape (batch_size, 1)
        target_ac = targets[:, 1].unsqueeze(1) # Shape (batch_size, 1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        pred_b, pred_ac = model(inputs)

        # Calculate loss
        loss_b = criterion(pred_b, target_b)
        loss_ac = criterion(pred_ac, target_ac)
        loss = loss_b + loss_ac # Combine losses (simple sum, could be weighted)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        running_mae_b += torch.abs(pred_b - target_b).sum().item()
        running_mae_ac += torch.abs(pred_ac - target_ac).sum().item()
        train_samples_count += inputs.size(0)

    epoch_loss = running_loss / train_samples_count
    epoch_mae_b = running_mae_b / train_samples_count
    epoch_mae_ac = running_mae_ac / train_samples_count
    history['train_loss'].append(epoch_loss)
    history['train_mae_b'].append(epoch_mae_b)
    history['train_mae_ac'].append(epoch_mae_ac)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_mae_b = 0.0
    val_mae_ac = 0.0
    val_samples_count = 0
    with torch.no_grad(): # Disable gradient calculation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            target_b = targets[:, 0].unsqueeze(1)
            target_ac = targets[:, 1].unsqueeze(1)

            pred_b, pred_ac = model(inputs)
            loss_b = criterion(pred_b, target_b)
            loss_ac = criterion(pred_ac, target_ac)
            loss = loss_b + loss_ac

            val_loss += loss.item() * inputs.size(0)
            val_mae_b += torch.abs(pred_b - target_b).sum().item()
            val_mae_ac += torch.abs(pred_ac - target_ac).sum().item()
            val_samples_count += inputs.size(0)

    epoch_val_loss = val_loss / val_samples_count
    epoch_val_mae_b = val_mae_b / val_samples_count
    epoch_val_mae_ac = val_mae_ac / val_samples_count
    history['val_loss'].append(epoch_val_loss)
    history['val_mae_b'].append(epoch_val_mae_b)
    history['val_mae_ac'].append(epoch_val_mae_ac)

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{EPOCHS} [{epoch_duration:.2f}s] - "
          f"Loss: {epoch_loss:.4f}, MAE_B: {epoch_mae_b:.4f}, MAE_AC: {epoch_mae_ac:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val MAE_B: {epoch_val_mae_b:.4f}, Val MAE_AC: {epoch_val_mae_ac:.4f}")

    # Learning rate scheduler step
    scheduler.step(epoch_val_loss)

    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), "multi_output_forecast_model_pytorch_best.pth")
        print(f"Validation loss improved. Saved best model to multi_output_forecast_model_pytorch_best.pth")
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

total_training_time = time.time() - start_time
print(f"Training finished in {total_training_time:.2f} seconds.")

# Load the best model weights for evaluation
print("Loading best model weights for final evaluation...")
model.load_state_dict(torch.load("multi_output_forecast_model_pytorch_best.pth"))

# ================================================================
# 7. Evaluation on Test Set
# ================================================================
print("Evaluating on test set...")
model.eval()
test_loss = 0.0
test_mae_b = 0.0
test_mae_ac = 0.0
test_samples_count = 0
all_preds_b = []
all_preds_ac = []
all_targets_b = []
all_targets_ac = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        target_b = targets[:, 0].unsqueeze(1)
        target_ac = targets[:, 1].unsqueeze(1)

        pred_b, pred_ac = model(inputs)
        loss_b = criterion(pred_b, target_b)
        loss_ac = criterion(pred_ac, target_ac)
        loss = loss_b + loss_ac

        test_loss += loss.item() * inputs.size(0)
        test_mae_b += torch.abs(pred_b - target_b).sum().item()
        test_mae_ac += torch.abs(pred_ac - target_ac).sum().item()
        test_samples_count += inputs.size(0)

        # Store predictions and targets for inspection
        all_preds_b.append(pred_b.cpu().numpy())
        all_preds_ac.append(pred_ac.cpu().numpy())
        all_targets_b.append(target_b.cpu().numpy())
        all_targets_ac.append(target_ac.cpu().numpy())

final_test_loss = test_loss / test_samples_count
final_test_mae_b = test_mae_b / test_samples_count
final_test_mae_ac = test_mae_ac / test_samples_count

print(f"Test Loss: {final_test_loss:.4f}")
print(f"Brightness MAE: {final_test_mae_b:.4f}")
print(f"AC Set Temperature MAE: {final_test_mae_ac:.4f}")

# Concatenate predictions and targets from all batches
pred_brightness = np.concatenate(all_preds_b).flatten()
pred_ac = np.concatenate(all_preds_ac).flatten()
true_brightness = np.concatenate(all_targets_b).flatten()
true_ac = np.concatenate(all_targets_ac).flatten()

print("\nFirst 10 predicted Brightness:", pred_brightness[:10])
print("First 10 true Brightness:", true_brightness[:10])
print("First 10 predicted AC_Set_Temp:", pred_ac[:10])
print("First 10 true AC_Set_Temp:", true_ac[:10])

# ================================================================
# 8. Save the Final Trained Model (Optional - best is already saved)
# ================================================================
# You might want to save the model after the full training run,
# even if early stopping occurred, or just rely on the 'best' saved model.
# torch.save(model.state_dict(), "multi_output_forecast_model_pytorch_final.pth")
# print("Final model state dict saved as 'multi_output_forecast_model_pytorch_final.pth'")

# ================================================================
# 9. Training and Testing Guidelines (Comments - Adapted for PyTorch)
# ================================================================
"""
Guidelines for further optimizing and productionizing this PyTorch model:

1. Data Preparation & Windowing:
   - Ensure that scaling is applied consistently (consider using StandardScaler from scikit-learn, fit on training data only).
   - Experiment with different window lengths (T).

2. Model Architecture & Capacity:
   - Adjust hidden_size, num_layers, dropout_prob.
   - Consider nn.LSTM's `bidirectional=True` argument.
   - Explore attention mechanisms (e.g., implement a separate attention layer).

3. Loss Functions and Multi-Task Learning:
   - If outputs have different scales/importance, weight the losses: `loss = w1 * loss_b + w2 * loss_ac`.
   - Explore advanced multi-task techniques if needed.

4. Validation and Evaluation:
   - Ensure the validation split respects the time-series nature if strict temporal order is critical (manual Subset splitting).
   - Monitor MAE, RMSE, etc.

5. Regularization and Training Tricks:
   - Tune dropout probability. Add weight decay (L2 regularization) in the optimizer: `optim.Adam(..., weight_decay=1e-5)`.
   - Experiment with different optimizers (e.g., AdamW).
   - Gradient clipping can help stabilize LSTM training: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

6. Advanced Extensions:
   - Implement attention mechanisms.
   - Build an ensemble of models.
   - Explore Transformer-based architectures using PyTorch implementations.

7. Deployment:
   - Save the model's state_dict (`torch.save(model.state_dict(), PATH)`). For deployment, you might need TorchScript (`torch.jit.script` or `torch.jit.trace`) or ONNX export.
   - Implement a retraining pipeline.
""" 