import os
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================================================================
# 1. Data Loading and Preprocessing
# ================================================================
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

# ================================================================
# 2. Create Sliding Windows for Sequence Data
# ================================================================
# Use a window of T timesteps (e.g. T = 48, which is 4 hours at 5-minute intervals)
T = 48
X_windows = []
y_windows = []

# Convert to numpy array. This should now be numeric.
data = df[feature_cols].values  # shape (N, 11)

# Create windows (forecast one step ahead from the window)
for i in range(len(data) - T):
    X_windows.append(data[i:i+T, :])
    y_windows.append(data[i+T, -2:])  # last two columns: [Brightness, AC_Set_Temp]

# Convert to numpy arrays with type float32.
X_windows = np.array(X_windows, dtype=np.float32)  # shape (num_samples, T, 11)
y_windows = np.array(y_windows, dtype=np.float32)  # shape (num_samples, 2)

print("X_windows shape:", X_windows.shape)
print("y_windows shape:", y_windows.shape)

# ================================================================
# 3. Train/Test Split (Time-based Split)
# ================================================================
num_samples = X_windows.shape[0]
train_size = int(0.8 * num_samples)
X_train = X_windows[:train_size]
y_train = y_windows[:train_size]
X_test = X_windows[train_size:]
y_test = y_windows[train_size:]
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Optionally, convert to tensors.
X_train_tf = tf.constant(X_train, dtype=tf.float32)
y_train_tf_b = tf.constant(y_train[:, 0], dtype=tf.float32)
y_train_tf_ac = tf.constant(y_train[:, 1], dtype=tf.float32)
X_test_tf = tf.constant(X_test, dtype=tf.float32)
y_test_tf_b = tf.constant(y_test[:, 0], dtype=tf.float32)
y_test_tf_ac = tf.constant(y_test[:, 1], dtype=tf.float32)

# ================================================================
# 4. Build the High-Capacity Multi-Output LSTM Model
# ================================================================
input_seq = Input(shape=(T, X_windows.shape[2]), name="input_sequence")

# Stacked LSTM layers with dropout.
x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="lstm_1")(input_seq)
x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="lstm_2")(x)
x = LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, name="lstm_3")(x)

# Shared dense layer.
shared = Dense(64, activation="relu", name="shared_dense")(x)
shared = Dropout(0.3)(shared)

# Branch for brightness.
brightness_branch = Dense(32, activation="relu", name="bright_dense")(shared)
brightness_output = Dense(1, activation="linear", name="brightness")(brightness_branch)

# Branch for AC set temperature.
ac_branch = Dense(32, activation="relu", name="ac_dense")(shared)
ac_output = Dense(1, activation="linear", name="ac")(ac_branch)

model = Model(inputs=input_seq, outputs=[brightness_output, ac_output])
model.summary()

# Compile the model with MSE loss for both outputs.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={"brightness": "mse", "ac": "mse"},
              metrics={"brightness": "mae", "ac": "mae"})

# ================================================================
# 5. Training Guidelines and Callbacks
# ================================================================
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

# Train the model using tensors. You can also use the numpy arrays directly if they are float32.
history = model.fit(X_train_tf, {"brightness": y_train_tf_b, "ac": y_train_tf_ac},
                    validation_split=0.1,
                    epochs=100,
                    batch_size=64,
                    callbacks=[early_stop, reduce_lr])

# ================================================================
# 6. Evaluation and Testing
# ================================================================
test_loss, test_loss_b, test_loss_ac, test_mae_b, test_mae_ac = model.evaluate(X_test_tf, {"brightness": y_test_tf_b, "ac": y_test_tf_ac})
print("Test Loss:", test_loss)
print("Brightness MAE:", test_mae_b)
print("AC Set Temperature MAE:", test_mae_ac)

# Make predictions on the test set.
predictions = model.predict(X_test_tf)
pred_brightness = predictions[0].flatten()
pred_ac = predictions[1].flatten()

print("First 10 predicted Brightness:", pred_brightness[:10])
print("First 10 true Brightness:", y_test_tf_b.numpy()[:10])
print("First 10 predicted AC_Set_Temp:", pred_ac[:10])
print("First 10 true AC_Set_Temp:", y_test_tf_ac.numpy()[:10])

# ================================================================
# 7. Training and Testing Guidelines (Comments)
# ================================================================
"""
Guidelines for further optimizing and productionizing this model:

1. Data Preparation & Windowing:
   - Ensure that scaling is applied consistently (consider using StandardScaler on the training set and applying to test).
   - Experiment with different window lengths (T = 12, 24, 48) to capture optimal historical context.

2. Model Architecture & Capacity:
   - The model uses 3 LSTM layers (128 units each) with dropout; adjust layers/units if underfitting.
   - Consider using bidirectional LSTMs or integrating an attention mechanism after the LSTM layers.

3. Loss Functions and Multi-Task Learning:
   - If the outputs are on different scales, consider weighting the losses differently.
   - Experiment with advanced multi-task techniques such as uncertainty weighting.

4. Validation and Evaluation:
   - Use time-series aware splits (e.g., walk-forward validation) to avoid lookahead bias.
   - Monitor metrics (MAE, RMSE) for both outputs. Consider using normalized error metrics if necessary.

5. Regularization and Training Tricks:
   - Utilize early stopping, dropout, and L2 regularization to reduce overfitting.
   - Data augmentation (e.g., slight input noise) might improve robustness.

6. Advanced Extensions:
   - Add an attention mechanism to focus on the most relevant past timesteps.
   - Build an ensemble of models to reduce variance.
   - Experiment with Transformer-based architectures (e.g., Temporal Fusion Transformer) for further improvements.

7. Deployment:
   - Export the trained model using TensorFlow SavedModel or TensorFlow Serving for real-time forecasting.
   - Implement a retraining schedule with new data to keep the model updated.
"""

# ================================================================
# 8. Save the Trained Model
# ================================================================
model.save("multi_output_forecast_model.h5")
print("Model saved as 'multi_output_forecast_model.h5'")
