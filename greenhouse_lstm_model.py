# ============================================================
# Greenhouse Digital Twin Framework
# LSTM Model Development and Scenario Simulation
# Author: Ashwini Bingi | Student ID: 16380309
# Module: 7150CEM Data Science Project
# Coventry University | April 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1 - DATA LOADING AND PREPROCESSING
# ============================================================

# Load greenhouse IoT sensor data
greenhouse_df = pd.read_csv('greenhouse_dataset_Shrivan.csv', parse_dates=['datetime'])
greenhouse_df.set_index('datetime', inplace=True)

# Load external Meteostat weather data
weather_df = pd.read_csv('shirvan_weather_hourly_station_2023-11_to_2024-04.csv', 
                          parse_dates=['datetime'])
weather_df.set_index('datetime', inplace=True)

# Merge both datasets on hourly timestamp
merged_df = greenhouse_df.merge(weather_df, left_index=True, 
                                 right_index=True, how='inner')

# Handle missing values using forward fill
merged_df = merged_df.fillna(method='ffill')

# Drop any remaining missing values
merged_df.dropna(inplace=True)

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")

# ============================================================
# SECTION 2 - FEATURE ENGINEERING
# ============================================================

# Cyclic time encoding for hour of day
merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df.index.hour / 24)
merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df.index.hour / 24)

# Cyclic time encoding for day of week
merged_df['dow_sin'] = np.sin(2 * np.pi * merged_df.index.dayofweek / 7)
merged_df['dow_cos'] = np.cos(2 * np.pi * merged_df.index.dayofweek / 7)

# Define 12 input features and target variable
features = [
    'AVGHumidityInside', 'OutsideTemp', 'HumidityOutside',
    'temp', 'rhum', 'wspd', 'pres', 'prcp',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
target = 'MeanInsideTemp'

X = merged_df[features].values
y = merged_df[target].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================
# SECTION 3 - CHRONOLOGICAL DATA SPLIT
# ============================================================

total_rows = len(X)
train_end = int(total_rows * 0.70)
val_end = int(total_rows * 0.85)

X_train_raw = X[:train_end]
X_val_raw = X[train_end:val_end]
X_test_raw = X[val_end:]

y_train_raw = y[:train_end]
y_val_raw = y[train_end:val_end]
y_test_raw = y[val_end:]

print(f"Training rows: {len(X_train_raw)}")
print(f"Validation rows: {len(X_val_raw)}")
print(f"Test rows: {len(X_test_raw)}")

# ============================================================
# SECTION 4 - STANDARD SCALER NORMALISATION
# Fitted on training data only to prevent data leakage
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ============================================================
# SECTION 5 - SEQUENCE CONSTRUCTION
# Lookback window: 72 hours | Forecast horizon: 168 hours
# ============================================================

LOOKBACK = 72
HORIZON = 168

def create_sequences(X, y, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback:i + lookback + horizon])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(X_train_scaled, y_train_raw, LOOKBACK, HORIZON)
X_val, y_val = create_sequences(X_val_scaled, y_val_raw, LOOKBACK, HORIZON)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, LOOKBACK, HORIZON)

print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

# ============================================================
# SECTION 6 - LSTM MODEL ARCHITECTURE
# ============================================================

model = Sequential([
    LSTM(128, input_shape=(LOOKBACK, len(features))),
    Dense(64, activation='relu'),
    Dense(HORIZON)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ============================================================
# SECTION 7 - MODEL TRAINING
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# SECTION 8 - MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

# Flatten for metric calculation
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print(f"\nModel Evaluation Results:")
print(f"MAE:  {mae:.4f} degrees C")
print(f"RMSE: {rmse:.4f} degrees C")
print(f"R2:   {r2:.4f}")

# ============================================================
# SECTION 9 - SIMULINK SCENARIO SIMULATION
# First-order transfer function: G(s) = 1 / (s + 1)
# ============================================================

from scipy.signal import lti, step

# Define first-order transfer function
numerator = [1]
denominator = [1, 1]
system = lti(numerator, denominator)

# Time array - 10 hours simulation
t = np.linspace(0, 10, 1000)

# Scenario 1 - Standard Heating
# Two step inputs simulating heating phases
def two_step_input(t, step1_time=0, step1_amp=25, 
                    step2_time=5, step2_amp=10):
    u = np.zeros_like(t)
    u[t >= step1_time] += step1_amp
    u[t >= step2_time] += step2_amp
    return u

# Scenario 2 - Heatwave
# Continuous elevated input
def heatwave_input(t, amplitude=40):
    return np.ones_like(t) * amplitude

# Scenario 3 - Maximum Temperature
# Two step inputs with higher second step
def max_temp_input(t, step1_amp=25, step2_amp=15):
    u = np.zeros_like(t)
    u[t >= 0] += step1_amp
    u[t >= 5] += step2_amp
    return u

# Scenario 4 - Minimum Temperature
# Step up then step down
def min_temp_input(t, rise_amp=15, fall_time=3):
    u = np.zeros_like(t)
    u[t >= 0] += rise_amp
    u[t >= fall_time] -= rise_amp
    return u

# Run simulations using lsim
t_out1, y_out1, _ = system.output(two_step_input(t), t)
t_out2, y_out2, _ = system.output(heatwave_input(t), t)
t_out3, y_out3, _ = system.output(max_temp_input(t), t)
t_out4, y_out4, _ = system.output(min_temp_input(t), t)

# Plot scenario results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(t_out1, y_out1, 'y-', linewidth=2)
axes[0,0].axhline(y=35, color='red', linestyle='--', label='Stress 35C')
axes[0,0].axhline(y=30, color='gray', linestyle='--', label='Warning 30C')
axes[0,0].set_title('Heating Simulation')
axes[0,0].set_xlabel('Time (hours)')
axes[0,0].set_ylabel('Temperature (C)')
axes[0,0].legend()

axes[0,1].plot(t_out2, y_out2, 'y-', linewidth=2)
axes[0,1].axhline(y=35, color='red', linestyle='--', label='Stress 35C')
axes[0,1].axhline(y=30, color='gray', linestyle='--', label='Warning 30C')
axes[0,1].set_title('Heatwave Simulation')
axes[0,1].set_xlabel('Time (hours)')
axes[0,1].set_ylabel('Temperature (C)')
axes[0,1].legend()

axes[1,0].plot(t_out3, y_out3, 'y-', linewidth=2)
axes[1,0].set_title('Maximum Temperature Simulation')
axes[1,0].set_xlabel('Time (hours)')
axes[1,0].set_ylabel('Temperature (C)')

axes[1,1].plot(t_out4, y_out4, 'y-', linewidth=2)
axes[1,1].set_title('Minimum Temperature Simulation')
axes[1,1].set_xlabel('Time (hours)')
axes[1,1].set_ylabel('Temperature (C)')

plt.tight_layout()
plt.savefig('scenario_simulations.png', dpi=300)
plt.show()

print("Simulation complete")
