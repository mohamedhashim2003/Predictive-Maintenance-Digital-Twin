import pandas as pd
import numpy as np
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore')

# File paths
CSV_HISTORY = "k48_history.csv"
LIVE_XLSX   = "live_feed.xlsx"

# Load and preprocess data
df = pd.read_csv(CSV_HISTORY)
df = df.interpolate()
features = ['vibration', 'temperature', 'motor_current', 'rpm', 'production_count']

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled, columns=features)

# Prepare sequences
X, y = [], []
seq_len = 10
for i in range(len(df_scaled) - seq_len):
    X.append(df_scaled.iloc[i:i+seq_len].values)
    y.append(df['failure_flag'].iloc[i + seq_len])
X = np.array(X)
y = np.array(y)

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Train model
print("Training LSTM for 500 epochs with full batch size...")
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=len(X_train),
    callbacks=[es],
    verbose=1
)

# Create a clean model for gradient calculations (without SHAP interference)
def create_clean_model():
    """Create a fresh model instance to avoid SHAP gradient registry conflicts"""
    clean_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    clean_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    clean_model.set_weights(model.get_weights())  # Copy trained weights
    return clean_model

# Initialize clean model for gradient calculations
gradient_model = create_clean_model()

def calculate_feature_importance_simple(X_sample):
    """Simple feature importance based on input magnitude and variation"""
    # Calculate variance across time steps for each feature
    feature_variance = np.var(X_sample[0], axis=0)
    
    # Calculate mean absolute values
    feature_magnitude = np.mean(np.abs(X_sample[0]), axis=0)
    
    # Combine variance and magnitude (normalized)
    importance = (feature_variance + feature_magnitude) / 2
    
    return importance

def calculate_permutation_importance(model, X_sample, baseline_pred, n_repeats=5):
    """Calculate feature importance using permutation method"""
    importances = []
    
    for feature_idx in range(len(features)):
        feature_importance = []
        
        for _ in range(n_repeats):
            # Create a copy of the input
            X_permuted = X_sample.copy()
            
            # Shuffle the feature across time steps
            feature_values = X_permuted[0, :, feature_idx].copy()
            np.random.shuffle(feature_values)
            X_permuted[0, :, feature_idx] = feature_values
            
            # Get prediction with permuted feature
            permuted_pred = float(model.predict(X_permuted, verbose=0)[0][0])
            
            # Calculate importance as absolute change in prediction
            importance = abs(baseline_pred - permuted_pred)
            feature_importance.append(importance)
        
        # Average importance across repeats
        importances.append(np.mean(feature_importance))
    
    return np.array(importances)

def analyze_feature_contributions(X_sample, prediction):
    """Analyze feature contributions using multiple methods"""
    try:
        # Method 1: Simple statistical importance
        simple_importance = calculate_feature_importance_simple(X_sample)
        
        # Method 2: Permutation importance
        perm_importance = calculate_permutation_importance(gradient_model, X_sample, prediction)
        
        # Combine both methods (weighted average)
        combined_importance = 0.3 * simple_importance + 0.7 * perm_importance
        
        # Normalize to sum to 1
        combined_importance = combined_importance / np.sum(combined_importance)
        
        return combined_importance, simple_importance, perm_importance
        
    except Exception as e:
        print(f"Feature importance calculation failed: {e}")
        # Fallback: equal importance
        return np.ones(len(features)) / len(features), None, None

# Historical data for anomaly detection
recent_predictions = []
recent_features = []

# Start real-time prediction loop
live_window = list(X_test[-1])
print("\nStarting live generation & PdM loop. Press Ctrl+C to stop.")

iteration = 0
while True:
    try:
        # Generate next data point (with occasional anomaly)
        if random.random() < 0.1:
            # Simulate anomaly
            new_scaled = np.array([1.2, 0.9, 1.5, 1.2, 1.3])
            anomaly_type = "High vibration/current anomaly detected"
        else:
            # Normal operation with noise
            noise = np.random.normal(0, 0.02, size=(len(features),))
            new_scaled = live_window[-1] * (1 + noise)
            # Ensure values stay within reasonable bounds
            new_scaled = np.clip(new_scaled, 0, 2)
            anomaly_type = None

        live_window.append(new_scaled)
        if len(live_window) > seq_len:
            live_window.pop(0)

        # Predict failure probability
        X_live = np.expand_dims(np.array(live_window), axis=0)
        failure_prob = float(model.predict(X_live, verbose=0)[0][0])
        alert = int(failure_prob > 0.5)

        # Store recent data for trend analysis
        recent_predictions.append(failure_prob)
        recent_features.append(new_scaled.copy())
        if len(recent_predictions) > 20:  # Keep last 20 predictions
            recent_predictions.pop(0)
            recent_features.pop(0)

        # Feature importance analysis (every 3 iterations to reduce computational load)
        if iteration % 3 == 0:
            print("Calculating feature importance...")
            combined_imp, simple_imp, perm_imp = analyze_feature_contributions(X_live, failure_prob)
            
            # Create importance DataFrame
            df_importance = pd.DataFrame({
                'Feature': features,
                'Importance': combined_imp,
                'Rank': range(1, len(features) + 1)
            }).sort_values('Importance', ascending=False)
            df_importance['Rank'] = range(1, len(features) + 1)
            
            print("Feature Importance Analysis:")
            print(df_importance.round(4))
            
            # Identify most critical features
            top_features = df_importance.head(2)['Feature'].tolist()
            print(f"Most critical features: {', '.join(top_features)}")

        # Trend analysis
        if len(recent_predictions) >= 5:
            recent_trend = np.mean(recent_predictions[-5:]) - np.mean(recent_predictions[-10:-5]) if len(recent_predictions) >= 10 else 0
            trend_direction = "Rising" if recent_trend > 0.01 else "Falling" if recent_trend < -0.01 else "Stable"
        else:
            trend_direction = "Insufficient data"

        # Inverse scale back to real values
        real_vals = scaler.inverse_transform([new_scaled])[0]
        record = dict(zip(features, real_vals))

        # Timestamp and prediction
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        record.update({
            'timestamp': timestamp,
            'failure_prob': round(failure_prob, 3),
            'alert': alert,
            'trend': trend_direction
        })

        # Write live feed for Tecnomatix
        try:
            df_out = pd.DataFrame([record])
            df_out = df_out[['timestamp'] + features + ['failure_prob', 'alert', 'trend']]
            df_out.to_excel(LIVE_XLSX, index=False)
        except Exception as e:
            print(f"Error writing to Excel: {e}")

        # Console log with enhanced information
        status_color = "🔴" if alert else "🟡" if failure_prob > 0.3 else "🟢"
        print(f"\n{status_color} {timestamp} | FailureProb: {failure_prob:.3f} | Alert: {alert} | Trend: {trend_direction}")
        
        if anomaly_type:
            print(f"⚠️  ANOMALY: {anomaly_type}")
        
        # Show current sensor values with status indicators
        sensor_status = []
        for feat, val in zip(features, real_vals):
            # Simple threshold-based status (adjust thresholds based on your domain knowledge)
            if feat == 'vibration' and val > 1.0:
                status = "⚠️ HIGH"
            elif feat == 'temperature' and val > 80:
                status = "⚠️ HIGH"
            elif feat == 'motor_current' and val > 15:
                status = "⚠️ HIGH"
            else:
                status = "✓"
            sensor_status.append(f"{feat}: {val:.2f} {status}")
        
        print("Sensors: " + " | ".join(sensor_status))
        
        # Show prediction confidence
        confidence = "High" if abs(failure_prob - 0.5) > 0.3 else "Medium" if abs(failure_prob - 0.5) > 0.1 else "Low"
        print(f"Prediction confidence: {confidence}")
        
        print("=" * 100)

        iteration += 1
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\n🛑 Predictive maintenance loop stopped by user.")
        
        # Summary statistics
        if recent_predictions:
            print("\n📊 Session Summary:")
            print(f"Total predictions: {len(recent_predictions)}")
            print(f"Average failure probability: {np.mean(recent_predictions):.3f}")
            print(f"Max failure probability: {np.max(recent_predictions):.3f}")
            print(f"Alerts triggered: {sum(1 for p in recent_predictions if p > 0.5)}")
        break
        
    except Exception as e:
        print(f"❌ Unexpected error in main loop: {e}")
        print("Continuing with next iteration...")
        time.sleep(5)  # Short pause before retrying