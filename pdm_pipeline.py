import os
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
CSV_HISTORY          = "k48_history.csv"
LIVE_XLSX            = "live_feed.xlsx"
HISTOGRAM_FILE       = "parameters_importance.jpg"
FEATURE_CSV          = "feature_importance.csv"

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

# Train model and capture history
print("Training LSTM for 500 epochs with full batch size...")
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    batch_size=len(X_train),
    callbacks=[es],
    verbose=1
)

# Determine optimal epoch
optimal_epoch = es.stopped_epoch if es.stopped_epoch > 0 else 500

# Plot optimization graph: Epochs vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='orange')
plt.axvline(x=optimal_epoch, color='red', linestyle='--', label=f'Opt EPOCH={optimal_epoch}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.yticks(np.arange(0.72, 1.05, 0.05))
plt.xticks(np.arange(0, 501, 100))
plt.title('Epoch vs. Accuracy')
plt.legend(loc='upper right')
plt.grid(True, color='lightgray')
plt.show()

# Create a clean model for gradient calculations
def create_clean_model():
    clean_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    clean_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    clean_model.set_weights(model.get_weights())
    return clean_model

# Initialize clean model for gradient calculations
gradient_model = create_clean_model()

def calculate_feature_importance_simple(X_sample):
    feature_variance = np.var(X_sample[0], axis=0)
    feature_magnitude = np.mean(np.abs(X_sample[0]), axis=0)
    importance = (feature_variance + feature_magnitude) / 2
    return importance

def calculate_permutation_importance(model, X_sample, baseline_pred, n_repeats=5):
    importances = []
    for feature_idx in range(len(features)):
        feature_importance = []
        for _ in range(n_repeats):
            X_permuted = X_sample.copy()
            feature_values = X_permuted[0, :, feature_idx].copy()
            np.random.shuffle(feature_values)
            X_permuted[0, :, feature_idx] = feature_values
            permuted_pred = float(model.predict(X_permuted, verbose=0)[0][0])
            importance = abs(baseline_pred - permuted_pred)
            feature_importance.append(importance)
        importances.append(np.mean(feature_importance))
    return np.array(importances)

def analyze_feature_contributions(X_sample, prediction):
    try:
        simple_importance = calculate_feature_importance_simple(X_sample)
        perm_importance   = calculate_permutation_importance(gradient_model, X_sample, prediction)
        combined_importance = 0.3 * simple_importance + 0.7 * perm_importance
        combined_importance = combined_importance / np.sum(combined_importance)
        return combined_importance, simple_importance, perm_importance
    except Exception as e:
        print(f"Feature importance calculation failed: {e}")
        return np.ones(len(features)) / len(features), None, None

# Historical data for anomaly detection
recent_predictions = []
recent_features    = []

# Start real-time prediction loop
live_window = list(X_test[-1])
print("\nStarting live generation & PdM loop. Press Ctrl+C to stop.")

iteration = 0
while True:
    try:
        if random.random() < 0.1:
            new_scaled   = np.array([1.2, 0.9, 1.5, 1.2, 1.3])
            anomaly_type = "High vibration/current anomaly detected"
        else:
            noise        = np.random.normal(0, 0.02, size=(len(features),))
            new_scaled   = live_window[-1] * (1 + noise)
            new_scaled   = np.clip(new_scaled, 0, 2)
            anomaly_type = None

        live_window.append(new_scaled)
        if len(live_window) > seq_len:
            live_window.pop(0)

        X_live       = np.expand_dims(np.array(live_window), axis=0)
        failure_prob = float(model.predict(X_live, verbose=0)[0][0])
        alert        = int(failure_prob > 0.5)

        recent_predictions.append(failure_prob)
        recent_features.append(new_scaled.copy())
        if len(recent_predictions) > 20:
            recent_predictions.pop(0)
            recent_features.pop(0)

        if iteration % 3 == 0:
            print("Calculating feature importance...")
            combined_imp, simple_imp, perm_imp = analyze_feature_contributions(X_live, failure_prob)
            df_importance = pd.DataFrame({
                'Feature': features,
                'Importance': combined_imp,
                'Rank': range(1, len(features) + 1)
            }).sort_values('Importance', ascending=False)
            df_importance['Rank'] = range(1, len(features) + 1)
            print("Feature Importance Analysis:")
            print(df_importance.round(4))
            top_features = df_importance.head(2)['Feature'].tolist()
            print(f"Most critical features: {', '.join(top_features)}")

            # ===== Begin Added Block =====
            # delete old histogram if present
            if os.path.exists(HISTOGRAM_FILE):
                os.remove(HISTOGRAM_FILE)

            # plot & save new histogram
            plt.figure(figsize=(8, 5))
            plt.bar(df_importance['Feature'], df_importance['Importance'])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(HISTOGRAM_FILE)
            plt.close()

            # print full ranking
            print("\nFull feature ranking:")
            for _, row in df_importance.iterrows():
                print(f"  {int(row['Rank']):>d}. {row['Feature']:<15} — {row['Importance']:.4f}")
            print(f"Saved updated histogram ➔ {HISTOGRAM_FILE}")

            # append percentages to CSV
            record = {'timestamp': datetime.now().strftime("%d/%m/%y %H:%M")}
            for feat, val in zip(df_importance['Feature'], df_importance['Importance']):
                record[feat] = val
            df_rec = pd.DataFrame([record])
            header = not os.path.exists(FEATURE_CSV)
            df_rec.to_csv(FEATURE_CSV, mode='a', index=False, header=header)
            print(f"Appended feature percentages to ➔ {FEATURE_CSV}")
            # ===== End Added Block =====

        if len(recent_predictions) >= 5:
            recent_trend = (
                np.mean(recent_predictions[-5:])
                - np.mean(recent_predictions[-10:-5])
                if len(recent_predictions) >= 10
                else 0
            )
            trend_direction = "Rising" if recent_trend > 0.01 else "Falling" if recent_trend < -0.01 else "Stable"
        else:
            trend_direction = "Insufficient data"

        real_vals = scaler.inverse_transform([new_scaled])[0]
        record    = dict(zip(features, real_vals))
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        record.update({
            'timestamp': timestamp,
            'failure_prob': round(failure_prob, 3),
            'alert': alert,
            'trend': trend_direction
        })

        try:
            df_out = pd.DataFrame([record])
            df_out = df_out[['timestamp'] + features + ['failure_prob', 'alert', 'trend']]
            df_out.to_excel(LIVE_XLSX, index=False)
        except Exception as e:
            print(f"Error writing to Excel: {e}")

        status_color = "🔴" if alert else "🟡" if failure_prob > 0.3 else "🟢"
        print(f"\n{status_color} {timestamp} | FailureProb: {failure_prob:.3f} | Alert: {alert} | Trend: {trend_direction}")
        if anomaly_type:
            print(f"⚠️  ANOMALY: {anomaly_type}")

        sensor_status = []
        for feat, val in zip(features, real_vals):
            if feat == 'vibration' and val > 0.5:
                status = "⚠️ HIGH"
            elif feat == 'temperature' and val > 80:
                status = "⚠️ HIGH"
            elif feat == 'motor_current' and val > 12:
                status = "⚠️ HIGH"
            elif feat == 'rpm' and val > 28000:
                status = "⚠️ HIGH"
            else:
                status = "✓"
            sensor_status.append(f"{feat}: {val:.2f} {status}")
        print("Sensors: " + " | ".join(sensor_status))

        confidence = (
            "High" if abs(failure_prob - 0.5) > 0.3
            else "Medium" if abs(failure_prob - 0.5) > 0.1
            else "Low"
        )
        print(f"Prediction confidence: {confidence}")
        print("=" * 100)

        iteration += 1
        time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Predictive maintenance loop stopped by user.")
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
        time.sleep(5)
