import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from save_load import save
import os

def datagen():
    # Ensure the save directory exists
    save_dir = './Saved data'
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv("Dataset\\cloud_resource_allocation_dataset.csv")

    # Encode Workload_Type
    le = LabelEncoder()
    df['Workload_Type_Encoded'] = le.fit_transform(df['Workload_Type'])
    df.drop(columns=['Workload_Type'], inplace=True)

    # Features and target
    X_raw = df.drop(columns=['Predicted_Workload (%)', 'Optimized_Resource_Allocation'])
    y_workload = df['Predicted_Workload (%)']

    # Sliding window parameters
    window_size = 10  # Adjust as needed
    lag_k = 1

    features = []

    for i in range(window_size, len(df)):
        window = X_raw.iloc[i - window_size:i]
        feat_vec = []

        for col in window.columns:
            if pd.api.types.is_numeric_dtype(window[col]):
                data = window[col].values

                # Mean
                mean_val = np.mean(data)

                # Variance
                var_val = np.var(data)

                # Skewness
                skew_val = skew(data)

                # Entropy
                counts = pd.Series(data).value_counts()
                prob = counts / counts.sum()
                entropy_val = entropy(prob)

                # Autocorrelation at lag k + Trend slope
                if np.std(data) != 0:
                    autocorr_val = pd.Series(data).autocorr(lag=lag_k)
                    X_idx = np.arange(len(data)).reshape(-1, 1)
                    model = LinearRegression().fit(X_idx, data)
                    slope_val = model.coef_[0]
                else:
                    autocorr_val = 0
                    slope_val = 0

                # Append all features
                feat_vec.extend([mean_val, var_val, skew_val, entropy_val, autocorr_val, slope_val])

        features.append(feat_vec)

    features = np.array(features)

    # Replace NaN and Inf with 0
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    y_aligned = y_workload.iloc[window_size:].values

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    train_data, test_data, train_lab, test_lab = [], [], [], []

    # Split and save
    learning_rates = [0.7, 0.8]  # Train sizes
    for i in range(2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_aligned, test_size=learning_rates[i], random_state=42
        )
        train_data.append(X_train)
        test_data.append(X_test)
        train_lab.append(y_train)
        test_lab.append(y_test)

    save('X_train', train_data)
    save('X_test', test_data)
    save('y_train', train_lab)
    save('y_test', test_lab)

# Run
datagen()
