import numpy as np
import pandas as pd


def LoadAndCleanData(csv_path):
  df = pd.read_csv(csv_path)
  df = df.drop_duplicates().reset_index(drop=True)

  features = ['cp', 'thalach', 'ca', 'oldpeak', 'exang', 'thal']
  X = df[features].values
  y = df['target'].values
  return X, y


def TrainTestSplit(X, y, test_ratio=0.2, val_ratio=0.1, seed=42):
  np.random.seed(seed)
  indices = np.arange(len(X))
  np.random.shuffle(indices)

  X = X[indices]
  y = y[indices]

  n_test = int(len(X) * test_ratio)
  n_val = int(len(X) * val_ratio)

  X_test, y_test = X[:n_test], y[:n_test]
  X_val, y_val = X[n_test : n_test + n_val], y[n_test : n_test + n_val]
  X_train, y_train = X[n_test + n_val :], y[n_test + n_val :]

  return X_train, y_train, X_val, y_val, X_test, y_test


def StandardizeFeatures(X_train, X_val, X_test):
  mean = np.mean(X_train, axis=0)
  std = np.std(X_train, axis=0)
  std[std == 0] = 1.0

  X_train_scaled = (X_train - mean) / std
  X_val_scaled = (X_val - mean) / std
  X_test_scaled = (X_test - mean) / std

  X_train_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
  X_val_bias = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]
  X_test_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

  return X_train_bias, X_val_bias, X_test_bias
