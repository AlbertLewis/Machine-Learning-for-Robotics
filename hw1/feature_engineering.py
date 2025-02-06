import numpy as np
from datasets import prepare_dataset
from train_linear_regression import SGDLinearRegression
from train_regression import AnalyticalLinearRegression
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error

def engineer_features(angles):
   """Engineer features for robot kinematics based on forward kinematics equations.
   
   Creates trigonometric features from joint angles that better capture the nonlinear 
   relationships in robot forward kinematics.
   
   
   Args:
       angles (np.ndarray): Input joint angles array of shape (n_samples, 6)
       
   Returns:
       np.ndarray: Engineered features array of shape (n_samples, 42)
       
   Example:
       >>> angles = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]) 
       >>> features = engineer_features(angles)
       >>> print(features.shape)
       (1, 42)
   """
   n_samples = angles.shape[0]
   engineered_features = np.zeros((n_samples, 42))

   for i in range(6):
      engineered_features[:, 2*i] = np.sin(angles[:, i]) # sine joint angles
      engineered_features[:, 2*i+1] = np.cos(angles[:, i]) # cosine joint angles

   # Interactions between joint angles, multiplying sin and cos
   idx = 12 
   for i in range(6):
      for j in range(i+1, 6):
         engineered_features[:, idx] = np.sin(angles[:, i]) * np.cos(angles[:, j])
         idx += 1
         engineered_features[:, idx] = np.cos(angles[:, i]) * np.sin(angles[:, j])
         idx += 1

#    print(f"Engineered features: {engineered_features}")

   return engineered_features
   
    
   
if __name__ == "__main__":
   """
   Script to compare performance between raw angles vs engineered features:
   1. Load and preprocess data
   2. Train linear regression with raw joint angles
   3. Train linear regression with engineered trigonometric features  
   4. Compare MSE, position error and rotation error metrics
   """
   
   # Load data
   X_train, X_test, y_train, y_test = prepare_dataset(
       # "robot_kinematics_normalized_dataset.csv"
       "ur10dataset.csv"
   )

   # Convert to numpy
   X_train = X_train.values
   y_train = y_train.values
   X_test = X_test.values
   y_test = y_test.values

   X_train_features = engineer_features(X_train)
   X_test_features = engineer_features(X_test)

#    print(f"xtrain shape {X_train.shape}")
#    print(f"xtest shape {X_test.shape}")
#    print(f"xtrainfeat shape {X_train_features.shape}")
#    print(f"xtestfeat shape {X_test_features.shape}")

#    print(f"Normal X_train: {X_train}")
#    print(f"Normal X_test: {X_test}\n")
#    print(f"Feature X_train: {X_train_features}")
#    print(f"Feature X_test: {X_test_features}")

   # Train model raw joint angles SGD
   model_raw = SGDLinearRegression()
   model_raw.fit(X_train, y_train, batch_size=32, epochs=1000)
   y_pred_raw = model_raw.predict(X_test)

   # Train model engineered features joint angles SGD
   model_features = SGDLinearRegression()
   model_features.fit(X_train_features, y_train, batch_size=32, epochs=1000)
   y_pred_features = model_features.predict(X_test_features)

#    # Train model raw joint angles Analytical
#    model_raw = AnalyticalLinearRegression()
#    model_raw.fit(X_train, y_train)
#    y_pred_raw = model_raw.predict(X_test)

#    # Train model engineered features joint angles Analytical
#    model_features = AnalyticalLinearRegression()
#    model_features.fit(X_train_features, y_train)
#    y_pred_features = model_features.predict(X_test_features)

   # Evaluate raw model
   mse_raw = compute_mse(y_pred_raw, y_test)
   pos_error_raw = compute_position_error(y_pred_raw, y_test)
   rot_error_raw = compute_rotation_error(y_pred_raw, y_test)

   # Evaluate features model
   mse_features = compute_mse(y_pred_features, y_test)
   pos_error_features = compute_position_error(y_pred_features, y_test)
   rot_error_feaures = compute_rotation_error(y_pred_features, y_test)

   # Print model errors raw angles
   print(f"Test Raw MSE: {mse_raw:.4f}")
   print(f"Raw Position Error: {pos_error_raw:.4f}")
   print(f"Raw Rotation Error: {rot_error_raw:.4f}\n")
   
   # Print model errors feature angles
   print(f"Test Feature MSE: {mse_features:.4f}")
   print(f"Position Feature Error: {pos_error_features:.4f}")
   print(f"Rotation Feature Error: {rot_error_feaures:.4f}")