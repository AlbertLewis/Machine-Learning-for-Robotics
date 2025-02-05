import numpy as np

def compute_mse(predictions, targets):
   """Compute Mean Squared Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions
       targets (np.ndarray): Target values
       
   Returns:
       float: MSE value
   """

   return ((targets - predictions)**2).mean()

def compute_rmse(predictions, targets): 
   """Compute Root Mean Squared Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions
       targets (np.ndarray): Target values
       
   Returns:
       float: RMSE value
   """
   pass

def compute_mae(predictions, targets):
   """Compute Mean Absolute Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions  
       targets (np.ndarray): Target values
       
   Returns:
       float: MAE value
   """
   pass

def compute_position_error(predictions, targets):
   """Compute mean Euclidean error for position predictions (x,y,z).
   
   Args:
       predictions (np.ndarray): Model predictions with position in first 3 columns
       targets (np.ndarray): Target values with position in first 3 columns
       
   Returns:
       float: Mean position error
   """
   diff = (predictions[:, :3] - targets[:, :3])

   sum = 0
   for col in diff:
    sum += (col[0]**2 + col[1]**2 + col[2]**2)**0.5

   return sum / diff.shape[1]
   
def compute_rotation_error(predictions, targets):
   """Compute mean Euclidean error for rotation predictions (rx,ry,rz).
   
   Args:
       predictions (np.ndarray): Model predictions with rotation in last 3 columns
       targets (np.ndarray): Target values with rotation in last 3 columns
       
   Returns:
       float: Mean rotation error
   """
   diff = (predictions[:, 3:] - targets[:, 3:])

   sum = 0
   for col in diff:
      sum += (col[0]**2 + col[1]**2 + col[2]**2)**0.5

   return sum / diff.shape[1]