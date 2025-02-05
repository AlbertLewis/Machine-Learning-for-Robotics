import torch
import torch.nn as nn
class MSELoss(nn.Module):
   """Standard Mean Squared Error loss implementation.
   
   Methods:
       forward(predictions, targets): Compute MSE loss between predictions and targets
       
   Example:
       criterion = MSELoss()
       loss = criterion(model_output, target)
   """
   
   def forward(self, predictions, targets):
       """Compute MSE loss between predictions and targets.
       
       Args:
           predictions (torch.Tensor): Model predictions
           targets (torch.Tensor): Target values
           
       Returns:
           torch.Tensor: Scalar MSE loss value
       """
       squared_error = (predictions - targets) ** 2
       return torch.mean(squared_error)
   

class CustomLoss(nn.Module):
   """Custom loss for robotic pose prediction with separate position and rotation terms.
   
   Methods:
       forward(predictions, targets): Compute weighted sum of position and rotation losses
       
   Args:
       position_weight (float): Weight for position loss term
       rotation_weight (float): Weight for rotation loss term
       
   Example:
       criterion = CustomLoss(position_weight=1.0, rotation_weight=0.5)
       loss = criterion(model_output, target)
   """
   
   def __init__(self, position_weight, rotation_weight):
       super(CustomLoss, self).__init__()
       self.position_weight = position_weight
       self.rotation_weight = rotation_weight
       self.mse = nn.MSELoss()
   
   def forward(self, predictions, targets):
       """Compute weighted position and rotation loss.
       
       Args:
           predictions (torch.Tensor): Model predictions with position (x,y,z) 
               and rotation (rx,ry,rz) components
           targets (torch.Tensor): Target poses with position and rotation
           
       Returns:
           torch.Tensor: Weighted sum of position and rotation MSE losses
       """
       loss_pos = self.mse(predictions[:, :3], targets[:, :3])
       loss_rot = self.mse(predictions[:, 3:], targets[:, 3:])

       return self.position_weight * loss_pos + self.rotation_weight * loss_rot