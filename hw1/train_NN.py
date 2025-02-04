import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helpers.data_transforms import StandardScaler, convert_to_tensor
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error


class MLP(nn.Module):
   """Multi-Layer Perceptron for robot kinematics prediction.
   
   Args:
       input_size (int): Number of input features (joint angles)
       hidden_sizes (list): List of hidden layer sizes
       output_size (int): Number of outputs (position + rotation)
       
   Example:
       >>> model = MLP(input_size=6, hidden_sizes=[128, 64], output_size=6)
       >>> output = model(input_tensor)
   """
   
   def forward(self, x):
       """Forward pass through network.
       
       Args:
           x (torch.Tensor): Input tensor of shape (batch_size, input_size)
           
       Returns:
           torch.Tensor: Output predictions of shape (batch_size, output_size)
       """
       pass

def train_nn(X_train, X_test, y_train, y_test, hidden_sizes=[128, 64],
            lr=0.001, batch_size=32, epochs=100, device="cpu"):
   """Train neural network model for robot kinematics.
   
   Args:
       X_train, X_test (np.ndarray): Training and test features
       y_train, y_test (np.ndarray): Training and test targets 
       hidden_sizes (list): Hidden layer sizes
       lr (float): Learning rate
       batch_size (int): Mini-batch size
       epochs (int): Number of training epochs
       device (str): Device to train on ('cpu' or 'cuda')
       
   Returns:
       tuple: Trained model, input scaler, output scaler
       
   Example:
       >>> model, in_scaler, out_scaler = train_nn(X_train, X_test, y_train, y_test)
       >>> y_pred = model(X_test_tensor)
   """
   pass




if __name__ == "__main__":
    from datasets import prepare_dataset

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset(
        "ur10dataset.csv"
    )

    # Train model
    model, input_scaler, output_scaler = train_nn(
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
        hidden_sizes=[128, 64],
        lr=0.001,
        epochs=100,
    )
