import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from helpers.data_transforms import StandardScaler, convert_to_tensor
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error, compute_rmse, compute_mae
from feature_engineering import engineer_features
import matplotlib.pyplot as plt

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

   def __init__(self, hidden_sizes=[10, 10], input_size=42, output_size=6):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_sizes[0])
    self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    # Optional: Define dropout or batch normalization for better training
    # self.dropout = nn.Dropout(0.3)  # Dropout for regularization

   def forward(self, x):
       """Forward pass through network.
       
       Args:
           x (torch.Tensor): Input tensor of shape (batch_size, input_size)
           
       Returns:
           torch.Tensor: Output predictions of shape (batch_size, output_size)
       """
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))  
       x = self.fc3(x)  # No activation in the last layer (for regression/classification)
       return x

def train_nn(X_train, X_test, y_train, y_test, hidden_sizes=[128, 64],
            lr=0.001, batch_size=32, epochs=100, device="cuda"):
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


   model = MLP(hidden_sizes=hidden_sizes, input_size=42, output_size=6)
   print(model)

   # Initialize scalers
   input_scaler = StandardScaler()
   output_scaler = StandardScaler()

   # Fit and transform input data
   input_scaler.fit(X_train)
   X_train_scaled = input_scaler.transform(X_train)
   X_test_scaled = input_scaler.transform(X_test)

   # Fit and transform target data
   output_scaler.fit(y_train)
   y_train_scaled = output_scaler.transform(y_train)
   y_test_scaled = output_scaler.transform(y_test)

   # convert x data to tensors
   X_train_tensor = convert_to_tensor(X_train_scaled)
   X_test_tensor = convert_to_tensor(X_test_scaled)

   # convert y data to tensors
   y_train_tensor = convert_to_tensor(y_train_scaled)
   y_test_tensor = convert_to_tensor(y_test_scaled)


   # Create DataLoader for mini-batch training
   dataset = TensorDataset(X_train_tensor, y_train_tensor)
   train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

   losses = []
   criterion = CustomLoss(position_weight=1.0, rotation_weight=0.5) # use our custom MSE weighted loss
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
   mse_values = []
   rmse_values = []
   mae_values = []
   pos_error_values = []
   rot_error_values = []

   for i in range(epochs):
      for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model.forward(X_batch) # get predicted results
        loss = criterion(y_pred, y_batch) # Measure the loss
        losses.append(loss.detach().numpy()) # Keep track of losses
        loss.backward() # Back Propogation
        optimizer.step()       

      # Normalize new test data before prediction
      X_new = input_scaler.transform(X_train)  # Scale input
      X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

      # Get predictions
      y_pred_scaled = model(X_new_tensor)

      # save training data 
      y_train_numpy = y_train_tensor.detach().numpy()
      y_pred_scaled_numpy = y_pred_scaled.detach().numpy()
      mse_values.append(compute_mse(y_pred_scaled_numpy, y_train_numpy))
      rmse_values.append(compute_rmse(y_pred_scaled_numpy, y_train_numpy))
      mae_values.append(compute_mae(y_pred_scaled_numpy, y_train_numpy))
      pos_error_values.append(compute_position_error(y_pred_scaled_numpy, y_train_numpy))
      rot_error_values.append(compute_rotation_error(y_pred_scaled_numpy, y_train_numpy))

      # Print every 10 epochs
      if i % 10 == 0:
        print(f"Epoch: {i} and loss: {loss}")

   plot_errors(mse_values, rmse_values, mae_values, pos_error_values, rot_error_values, epochs)


   # Normalize new test data before prediction
   X_new = input_scaler.transform(X_test)  # Scale input
   X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

   # Get predictions
   y_pred_scaled = model(X_new_tensor)

   # Convert predictions back to original scale
   y_pred_original = output_scaler.inverse_transform(y_pred_scaled.detach().numpy())

   print(y_pred_original)  # Final predictions in original scale
    
   return model, input_scaler, output_scaler

def plot_errors(mse_values, rmse_values, mae_values, pos_error_values, rot_error_values, epochs):
    """Plot the errors (MSE, position, rotation) vs epochs."""
    epochs_range = range(epochs)
    
    plt.figure(figsize=(10, 8))
    
    # Plot MSE
    plt.subplot(3, 2, 1)
    plt.plot(epochs_range, mse_values, label='MSE', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error vs Epochs')
    plt.grid(True)

    # Plot RMSE
    plt.subplot(3, 2, 3)
    plt.plot(epochs_range, rmse_values, label='Root Mean Squared Error', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Root Mean Squared Error vs Epochs')
    plt.grid(True)

    # Plot MAE
    plt.subplot(3, 2, 5)
    plt.plot(epochs_range, mae_values, label='Mean Absolute Error', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error vs Epochs')
    plt.grid(True)
    
    # Plot Position Error
    plt.subplot(3, 2, 2)
    plt.plot(epochs_range, pos_error_values, label='Position Error', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Position Error')
    plt.title('Position Error vs Epochs')
    plt.grid(True)
    
    # Plot Rotation Error
    plt.subplot(3, 2, 4)
    plt.plot(epochs_range, rot_error_values, label='Rotation Error', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Rotation Error')
    plt.title('Rotation Error vs Epochs')
    plt.grid(True)
    

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from datasets import prepare_dataset

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset(
        "ur10dataset.csv"
    )

    X_train_features = engineer_features(X_train.values)
    X_test_features = engineer_features(X_test.values)

    # Train model
    model, input_scaler, output_scaler = train_nn(
        X_train_features,
        X_test_features,
        y_train.values,
        y_test.values,
        hidden_sizes=[50, 100],
        lr=0.001,
        epochs=100,
    )

    torch.save(model, "NN.pth")
