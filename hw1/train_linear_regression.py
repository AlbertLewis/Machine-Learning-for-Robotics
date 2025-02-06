import numpy as np
from helpers.metrics import compute_mse, compute_rmse, compute_mae, compute_position_error, compute_rotation_error
import matplotlib.pyplot as plt

class SGDLinearRegression:
   """Linear regression implementation using stochastic gradient descent optimization.
   
   Attributes:
       weights (np.ndarray): Model weights 
       bias (np.ndarray): Model bias
       lr (float): Learning rate for gradient descent
       
   Methods:
       fit(X, y): Train model using mini-batch SGD
       predict(X): Make predictions on new data
       
   Example:
       >>> model = SGDLinearRegression(learning_rate=0.01)
       >>> model.fit(X_train, y_train, batch_size=32, epochs=100)
       >>> y_pred = model.predict(X_test)
   """
   
   def _initialize_parameters(self, input_dim, output_dim):
       """Initialize model weights and bias.
       
       Args:
           input_dim (int): Number of input features
           output_dim (int): Number of output dimensions
       """
       self.weights = np.zeros((input_dim, output_dim)) # shape (n_features, n_outputs)
       self.bias = np.zeros((1, output_dim)) # predicted - true, shape (1, n_outputs)
       self.lr = 0.01 # Learning rate

       pass
       
   def _compute_loss(self, y_pred, y_true):
       """Compute MSE loss between predictions and targets.
       
       Args:
           y_pred (np.ndarray): Model predictions
           y_true (np.ndarray): Ground truth values
           
       Returns:
           float: MSE loss value
       """
       loss = self.lr * (y_true - y_pred)
       return loss
       pass
       
   def _compute_gradients(self, X, y_true, y_pred):
       """Compute gradients for weights and bias.
       
       Args:
           X (np.ndarray): Input features
           y_true (np.ndarray): Ground truth values  
           y_pred (np.ndarray): Model predictions
           
       Returns:
           tuple: Weight gradients and bias gradients
       """
       bias_g = y_true - y_pred
    #    print(f"bias shape: {bias_g.shape}")
    #    print(bias_g)
    #    print(f"X shape: {X.shape}")
    #    print(X)
       rows = X.shape[0]
       X = X.reshape(rows, 1)
       weight_g = X @ bias_g
    #    print(f"weight_g: {weight_g}")
       return weight_g, bias_g
       pass
       
   def fit(self, X, y, batch_size=32, epochs=10000):
       """Train model using mini-batch SGD.
       
       Args:
           X (np.ndarray): Training features of shape amples(n_s, n_features)
           y (np.ndarray): Target values of shape (n_samples, n_outputs)
           batch_size (int): Mini-batch size for SGD
           epochs (int): Number of training epochs
       """
       self._initialize_parameters(X.shape[1], y.shape[1])

       mse_values = []
       rmse_values = []
       mae_values = []
       pos_error_values = []
       rot_error_values = []
       

       for _ in range(epochs):
           indices = np.random.randint(0, X.shape[0], size=batch_size)
           weight_g_sum = 0
           bias_g_sum = 0
           for i in indices: 
              y_pred = self.weights.T @ X[i] + self.bias
              loss = self._compute_loss(y[i], y_pred)
              weight_g, bias_g = self._compute_gradients(X[i], y[i], y_pred)

              weight_g_sum += weight_g
              bias_g_sum += bias_g
              
           self.weights += (self.lr * weight_g_sum) / batch_size
           self.bias += (self.lr * bias_g_sum) / batch_size
           
           # Evaluate error after each epoch
           y_pred_train = self.predict(X)
           mse_values.append(compute_mse(y_pred_train, y))
           rmse_values.append(compute_rmse(y_pred_train, y))
           mae_values.append(compute_mae(y_pred_train, y))
           pos_error_values.append(compute_position_error(y_pred_train, y))
           rot_error_values.append(compute_rotation_error(y_pred_train, y))

       self.plot_errors(mse_values, rmse_values, mae_values, pos_error_values, rot_error_values, epochs)
       
       
   def predict(self, X):
       """Make predictions for given input features.
       
       Args:
           X (np.ndarray): Input features of shape (n_samples, n_features)
           
       Returns:
           np.ndarray: Predicted values of shape (n_samples, n_outputs)
       """
    #    print(f"shape weight: {self.weights.shape}")
    #    print(f"shapeX: {X.shape}")
       return X @ self.weights
       

   def plot_errors(self, mse_values, rmse_values, mae_values, pos_error_values, rot_error_values, epochs):
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

    # Train model
    model = SGDLinearRegression()
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = compute_mse(y_pred, y_test)
    pos_error = compute_position_error(y_pred, y_test)
    rot_error = compute_rotation_error(y_pred, y_test)

    print(f"Test MSE: {mse:.4f}")
    print(f"Position Error: {pos_error:.4f}")
    print(f"Rotation Error: {rot_error:.4f}")

