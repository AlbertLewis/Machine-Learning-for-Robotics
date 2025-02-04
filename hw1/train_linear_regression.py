import numpy as np
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error

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
       self.bias = np.zeros((1, output_dim)) # predicted - true, shape (1, n_ouitputs)
       self.lr = 0.1 # Learning rate

       pass
       
   def _compute_loss(self, y_pred, y_true):
       """Compute MSE loss between predictions and targets.
       
       Args:
           y_pred (np.ndarray): Model predictions
           y_true (np.ndarray): Ground truth values
           
       Returns:
           float: MSE loss value
       """
       loss = self.lr*(y_true - y_pred)
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
       weight_g = bias_g * X
       return weight_g, bias_g
       pass
       
   def fit(self, X, y, batch_size=32, epochs=100):
       """Train model using mini-batch SGD.
       
       Args:
           X (np.ndarray): Training features of shape amples(n_s, n_features)
           y (np.ndarray): Target values of shape (n_samples, n_outputs)
           batch_size (int): Mini-batch size for SGD
           epochs (int): Number of training epochs
       """
       self._initialize_parameters(X.shape[1], y.shape[1])
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
              
            #   sum += self._compute_gradients(X[i], y[i], y_pred) + loss
           self.weights += self.lr*weight_g_sum
           self.bias += self.lr*bias_g_sum
       
       
   def predict(self, X):
       """Make predictions for given input features.
       
       Args:
           X (np.ndarray): Input features of shape (n_samples, n_features)
           
       Returns:
           np.ndarray: Predicted values of shape (n_samples, n_outputs)
       """
       return self.weights @ X.T
       pass


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
    # model = SGDLinearRegression(learning_rate=0.01)
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
