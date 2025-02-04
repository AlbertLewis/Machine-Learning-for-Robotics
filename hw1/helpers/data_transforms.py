class StandardScaler:
    """Scales features using standardization: (x - mean) / std.
    
    Methods:
        fit(data): Compute mean and standard deviation for scaling.
        transform(data): Scale features using precomputed statistics.
        inverse_transform(data): Convert scaled data back to original scale.
        
    Attributes:
        mean: Array of mean values for each feature
        std: Array of standard deviations for each feature
        
    Example:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    """
    
    def fit(self, data):
        """Compute mean and standard deviation of features for scaling.
        
        Args:
            data (np.ndarray): Input features of shape (n_samples, n_features)
        """
        pass
    
    def transform(self, data):
        """Scale features by removing mean and scaling to unit variance.
        
        Args:
            data (np.ndarray): Input features to scale
            
        Returns:
            np.ndarray: Scaled features
        """
        pass
    
    def inverse_transform(self, data):
        """Convert scaled features back to original scale.
        
        Args:
            data (np.ndarray): Scaled input features
            
        Returns:
            np.ndarray: Features in original scale
        """
        pass

def convert_to_tensor(data, device="cpu"):
    """Convert numpy array or list to PyTorch tensor.
    
    Args:
        data (Union[np.ndarray, list]): Input data to convert
        device (str): Target device for tensor ('cpu' or 'cuda')
        
    Returns:
        torch.Tensor: PyTorch tensor on specified device
        
    Example:
        X_tensor = convert_to_tensor(X_numpy, device='cuda')
    """
    pass