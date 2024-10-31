import numpy as np
import pandas as pd
import torch

def create_sliding_window(data, labels, window_size, step_size=1):
    """
    Create sliding window subsequences from the data.
    Args:
    - data: Numpy array of shape (num_timesteps, num_features)
    - labels: Numpy array of shape (num_timesteps,), corresponding RUL values
    - window_size: Number of time steps in each window
    - step_size: Step size for the sliding window (controls overlap)
    Returns:
    - X_windows: List of feature windows
    - y_windows: Corresponding target for each window (target at the last time step)
    """
    X_windows = []
    y_windows = []

    for i in range(0, len(data) - window_size + 1, step_size):
        X_windows.append(data[i:i + window_size])
        y_windows.append(labels[i + window_size - 1])  # Target is the RUL at the end of the window

    return np.array(X_windows), np.array(y_windows)

def median_smoothing_with_kernel(data, window_size=3):
    """
    Applies median smoothing to a 2D array using a moving median filter on each column independently.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The input data to be smoothed, where each column represents a feature (e.g., a time series).
        
    window_size : int, optional, default=3
        The size of the moving window used for calculating the median. A larger window will smooth more 
        but may also reduce detail in the signal.
    
    Returns:
    --------
    smoothed_data : numpy array, shape (n_samples, n_features)
        The smoothed data as a 2D NumPy array with each column independently median-smoothed.
    """

    smoothed_df = pd.DataFrame(data)
    
    # Apply the moving median with padding to each column independently
    for col in smoothed_df.columns:
        series = pd.Series(smoothed_df[col])
        smoothed_padded_data = series.rolling(window=window_size, center=True).median()
        smoothed_df[col] = smoothed_padded_data.values

    return smoothed_df.to_numpy()

def create_matrix_from_arange(vec1, vec2, n):
    """
    Creates a matrix where each row is generated from an evenly spaced sequence between 
    corresponding values in two input vectors.

    Parameters:
    -----------
    vec1 : 1D array-like (torch.Tensor or list)
        The starting values of the range for each row in the matrix.
        
    vec2 : 1D array-like (torch.Tensor or list)
        The ending values of the range for each row in the matrix (exclusive).

    n : int
        The number of evenly spaced elements between each pair of values from vec1 and vec2.

    Returns:
    --------
    matrix : torch.Tensor, shape (len(vec1), max_length)
        A 2D tensor where each row contains an evenly spaced sequence between the corresponding
        elements in vec1 and vec2. If rows are of varying lengths, they are padded with zeros to match 
        the maximum row length in the matrix.
    """
    rows = []
    
    for v1, v2 in zip(vec1, vec2):
        # Create a range from v1 to v2 (exclusive) and append to rows
        row = torch.linspace(v1, v2, n)
        rows.append(row)
    
    max_length = max(len(row) for row in rows)
    
    padded_rows = [torch.cat([row, torch.zeros(max_length - len(row))]) for row in rows]
    
    matrix = torch.stack(padded_rows)
    
    return matrix