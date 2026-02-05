import numpy as np

def compute_mean_std(X):
  """
  X shape: (N, 100, 49)
  Computes mean and std over ALL samples and time steps
  for each feature.

  Returns:
      mean: (49,)
      std:  (49,)
  """
  mean = X.mean(axis=(0, 1))
  std = X.std(axis=(0, 1))
  return mean, std


def normalize(X, mean, std):
    """
    Z-normalization with NaN and zero-std protection
    """
    std_safe = np.where(std == 0, 1.0, std)
    X_norm = (X - mean) / std_safe
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return X_norm
