from src.data_loader import load_sa_data
from src.normalization import compute_mean_std, normalize
import numpy as np

X, y = load_sa_data("data/train_data.mat")

mean, std = compute_mean_std(X)
X_norm = normalize(X, mean, std)

print("Mean shape:", mean.shape)
print("Std shape:", std.shape)

print("Normalized mean (approx):", np.mean(X_norm))
print("Normalized std (approx):", np.std(X_norm))
