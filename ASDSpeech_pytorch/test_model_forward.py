import torch
from src.data_loader import load_sa_data
from src.normalization import compute_mean_std, normalize
from src.model import ASDSpeechCNN

# Load data
X, y = load_sa_data("data/train_data.mat")

# Normalize
mean, std = compute_mean_std(X)
X_norm = normalize(X, mean, std)

# Take a small batch
X_batch = X_norm[:8]  # batch size = 8

# Convert to torch tensor
X_batch = torch.tensor(X_batch, dtype=torch.float32)

# Create model
model = ASDSpeechCNN()

# Forward pass
output = model(X_batch)

print("Input batch shape:", X_batch.shape)
print("Output shape:", output.shape)
print("Output values:\n", output)
