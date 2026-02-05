import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import load_sa_data
from src.normalization import compute_mean_std, normalize
from src.model import ASDSpeechCNN
import scipy.io as sio
import numpy as np

# -----------------------
# READ TARGET NAME
# -----------------------
# usage: python train_score.py sa
score_name = sys.argv[1]   # "sa", "rrb", or "ADOS"

# -----------------------
# LOAD DATA
# -----------------------
mat = sio.loadmat("data/train_data.mat")

features = mat["features"]
labels = mat[score_name]   # sa / rrb / ADOS

X = []
y = []

num_children = labels.shape[0]
num_mats = features.shape[0]
mats_per_child = num_mats // num_children

for i in range(num_children):
    for j in range(mats_per_child):
        idx = i * mats_per_child + j
        X.append(features[idx, 0])
        y.append(labels[i, 0])

X = np.array(X)
y = np.array(y)

# -----------------------
# NORMALIZATION
# -----------------------
mean, std = compute_mean_std(X)
X = normalize(X, mean, std)

# -----------------------
# TORCH CONVERSION
# -----------------------
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------
# MODEL
# -----------------------
model = ASDSpeechCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# TRAINING
# -----------------------
epochs = 5

for epoch in range(epochs):
    total_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[{score_name.upper()}] Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")



import os
os.makedirs("artifacts", exist_ok=True)

torch.save(model.state_dict(), f"artifacts/{score_name}_model.pth")
np.save(f"artifacts/{score_name}_mean.npy", mean)
np.save(f"artifacts/{score_name}_std.npy", std)

print(f"Saved {score_name} model and normalization artifacts")
