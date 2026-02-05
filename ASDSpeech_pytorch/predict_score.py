import sys
import torch
import numpy as np
import scipy.io as sio

from src.model import ASDSpeechCNN
from src.normalization import normalize

# -----------------------
# USAGE:
# python predict_score.py sa
# -----------------------

score_name = sys.argv[1]  # sa / rrb / ADOS

# -----------------------
# LOAD DATA
# -----------------------
mat = sio.loadmat("data/train_data.mat")
features = mat["features"]

# take ONE sample for demo (first matrix)
X = features[0, 0]          # shape (100, 49)
X = np.expand_dims(X, 0)    # (1, 100, 49)

# -----------------------
# LOAD NORMALIZATION
# -----------------------
mean = np.load(f"artifacts/{score_name}_mean.npy")
std  = np.load(f"artifacts/{score_name}_std.npy")

X = normalize(X, mean, std)

# -----------------------
# TORCH CONVERSION
# -----------------------
X = torch.tensor(X, dtype=torch.float32)

# -----------------------
# LOAD MODEL
# -----------------------
model = ASDSpeechCNN()
model.load_state_dict(torch.load(
    f"artifacts/{score_name}_model.pth",
    weights_only=True
))

model.eval()

# -----------------------
# PREDICTION
# -----------------------
# -----------------------
# PREDICT ALL MATRICES
# -----------------------
model.eval()
preds = []

with torch.no_grad():
    for i in range(10):  # first child = first 10 matrices
        X = features[i, 0]
        X = np.expand_dims(X, 0)
        X = normalize(X, mean, std)
        X = torch.tensor(X, dtype=torch.float32)

        pred = model(X)
        preds.append(pred.item())

final_score = sum(preds) / len(preds)

print(f"Predicted {score_name.upper()} score (averaged): {final_score:.2f}")
