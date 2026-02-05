import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import load_sa_data
from src.normalization import compute_mean_std, normalize
from src.model import ASDSpeechCNN

# -----------------------
# Load data
# -----------------------
X, y = load_sa_data("data/train_data.mat")

mean, std = compute_mean_std(X)
X = normalize(X, mean, std)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# -----------------------
# Dataset & DataLoader
# -----------------------
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------
# Model, loss, optimizer
# -----------------------
model = ASDSpeechCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# Training loop
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

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
