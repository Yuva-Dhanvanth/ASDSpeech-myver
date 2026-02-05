from src.data_loader import load_sa_data

X, y = load_sa_data("data/train_data.mat")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First sample label:", y[0])
print("First sample matrix shape:", X[0].shape)
