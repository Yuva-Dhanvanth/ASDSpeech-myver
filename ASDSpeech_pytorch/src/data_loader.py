import numpy as np
import scipy.io as sio

def load_sa_data(mat_path):
  """
    Loads features and SA labels from train_data.mat

    Returns:
        X : np.ndarray of shape (N, 100, 49)
        y : np.ndarray of shape (N,)
  """
  mat = sio.loadmat(mat_path)

  features = mat["features"]  # shape = [1360,1]
  sa = mat["sa"]              # shape = [136,1]

  num_mats = features.shape[0]   # is a tuple returns the no.of rows i.e (1360,1)  1360
  num_childs = sa.shape[0]         # same

  x = []  # features
  y = []  # sa labels

  mat_per_child = num_mats//num_childs

  for chld in range(num_childs):
    for m in range(mat_per_child):
      idx = chld*mat_per_child + m

      x.append(features[idx,0])
      y.append(sa[chld,0])

  X = np.array(x)  # (1360,100,49)
  Y = np.array(y) # (1360,1)

  return X,Y