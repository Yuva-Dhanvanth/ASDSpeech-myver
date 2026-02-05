import scipy.io as sio

mat = sio.loadmat("data/train_data.mat")

features = mat["features"]
sa = mat["sa"]

# print("Type of features:", type(features))
# print("Shape of features:", features.shape)

# print("\nType of sa:", type(sa))
# print("Shape of sa:", sa.shape)

# print("\n the sa of first person is ",sa[0])
# print("The feature of first person is ",features[0])


# print("Variables inside train_data.mat:\n")
# for key in mat.keys():
#     if not key.startswith("__"):
#         print(key)


# examinig inside the feature matrix

one_mat = features[0,0]

# print("Type of the matrix ",type(one_mat))   // nd-array 
# print("Shape of one_mat ",one_mat.shape)   //  (100, 49)



