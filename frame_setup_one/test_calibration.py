import numpy as np

# Temporary identity matrix for testing
homography_matrix = np.eye(3)
np.save("homography_matrix.npy", homography_matrix)

print("Created temporary homography matrix for testing!")