from first import reversal_matrix
import numpy as np

C1 = np.array([1, 2, 2, 4, 1, 2, 4, 2, 3], dtype=np.float).reshape((3, 3))
C2 = np.array([0, 2, 1, 0], dtype=np.float).reshape((2, 2))

print reversal_matrix(C1)
print reversal_matrix(C2)
