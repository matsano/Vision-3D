import numpy as np

vet1 = np.array([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
vet2 = np.array([2, 4, 6, 8, 10], [1, 3, 5, 7, 9])

print(np.maximum(vet1, vet2))