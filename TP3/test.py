import numpy as np

# Créez une matrice NumPy de test
U = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

V = np.array([[9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17]])

sum = V * U.T

print("UT =", U.T)

print("sum =", sum)

sum = np.dot(U, V)

print("sum =", sum)

