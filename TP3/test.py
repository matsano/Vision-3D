import numpy as np

# Créez une matrice NumPy de test
matrice = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

translation = np.array([[0, -0.1, 0.1]]).T

points = matrice + translation

print("points antes =",points)

points -= translation
print("points dps =",points)

