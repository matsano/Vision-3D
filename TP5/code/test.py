import numpy as np

# Exemplo 1: Encontrar índices onde um array atende a uma condição
arr = np.array([1, 2, 3, 4, 5])
indices = np.where(arr > 3)
print(indices)  # Saída: (array([3, 4]),)

# Exemplo 2: Atribuir um valor a elementos com base em uma condição
arr = np.array([1, 2, 3, 4, 5])
arr[np.where(arr > 3)] = 10
print(arr)  # Saída: [ 1  2  3 10 10]
