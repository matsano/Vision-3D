import numpy as np

# Suponha que você tenha uma matriz numpy bidimensional chamada 'matriz'
matriz = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])

# Encontre o valor mínimo e máximo da matriz
min_valor = np.min(matriz)
max_valor = np.max(matriz)

# Normalize a matriz para o intervalo [0, 1]
matriz_normalizada = (matriz - min_valor) / (max_valor - min_valor)

print(matriz_normalizada)
