import numpy as np

# Créez une matrice NumPy de test
matrice = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

# Spécifiez les indices des lignes et des colonnes que vous souhaitez extraire
indices_colonnes = [0, 2]  # Par exemple, nous voulons extraire les lignes 0 et 2
indices_lignes = [1, 2]  # Par exemple, nous voulons extraire les colonnes 1 et 2

# Utilisez l'indexation NumPy pour extraire le sous-tableau
sous_tableau = matrice[indices_lignes][:, indices_colonnes]

# Affichez le sous-tableau résultant
print(sous_tableau)



n = data.shape[1]
n_ech = int(n / k_ech)

# Créez un tableau de tranches pour extraire les colonnes de data
slices = [slice(k_ech * i, k_ech * (i + 1)) for i in range(n_ech)]

# Utilisez la compréhension de liste pour extraire les colonnes en une seule opération
decimated = np.hstack([data[:, s] for s in slices])
