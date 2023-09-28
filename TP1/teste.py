import numpy as np

X_init = np.array([[0,       0],
                   [0,     500],
                   [700,   500],
                   [700,     0]])

T_norm = np.array([[2/700,  0,       -1],
                   [0,      2/500,   -1],
                   [0,      0,        1]])

X_init = np.hstack((X_init, np.ones((X_init.shape[0], 1))))

print(X_init)

X_init_norm = np.dot(T_norm, np.transpose(X_init))

print(X_init_norm)

X_init_norm = np.transpose(X_init_norm)
print(X_init_norm)

X_init_norm = X_init_norm[:, :-1]
print(X_init_norm)

print(np.linalg.inv(T_norm))
