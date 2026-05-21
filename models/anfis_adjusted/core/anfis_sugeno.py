import numpy as np
from itertools import product
from numpy.linalg import lstsq
from skfuzzy import gaussmf

# Número de variables y generación de reglas
n_vars = 7
labels = ['bajo','alto']
reglas = list(product(labels, repeat=n_vars))

def compute_weights(x, mf_params):
    grados = []
    for i in range(n_vars):
        mu_b, sig_b = mf_params[2*i+0]
        mu_a, sig_a = mf_params[2*i+1]
        grados.append({
            'bajo': gaussmf(x[i], mu_b, sig_b),
            'alto': gaussmf(x[i], mu_a, sig_a)
        })
    w = np.array([np.prod([grados[i][etq] for i,etq in enumerate(regla)])
                  for regla in reglas])
    return w/w.sum() if w.sum()>0 else w

def solve_sugeno_params(X, y, mf_params):
    N = X.shape[0]; R = len(reglas)
    F = np.zeros((N, R*(n_vars+1)))
    for k in range(N):
        w_bar = compute_weights(X[k], mf_params)
        row = []
        for j in range(R):
            row.extend(w_bar[j] * X[k])   # p_i * x
            row.append(w_bar[j] * 1.0)    # sesgo r_i
        F[k] = row
    θ, *_ = lstsq(F, y, rcond=None)
    return θ

def init_mf_params(X_train):
    """
    Inicialización heurística de (mu,sigma) para cada variable:
    primero bajo = mean-std, sigma=0.5*std; luego alto = mean+std, sigma=0.5*std.
    Devuelve array (14,2).
    """
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    params = []
    for m, s in zip(means, stds):
        params.append((m - s, 0.5*s))
        params.append((m + s, 0.5*s))
    return np.array(params)
