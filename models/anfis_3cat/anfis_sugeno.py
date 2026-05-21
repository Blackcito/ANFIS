import numpy as np
from itertools import product
from numpy.linalg import lstsq
from skfuzzy import gaussmf

# Número de variables y generación de reglas
n_vars = 7
n_classes = 3  # Nueva variable para número de clases
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

def solve_sugeno_params_multiclass(X, y, mf_params):
    """
    Versión multiclase de solve_sugeno_params
    Resuelve parámetros para cada clase por separado
    """
    N = X.shape[0]
    R = len(reglas)
    
    # Construir matriz F (igual que antes)
    F = np.zeros((N, R*(n_vars+1)))
    for k in range(N):
        w_bar = compute_weights(X[k], mf_params)
        row = []
        for j in range(R):
            row.extend(w_bar[j] * X[k])   # p_i * x
            row.append(w_bar[j] * 1.0)    # sesgo r_i
        F[k] = row
    
    # One-hot encoding de las etiquetas
    y_onehot = np.eye(n_classes)[y]
    
    # Resolver para cada clase
    thetas = []
    for c in range(n_classes):
        θ_c, *_ = lstsq(F, y_onehot[:, c], rcond=None)
        thetas.append(θ_c)
    
    return np.concatenate(thetas)

def init_mf_params(X_train):
    """
    Inicialización heurística de (mu,sigma) para cada variable
    """
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    params = []
    for m, s in zip(means, stds):
        params.append((m - s, 0.5*s))
        params.append((m + s, 0.5*s))
    return np.array(params)

# Nueva función para predicción multiclase
def predict_multiclass(x, mf_params, theta):
    """
    Predicción para modelo multiclase
    Devuelve probabilidades para cada clase
    """
    w_bar = compute_weights(x, mf_params)
    outputs = np.zeros(n_classes)
    
    for c in range(n_classes):
        # Obtener parámetros para esta clase
        start_idx = c * len(reglas) * (n_vars+1)
        end_idx = (c+1) * len(reglas) * (n_vars+1)
        theta_c = theta[start_idx:end_idx]
        
        total = 0.0
        for j in range(len(reglas)):
            p_j = theta_c[j*(n_vars+1) : j*(n_vars+1)+n_vars]
            r_j = theta_c[j*(n_vars+1) + n_vars]
            total += w_bar[j] * (p_j.dot(x) + r_j)
        
        outputs[c] = total
    
    # Aplicar softmax para obtener probabilidades
    exp_outputs = np.exp(outputs - np.max(outputs))  # Para estabilidad numérica
    return exp_outputs / np.sum(exp_outputs)