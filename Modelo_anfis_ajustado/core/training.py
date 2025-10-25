# core/training.py - ACTUALIZADO

import time
import numpy as np
from pyswarm import pso
from core.anfis_sugeno import compute_weights, solve_sugeno_params, init_mf_params, n_vars, reglas
from datetime import timedelta

# Importar el sistema de caché centralizado
from utils.cache import sistema_cache

def train_anfis(X_train, y_train, swarmsize=50, maxiter=20, guardar_modelo=True, nombre_modelo="modelo_anfis"):
    """
    Entrena modelo ANFIS usando el sistema de caché centralizado
    """
    # 1) Inicializar parámetros MF
    mf0 = init_mf_params(X_train).reshape(-1)

    # 2) Definir límites
    mins = mf0 - 3.0
    maxs = mf0 + 3.0

    # 3) Función objetivo para PSO
    def error_func(flat_params):
        mf_params = flat_params.reshape((-1,2))
        θ = solve_sugeno_params(X_train, y_train, mf_params)
        
        y_hat = []
        for x in X_train:
            w_bar = compute_weights(x, mf_params)
            total = 0.0
            for j in range(len(reglas)):
                pj = θ[j*(n_vars+1): j*(n_vars+1)+n_vars]
                rj = θ[j*(n_vars+1)+n_vars]
                total += w_bar[j] * (pj.dot(x) + rj)
            y_hat.append(total)
        return np.mean((np.array(y_hat) - y_train)**2)

    # 4) Ejecutar PSO
    print(" Iniciando PSO...")
    start = time.time()
    opt_flat, opt_err = pso(error_func, mins, maxs,
                            swarmsize=swarmsize, maxiter=maxiter,
                            phip=1.5, phig=2.0, omega=0.5)
    elapsed_pso = time.time() - start
    print(f" PSO completo en {timedelta(seconds=int(elapsed_pso))}, error={opt_err:.4f}")

    # 5) Reconstruir parámetros finales
    mf_opt = opt_flat.reshape((-1,2))
    θ_opt  = solve_sugeno_params(X_train, y_train, mf_opt)
    
    # 6) Guardar modelo si se solicita
    if guardar_modelo:
        info_entrenamiento = {
            'swarmsize': swarmsize,
            'maxiter': maxiter,
            'error_final': float(opt_err),
            'n_muestras_entrenamiento': len(X_train),
            'n_caracteristicas': X_train.shape[1]
        }
        
        sistema_cache.guardar_modelo(mf_opt, θ_opt, 
                                   info_entrenamiento=info_entrenamiento,
                                   nombre=nombre_modelo)
    
    return mf_opt, θ_opt