import time
import numpy as np
from pyswarm import pso
from anfis_sugeno import compute_weights, solve_sugeno_params, init_mf_params, n_vars, reglas
from procesamiento_image import process_all_images  # tu GLCM extractor

def train_anfis(X_train, y_train, swarmsize=50, maxiter=20):
    # 1) Inicializar parámetros MF
    mf0 = init_mf_params(X_train).reshape(-1)

    # 2) Definir límites (a ajustar según el rango real de tus datos)
    mins = mf0 - 3.0  # ejemplo: mu ±3
    maxs = mf0 + 3.0

    # 3) Función objetivo para PSO
    def error_func(flat_params):
        mf_params = flat_params.reshape((-1,2))
        # 3a) Calcular parámetros de salida por mínimos cuadrados
        θ = solve_sugeno_params(X_train, y_train, mf_params)
        # 3b) Predecir y calcular MSE
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
    print("Iniciando PSO...")
    start = time.time()
    opt_flat, opt_err = pso(error_func, mins, maxs,
                            swarmsize=swarmsize, maxiter=maxiter,
                            phip=1.5, phig=2.0, omega=0.5)
    print(f"PSO completo en {time.time()-start:.1f}s, error={opt_err:.4f}")

    # 5) Reconstruir parámetros finales
    mf_opt = opt_flat.reshape((-1,2))
    θ_opt  = solve_sugeno_params(X_train, y_train, mf_opt)
    return mf_opt, θ_opt

if __name__ == "__main__":
    # 1) Extrae tus datos GLCM
    X_train, y_train = process_all_images()
    # 2) Entrena ANFIS-Sugeno
    mf_opt, θ_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    # 3) Predice y evalúa (puedes volver a usar el snippet de predict_binary)
