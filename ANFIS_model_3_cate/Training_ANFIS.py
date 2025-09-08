import time
import numpy as np
from pyswarm import pso
from anfis_sugeno import compute_weights, solve_sugeno_params_multiclass, init_mf_params, n_vars, reglas, n_classes,predict_multiclass
from procesamiento_image import process_all_images

def train_anfis_multiclass(X_train, y_train, swarmsize=50, maxiter=20):
    # 1) Inicializar parámetros MF
    mf0 = init_mf_params(X_train).reshape(-1)

    # 2) Definir límites
    mins = mf0 - 3.0
    maxs = mf0 + 3.0

    # 3) Función objetivo para PSO (multiclase)
    def error_func(flat_params):
        mf_params = flat_params.reshape((-1,2))
        
        # 3a) Calcular parámetros de salida por mínimos cuadrados
        θ = solve_sugeno_params_multiclass(X_train, y_train, mf_params)
        
        # 3b) Predecir y calcular error (cross-entropy para multiclase)
        error = 0.0
        for i, x in enumerate(X_train):
            probs = predict_multiclass(x, mf_params, θ)
            true_class = y_train[i]
            error -= np.log(probs[true_class] + 1e-10)  # Cross-entropy
            
        return error / len(X_train)

    # 4) Ejecutar PSO
    print("Iniciando PSO para modelo multiclase...")
    start = time.time()
    opt_flat, opt_err = pso(error_func, mins, maxs,
                            swarmsize=swarmsize, maxiter=maxiter,
                            phip=1.5, phig=2.0, omega=0.5)
    print(f"PSO completo en {time.time()-start:.1f}s, error={opt_err:.4f}")

    # 5) Reconstruir parámetros finales
    mf_opt = opt_flat.reshape((-1,2))
    θ_opt  = solve_sugeno_params_multiclass(X_train, y_train, mf_opt)
    return mf_opt, θ_opt

if __name__ == "__main__":
    # Ejemplo de uso para multiclase
    X_train, y_train = process_all_images()
    mf_opt, θ_opt = train_anfis_multiclass(X_train, y_train, swarmsize=30, maxiter=10)