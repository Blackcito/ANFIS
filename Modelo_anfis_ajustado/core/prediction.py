import numpy as np
from core.anfis_sugeno import compute_weights, n_vars, reglas

def predict_sugeno(X, mf_params, theta, threshold=0.5):
    """
    Predicción pura del modelo ANFIS-Sugeno
    Devuelve salidas continuas y binarias
    """
    y_cont = []
    for x in X:
        w_bar = compute_weights(x, mf_params)
        total = 0.0
        for j in range(len(reglas)):
            p_j = theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
            r_j = theta[j*(n_vars+1) + n_vars]
            total += w_bar[j] * (p_j.dot(x) + r_j)
        y_cont.append(total)
    
    y_cont = np.array(y_cont)
    y_bin = (y_cont > threshold).astype(int)
    return y_cont, y_bin

def predict_single_with_explanation(x_sample, mf_params, theta, n_top_reglas=5):
    """
    Predice una muestra individual con explicación detallada
    """
    w_bar = compute_weights(x_sample, mf_params)
    
    contribuciones = []
    total = 0.0
    
    for j in range(len(reglas)):
        p_j = theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
        r_j = theta[j*(n_vars+1) + n_vars]
        contribucion = w_bar[j] * (p_j.dot(x_sample) + r_j)
        total += contribucion
        
        contribuciones.append({
            'regla_idx': j,
            'regla': reglas[j],
            'activacion': w_bar[j],
            'contribucion': contribucion,
            'parametros': p_j,
            'bias': r_j
        })
    
    # Ordenar por contribución absoluta
    contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
    
    prediction = (total > 0.5).astype(int)
    
    return {
        'prediccion_continua': total,
        'prediccion_binaria': prediction,
        'top_reglas_activas': contribuciones[:n_top_reglas],
        'todas_contribuciones': contribuciones
    }