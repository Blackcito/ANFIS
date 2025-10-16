import numpy as np
from core.anfis_sugeno import compute_weights, reglas, n_vars

class ExplicadorANFIS:
    def __init__(self, modelo):
        self.mf_params = modelo['mf_params']
        self.theta = modelo['theta']
    
    def explicar_prediccion(self, x_sample, n_top_reglas=5):
        """Explica una predicción individual"""
        w_bar = compute_weights(x_sample, self.mf_params)
        
        contribuciones = []
        total = 0.0
        
        for j in range(len(reglas)):
            p_j = self.theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
            r_j = self.theta[j*(n_vars+1) + n_vars]
            contribucion = w_bar[j] * (p_j.dot(x_sample) + r_j)
            total += contribucion
            
            contribuciones.append({
                'regla_idx': j,
                'regla': reglas[j],
                'activacion': w_bar[j],
                'contribucion': contribucion
            })
        
        contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
        prediction = (total > 0.5).astype(int)
        
        return {
            'prediccion_continua': total,
            'prediccion_binaria': prediction,
            'top_reglas_activas': contribuciones[:n_top_reglas]
        }
    
    def comparar_por_clase(self, X_data, y_data):
        """Compara reglas por clase"""
        indices_no_tumor = np.where(y_data == 0)[0]
        indices_tumor = np.where(y_data == 1)[0]
        
        activaciones_no_tumor = np.zeros(len(reglas))
        activaciones_tumor = np.zeros(len(reglas))
        
        for idx in indices_no_tumor:
            w = compute_weights(X_data[idx], self.mf_params)
            activaciones_no_tumor += w
        activaciones_no_tumor /= len(indices_no_tumor)
        
        for idx in indices_tumor:
            w = compute_weights(X_data[idx], self.mf_params)
            activaciones_tumor += w
        activaciones_tumor /= len(indices_tumor)
        
        diferencia_activaciones = activaciones_tumor - activaciones_no_tumor
        
        return {
            'activaciones_no_tumor': activaciones_no_tumor,
            'activaciones_tumor': activaciones_tumor,
            'diferencia_activaciones': diferencia_activaciones
        }