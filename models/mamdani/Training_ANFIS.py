import time
import sys
import numpy as np
from pyswarm import pso  # Requiere instalación: pip install pyswarm
import skfuzzy as fuzz
from ANFIS_model import contraste, homogeneidad, entropia, asm, energia, varianza, media, simulador
############# Entrenamiento ANFIS #############

def train_anfis(X_train, y_train):
    """
    Entrena el sistema ANFIS ajustando 24 parámetros (6 variables x 4 parámetros)
    utilizando PSO y retorna los parámetros óptimos.
    """
    iteration_count = 0  # Variable de control local

    # Función de pérdida ajustada para 24 parámetros (6 variables × 4 parámetros)
    def error_func(params):
        nonlocal iteration_count
        iteration_count += 1

        # Ajustar todas las funciones de membresía
        # Contraste (4 parámetros)
        contraste['bajo'].mf = fuzz.gaussmf(contraste.universe, params[0], params[1])
        contraste['alto'].mf = fuzz.gaussmf(contraste.universe, params[2], params[3])
        
        # Homogeneidad (4 parámetros)
        homogeneidad['bajo'].mf = fuzz.gaussmf(homogeneidad.universe, params[4], params[5])
        homogeneidad['alto'].mf = fuzz.gaussmf(homogeneidad.universe, params[6], params[7])
        
        # Entropía (4 parámetros)
        entropia['bajo'].mf = fuzz.gaussmf(entropia.universe, params[8], params[9])
        entropia['alto'].mf = fuzz.gaussmf(entropia.universe, params[10], params[11])
        
        # ASM (4 parámetros)
        asm['bajo'].mf = fuzz.gaussmf(asm.universe, params[12], params[13])
        asm['alto'].mf = fuzz.gaussmf(asm.universe, params[14], params[15])
        
        # Energía (4 parámetros)
        energia['bajo'].mf = fuzz.gaussmf(energia.universe, params[16], params[17])
        energia['alto'].mf = fuzz.gaussmf(energia.universe, params[18], params[19])
        
        # Varianza (4 parámetros)
        varianza['bajo'].mf = fuzz.gaussmf(varianza.universe, params[20], params[21])
        varianza['alto'].mf = fuzz.gaussmf(varianza.universe, params[22], params[23])

        # Media (4 parámetros)
        media['bajo'].mf = fuzz.gaussmf(media.universe, params[24], params[25])
        media['alto'].mf = fuzz.gaussmf(media.universe, params[26], params[27])
        
        start_error = time.time()
        error_total = 0

        for i, x in enumerate(X_train):
            # Mostrar progreso cada 500 muestras
            if i % 250 == 0 and i != 0:
                elapsed = time.time() - start_error
                print(f"  Muestra {i}/{len(X_train)} - Error parcial: {error_total/i:.4f} - Tiempo: {elapsed:.2f}s")
                sys.stdout.flush()
            # Establecer todas las entradas relevantes
            simulador.input['contraste'] = x[0]
            simulador.input['homogeneidad'] = x[2]
            simulador.input['entropia'] = x[4]
            simulador.input['asm'] = x[1]
            simulador.input['energia'] = x[3]
            simulador.input['varianza'] = x[5]
            simulador.input['media'] = x[6]

            simulador.compute()
            y_pred = simulador.output['diagnostico']
            error_total += (y_pred - y_train[i])**2

        error_promedio = error_total / len(X_train)
        print(f"Iteración {iteration_count} - Error: {error_promedio:.4f} - Tiempo: {time.time() - start_error:.2f}s")
        #error_total / len(X_train)
        return error_promedio

    # Límites para PSO (24 parámetros)
    #lb = [-3, 0.1] * 14  # 12 pares de parámetros
    #ub = [3, 1] * 14

    # Calcular media y desviación estándar de cada característica:
    feature_means = np.mean(X_train, axis=0)
    feature_stds = np.std(X_train, axis=0)

    # Definir límites para 28 parámetros (7 características × 4 parámetros)
    lb = []
    ub = []
    for mean, std in zip(feature_means, feature_stds):
        # Límites para 'bajo' (media y sigma)
        lb.extend([mean - 3*std, 0.1])  # mean bajo, sigma bajo
        lb.extend([mean - 3*std, 0.1])  # mean alto, sigma alto
        
        # Límites superiores
        ub.extend([mean + 3*std, 1.0])  # mean bajo, sigma bajo
        ub.extend([mean + 3*std, 1.0])  # mean alto, sigma alto

    # Variables de monitoreo PSO
    iteration_count = 0

    print("\nIniciando optimización PSO...")
    start_pso = time.time()

    # Optimización con más iteraciones
    # Aumentar capacidad de exploración de PSO
    params_opt, _ = pso(error_func, lb, ub, 
                    swarmsize=50, 
                    maxiter=10,
                    phip=1.5,
                    phig=2.0,
                    omega=0.4,
                    debug=True)  # Activar salida de depuración interna

    print(f"\nOptimización completada. Tiempo total: {time.time() - start_pso:.2f} segundos")
    return params_opt

if __name__ == "__main__":
    # Código de prueba: se debe llamar a train_anfis() con datos de entrenamiento
    pass