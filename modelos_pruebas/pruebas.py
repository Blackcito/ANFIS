import os
import time
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
    """Extrae 6 características GLCM válidas."""
    if np.std(image) == 0:
        raise ValueError("Imagen uniforme: no se puede calcular GLCM")
    
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    try:
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Cálculo manual
        variance = graycoprops(glcm, 'variance')[0, 0]
    except Exception as e:
        raise ValueError(f"Error en GLCM: {e}")
    
    return np.array([contrast, asm, homogeneity, energy, entropy, variance])

def preprocess_image(image_path):
    """Preprocesamiento con umbralización adaptativa y operaciones morfológicas."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error al cargar: {image_path}")
    
    # 1. Ecualización del histograma
    equalized_img = cv2.equalizeHist(img)
    
    # 2. Umbralización adaptativa optimizada
    binary_adaptive = cv2.adaptiveThreshold(
        equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 4  # blockSize=11, C=4
    )
    
    # 3. Operaciones morfológicas: apertura + cierre
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

# Procesar todas las imágenes de ambos directorios


# Configurar rutas base
base_dir = "D:/Google Drive/universidad/Tesis/Codigos/python/CNN_ANFIS/archive/test"
meningioma_dir = os.path.join(base_dir, "meningioma", "Tr-me_*.jpg")
notumor_dir = os.path.join(base_dir, "notumor", "Tr-no_*.jpg")

# Obtener todas las rutas de imágenes
image_paths = glob.glob(meningioma_dir) + glob.glob(notumor_dir)

# Procesar imágenes con monitoreo
features = []
labels = []
error_count = 0
start_time = time.time()

print("\nProcesando imágenes:")
for i, path in enumerate(image_paths):
    try:
        # Mostrar progreso cada 100 imágenes
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\nImagen {i+1}/{len(image_paths)} - Tiempo: {elapsed:.2f}s")
            sys.stdout.flush()
            
        processed_img = preprocess_image(path)
        glcm_features = extract_glcm_features(processed_img)
        features.append(glcm_features)
        labels.append(1 if "meningioma" in path else 0)
        
    except Exception as e:
        error_count += 1
        print(f"\nERROR en imagen {i+1}: {path}")
        print(f"Tipo error: {str(e)}")
        print("Saltando imagen...")
        continue

print(f"\nProcesamiento completado. Errores: {error_count}/{len(image_paths)}")
print(f"Tiempo total: {time.time() - start_time:.2f} segundos")
# Normalizar
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

print(f"\nNormalización exitosa. Dimensiones: {normalized_features.shape}")
print("Ejemplo de características normalizadas:\n", normalized_features[0])


############# Entrenamiento ANFIS #############

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variables de Entrada (5 características GLCM)
contraste = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'contraste')
homogeneidad = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'homogeneidad')
entropia = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'entropia')
asm = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'asm')
energia = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'energia')
varianza = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'varianza')

# Variable de Salida
diagnostico = ctrl.Consequent(np.arange(0, 1, 0.01), 'diagnostico')


# Funciones de membresía para cada entrada (ej: contraste)
contraste['bajo'] = fuzz.gaussmf(contraste.universe, -1.5, 0.5)
contraste['alto'] = fuzz.gaussmf(contraste.universe, 1.5, 0.5)

homogeneidad['bajo'] = fuzz.gaussmf(homogeneidad.universe, -1, 0.5)
homogeneidad['alto'] = fuzz.gaussmf(homogeneidad.universe, 1.1, 0.5)

entropia['bajo'] = fuzz.gaussmf(entropia.universe, -1.2, 0.5)
entropia['alto'] = fuzz.gaussmf(entropia.universe, 1.1, 0.5)

asm['bajo'] = fuzz.gaussmf(asm.universe, -1.2, 0.5)
asm['alto'] = fuzz.gaussmf(asm.universe, 1.2, 0.5)

energia['bajo'] = fuzz.gaussmf(energia.universe, -1.2, 0.5)
energia['alto'] = fuzz.gaussmf(energia.universe, 1.2, 0.5)

varianza['bajo'] = fuzz.gaussmf(varianza.universe, -1.2, 0.5)
varianza['alto'] = fuzz.gaussmf(varianza.universe, 1.2, 0.5)

# Usar función trapezoidal para mejor interpretación
diagnostico['tumor'] = fuzz.trapmf(diagnostico.universe, [0, 0, 0.4, 0.6])
diagnostico['no_tumor'] = fuzz.trapmf(diagnostico.universe, [0.4, 0.6, 1, 1])

# Regla 1: Si contraste es bajo Y homogeneidad es alta Y entropía es baja -> tumor
regla1 = ctrl.Rule(contraste['bajo'] & homogeneidad['alto'] & entropia['bajo'],diagnostico['tumor'])

# Regla 2: Si contraste es alto Y homogeneidad es baja Y entropía es alta -> no tumor
regla2 = ctrl.Rule(contraste['alto'] & homogeneidad['bajo'] & entropia['alto'],diagnostico['no_tumor'])

regla3 = ctrl.Rule(energia['bajo'] & asm['bajo'] & varianza['alto'],diagnostico['tumor'])

# Regla 4: Si contraste es bajo Y ASM es alto Y homogeneidad es alta -> tumor
regla4 = ctrl.Rule(contraste['bajo'] & asm['alto'] & homogeneidad['alto'], diagnostico['tumor'])

# Regla 5: Si energía es alta Y entropía es baja Y varianza es baja -> tumor
regla5 = ctrl.Rule(energia['alto'] & entropia['bajo'] & varianza['bajo'], diagnostico['tumor'])

# Regla 6: Si contraste es alto Y ASM es bajo Y homogeneidad es baja -> no tumor
regla6 = ctrl.Rule(contraste['alto'] & asm['bajo'] & homogeneidad['bajo'], diagnostico['no_tumor'])

# Regla 7: Si energía es baja Y entropía es alta Y varianza es alta -> no tumor
regla7 = ctrl.Rule(energia['bajo'] & entropia['alto'] & varianza['alto'], diagnostico['no_tumor'])

# Regla 8: Si contraste es bajo Y energía es alta Y entropía es baja -> tumor
regla8 = ctrl.Rule(contraste['bajo'] & energia['alto'] & entropia['bajo'], diagnostico['tumor'])

# Regla 9: Si contraste es alto Y energía es baja Y entropía es alta -> no tumor
regla9 = ctrl.Rule(contraste['alto'] & energia['bajo'] & entropia['alto'], diagnostico['no_tumor'])

# Sistema actualizado con las nuevas reglas
global sistema
sistema = ctrl.ControlSystem([
    regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, regla9
])
simulador = ctrl.ControlSystemSimulation(sistema)


from pyswarm import pso  # Requiere instalación: pip install pyswarm

############# Entrenamiento ANFIS #############

# Definir datos de entrenamiento
X_train = normalized_features
y_train = labels

# Antes de la optimización, convertir a numpy arrays
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)


# Configurar número de procesos (ajustar según tu CPU)
N_PROCESOS = max(cpu_count() - 2, 2)

def calcular_error_chunk(args):
    """Función auxiliar para multiprocessing"""
    chunk, params = args
    # Crear una copia local del simulador para cada proceso
    local_simulador = ctrl.ControlSystemSimulation(sistema)
    
    # Aplicar parámetros a las funciones de membresía

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

    
    error_chunk = 0
    for x, y in chunk:
        local_simulador.input['contraste'] = x[0]
        local_simulador.input['homogeneidad'] = x[2]
        local_simulador.input['entropia'] = x[4]
        local_simulador.input['asm'] = x[1]
        local_simulador.input['energia'] = x[3]
        local_simulador.input['varianza'] = x[5]

        local_simulador.compute()
        y_pred = local_simulador.output['diagnostico']
        error_chunk += (y_pred - y)**2
    return error_chunk / len(chunk)

# Función de pérdida ajustada para 24 parámetros (6 variables × 4 parámetros)
def error_func(params):
    global iteration_count
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
        
        simulador.compute()
        # Modificar en error_func:
        y_pred = np.clip(simulador.output['diagnostico'], 1e-10, 1-1e-10)
        error_total += - (y_train[i] * np.log(y_pred) + (1 - y_train[i]) * np.log(1 - y_pred))

    error_promedio = error_total / len(X_train)
    print(f"Iteración {iteration_count} - Error: {error_promedio:.4f} - Tiempo: {time.time() - start_error:.2f}s")
    #error_total / len(X_train)
    return error_total/len(X_train)

# Límites para PSO (24 parámetros)
lb = [-3, 0.3] * 12  # 12 pares de parámetros
ub = [3, 0.8] * 12

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
############# Predicción y Evaluación Final #############

print("\nRealizando predicciones...")
start_pred = time.time()
y_pred = []
total_samples = len(X_train)


# Aplicar todos los parámetros optimizados
contraste['bajo'].mf = fuzz.gaussmf(contraste.universe, params_opt[0], params_opt[1])
contraste['alto'].mf = fuzz.gaussmf(contraste.universe, params_opt[2], params_opt[3])
homogeneidad['bajo'].mf = fuzz.gaussmf(homogeneidad.universe, params_opt[4], params_opt[5])
homogeneidad['alto'].mf = fuzz.gaussmf(homogeneidad.universe, params_opt[6], params_opt[7])
entropia['bajo'].mf = fuzz.gaussmf(entropia.universe, params_opt[8], params_opt[9])
entropia['alto'].mf = fuzz.gaussmf(entropia.universe, params_opt[10], params_opt[11])
asm['bajo'].mf = fuzz.gaussmf(asm.universe, params_opt[12], params_opt[13])
asm['alto'].mf = fuzz.gaussmf(asm.universe, params_opt[14], params_opt[15])
energia['bajo'].mf = fuzz.gaussmf(energia.universe, params_opt[16], params_opt[17])
energia['alto'].mf = fuzz.gaussmf(energia.universe, params_opt[18], params_opt[19])
varianza['bajo'].mf = fuzz.gaussmf(varianza.universe, params_opt[20], params_opt[21])
varianza['alto'].mf = fuzz.gaussmf(varianza.universe, params_opt[22], params_opt[23])

# Predecir con todas las características
y_pred = []
for idx, x in enumerate(X_train):
    if idx % 500 == 0:
        elapsed = time.time() - start_pred
        print(f"Prediciendo {idx}/{total_samples} - Tiempo: {elapsed:.2f}s")
        sys.stdout.flush()
    simulador.input['contraste'] = x[0]
    simulador.input['homogeneidad'] = x[2]
    simulador.input['entropia'] = x[4]
    simulador.input['asm'] = x[1]
    simulador.input['energia'] = x[3]
    simulador.input['varianza'] = x[5]
    
    simulador.compute()
    y_pred.append(1 if simulador.output['diagnostico'] > 0.5 else 0)
    
print(f"\nPredicción completada. Tiempo: {time.time() - start_pred:.2f}s")
# Métricas extendidas
from sklearn.metrics import classification_report, confusion_matrix

print("\nReporte de Clasificación:")
print(classification_report(y_train, y_pred, target_names=['No Tumor', 'Tumor']))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_train, y_pred))
