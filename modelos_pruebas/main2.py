import os
import time
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

def extract_glcm_features(image):
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    
    features = []
    props = ['contrast', 'ASM', 'homogeneity', 'energy', 'entropy', 'variance', 'correlation']
    for prop in props:
        temp = graycoprops(glcm, prop)
        features.extend([temp.mean(), temp.std()])  # Media y desviación
        
    # Característica adicional: disimilitud
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    features.append(dissimilarity)
    
    return np.array(features)



def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. CLAHE para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    
    # 2. Suavizado adaptativo
    blurred = cv2.bilateralFilter(clahe_img, 9, 75, 75)
    
    # 3. Umbralización adaptativa mejorada
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # 4. Operaciones morfológicas optimizadas
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return morph

# Procesar todas las imágenes de ambos directorios


# Configurar rutas base
base_dir = "D:/Google Drive/universidad/Tesis/Codigos/python/CNN_ANFIS/archive/test_2"
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

# Añadir:
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Reducción de dimensionalidad
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(normalized_features)

# 2. Selección de características
# Error original:
selector = SelectKBest(f_classif, k=12)

# Corrección (ajustar k al número real de características):
n_features_after_pca = pca_features.shape[1]
k_value = min(12, n_features_after_pca)
selector = SelectKBest(f_classif, k=k_value)
selected_features = selector.fit_transform(pca_features, labels)

############# Entrenamiento ANFIS #############

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir datos de entrenamiento
X_train = selected_features
y_train = labels
# --------------------------------------------------
# Sección ANFIS completamente funcional
# --------------------------------------------------

# 1. Calcular correlaciones válidas
correlations = []
for i in range(X_train.shape[1]):
    corr_coef = np.corrcoef(X_train[:, i], y_train)[0, 1]
    correlations.append(corr_coef)

# 2. Filtrar características relevantes
# 1. Mejorar la generación de reglas (umbral adaptativo)
corr_threshold = np.percentile(np.abs(correlations), 75)  # Usar percentil 75
selected_indices = [i for i, corr in enumerate(correlations) if abs(corr) > corr_threshold]
X_train_filtered = X_train[:, selected_indices]
n_features = len(selected_indices)

from imblearn.over_sampling import SMOTE


smote = SMOTE(sampling_strategy='minority', random_state=42)
X_res, y_res = smote.fit_resample(X_train_filtered, y_train)

# Reemplazar datos de entrenamiento
X_train = X_res
y_train = y_res

# 3. Configurar sistema ANFIS
antecedents = [ctrl.Antecedent(np.arange(-3, 3, 0.1), f'feature_{i}') for i in range(n_features)]
diagnostico = ctrl.Consequent(np.arange(0, 1, 0.01), 'diagnostico')

# Membresías para antecedentes
for ant in antecedents:
    ant['low'] = fuzz.gaussmf(ant.universe, -1.5, 0.5)
    ant['medium'] = fuzz.gaussmf(ant.universe, 0, 0.5)
    ant['high'] = fuzz.gaussmf(ant.universe, 1.5, 0.5)

# Membresías del consecuente
diagnostico['tumor'] = fuzz.trimf(diagnostico.universe, [0, 0, 0.5])
diagnostico['no_tumor'] = fuzz.trimf(diagnostico.universe, [0.5, 1, 1])

# Generar reglas dinámicas
rules = []
for idx, i in enumerate(selected_indices):
    # Regla base
    if correlations[i] > 0:
        rules.append(ctrl.Rule(
            antecedents[idx]['low'] & antecedents[idx]['medium'], 
            diagnostico['no_tumor']
        ))
    else:
        rules.append(ctrl.Rule(
            antecedents[idx]['medium'] & antecedents[idx]['high'], 
            diagnostico['tumor']
        ))

# Crear sistema de control
sistema = ctrl.ControlSystem(rules)
simulador = ctrl.ControlSystemSimulation(sistema)
from pyswarm import pso

# Función de error actualizada
def error_func(params):
    global iteration_count
    iteration_count += 1
    
    for i, ant in enumerate(antecedents):
        start = i * 6  # 2 params por cada una de 3 funciones
        ant['low'].mf = fuzz.gaussmf(ant.universe, params[start], params[start+1])
        ant['medium'].mf = fuzz.gaussmf(ant.universe, params[start+2], params[start+3])
        ant['high'].mf = fuzz.gaussmf(ant.universe, params[start+4], params[start+5])
    
    error_total = 0
    for i, x in enumerate(X_train):
        for j, ant in enumerate(antecedents):
            simulador.input[ant.label] = x[j]
        try:
            simulador.compute()
            y_pred = simulador.output['diagnostico']
            error = (y_pred - y_train[i])**2
        except:
            error_total += error * 3 if y_train[i] == 1 else error
    
    return error_total / len(X_train)

# Configurar límites dinámicos
n_params = len(antecedents) * 6  # 3 funciones * 2 parámetros cada una
lb = [-3, 0.1] * (n_params // 2)
ub = [3, 1] * (n_params // 2)

# Variables de monitoreo PSO
iteration_count = 0

print("\nIniciando optimización PSO...")
start_pso = time.time()

# Optimización con más iteraciones
# Aumentar capacidad de exploración de PSO
from scipy.optimize import minimize

def optimizacion_hibrida():
    # Fase 1: Búsqueda global con PSO
    params_pso, _ = pso(error_func, lb, ub, swarmsize=50, maxiter=50)
    
    # Fase 2: Afinamiento local
    result = minimize(error_func, params_pso, method='Nelder-Mead', options={'maxiter':30})
    
    return result.x

params_opt = optimizacion_hibrida()
print(f"\nOptimización completada. Tiempo total: {time.time() - start_pso:.2f} segundos")
############# Predicción y Evaluación Final #############

print("\nRealizando predicciones...")
start_pred = time.time()
y_pred = []
total_samples = len(X_train)


# Predecir con todas las características
y_pred = []
for idx, x in enumerate(X_train):
    # Establecer todos los inputs dinámicos
    for j, ant in enumerate(antecedents):
        simulador.input[ant.label] = x[j]
    
    simulador.compute()
    y_pred.append(1 if simulador.output['diagnostico'] > 0.5 else 0)
    
print(f"\nPredicción completada. Tiempo: {time.time() - start_pred:.2f}s")
# Métricas extendidas
from sklearn.metrics import classification_report, confusion_matrix

print("\nReporte de Clasificación:")
print(classification_report(y_train, y_pred, target_names=['No Tumor', 'Tumor']))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_train, y_pred))