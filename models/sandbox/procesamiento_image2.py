import os
import time
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================
# Función: Preprocesamiento moderado de la imagen MRI
# ============================================================
def preprocess_image_new(image_path):
    """
    Realiza un preprocesamiento que mejora el contraste y conserva la textura:
      1. Lectura en escala de grises.
      2. Aplicación de CLAHE para realzar detalles.
      3. Suavizado bilateral para reducir ruido sin perder bordes.
      4. Umbral adaptativo para obtener una máscara de segmentación (no demasiado agresiva).
      5. Operaciones morfológicas suaves para eliminar ruido aislado.
    
    Retorna un diccionario con:
      - 'enhanced': imagen mejorada (para extracción de textura)
      - 'segmentation_mask': máscara binaria (para extracción morfológica)
    """
    # Leer imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    # 1. CLAHE: mejora el contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # 2. Suavizado bilateral: reduce ruido preservando bordes
    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Umbral adaptativo (invertido): se obtiene una máscara donde se destacan zonas de interés
    seg_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )
    
    # 4. Operaciones morfológicas suaves para quitar ruido muy pequeño
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return {"enhanced": enhanced, "segmentation_mask": seg_mask}

# ============================================================
# Función: Extracción de características texturales (GLCM)
# ============================================================
def extract_texture_features(image):
    """
    Extrae características de textura mediante GLCM de la imagen.
    Se usan varios valores de distancia y ángulos.
    Retorna un vector de características.
    """
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    features = []
    props = ['contrast', 'ASM', 'homogeneity', 'energy', 'entropy', 'variance', 'correlation']
    for prop in props:
        temp = graycoprops(glcm, prop)
        features.extend([temp.mean(), temp.std()])
    # Característica adicional: disimilitud
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    features.append(dissimilarity)
    
    return np.array(features)

# ============================================================
# Función: Extracción de características morfológicas
# ============================================================
def extract_morphological_features(mask):
    """
    A partir de la máscara de segmentación, extrae algunas medidas
    del contorno principal (mayor componente).
    Retorna: [área, perímetro, compactidad, aspecto (solidez)]
    Si no se encuentra contorno significativo, retorna ceros.
    """
    # Asegurarse de trabajar con máscara binaria 0/255
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(4)
    
    # Seleccionar el contorno de mayor área
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
    
    # Solidez: relación entre el área del contorno y el área de su convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return np.array([area, perimeter, compactness, solidity])

# ============================================================
# Función: Combina las características texturales y morfológicas
# ============================================================
def combine_features(texture_feat, morph_feat):
    """
    Concatena las características texturales y morfológicas.
    """
    return np.concatenate([texture_feat, morph_feat])

# ============================================================
# Bloque Principal
# ============================================================
def main():
    # Configurar rutas base (ajusta según tu estructura)
    base_dir = "D:/Google Drive/universidad/Tesis/Codigos/python/CNN_ANFIS/archive/test_2"
    meningioma_dir = os.path.join(base_dir, "meningioma", "Tr-me_*.jpg")
    notumor_dir = os.path.join(base_dir, "notumor", "Tr-no_*.jpg")
    
    # Obtener todas las rutas de imágenes
    image_paths = glob.glob(meningioma_dir) + glob.glob(notumor_dir)
    if len(image_paths) == 0:
        print("No se encontraron imágenes en las rutas especificadas.")
        return
    
    features = []
    labels = []
    error_count = 0
    start_time = time.time()
    
    print("\nProcesando imágenes:")
    for i, path in enumerate(image_paths):
        try:
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"\nImagen {i+1}/{len(image_paths)} - Tiempo: {elapsed:.2f}s")
                sys.stdout.flush()
            
            preproc = preprocess_image_new(path)
            # Para características texturales usamos la imagen mejorada
            texture_feat = extract_texture_features(preproc["enhanced"])
            # Para características morfológicas usamos la máscara de segmentación
            morph_feat = extract_morphological_features(preproc["segmentation_mask"])
            
            combined = combine_features(texture_feat, morph_feat)
            features.append(combined)
            
            # Etiqueta: 1 si el nombre del archivo contiene "meningioma", 0 si "notumor"
            label = 1 if "meningioma" in path.lower() else 0
            labels.append(label)
        
        except Exception as e:
            error_count += 1
            print(f"\nERROR en imagen {i+1}: {path}")
            print(f"Tipo error: {str(e)}")
            print("Saltando imagen...")
            continue
            
    features = np.array(features)
    labels = np.array(labels)
    print(f"\nProcesamiento completado. Errores: {error_count}/{len(image_paths)}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")
    print(f"Características extraídas, forma: {features.shape}")
    
    # Normalización de características
    scaler = StandardScaler()
    norm_features = scaler.fit_transform(features)
    print("\nEjemplo de características normalizadas (primer muestra):")
    print(norm_features[0])
    
    # Opcional: Reducción de dimensionalidad y selección de características
    pca = PCA(n_components=0.95)  # conservar 95% de varianza
    pca_features = pca.fit_transform(norm_features)
    n_features_after_pca = pca_features.shape[1]
    k_value = min(12, n_features_after_pca)
    selector = SelectKBest(f_classif, k=k_value)
    selected_features = selector.fit_transform(pca_features, labels)
    print(f"\nDespués de PCA y selección, forma de las características: {selected_features.shape}")
    
    # ============================================================
    # Ejemplo básico de sistema difuso (placeholder ANFIS)
    # ============================================================
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    
    # Para este ejemplo, tomamos dos características (las dos primeras)
    # Estas deberían ser las más discriminantes según tu análisis
    feat1 = ctrl.Antecedent(np.linspace(-3, 3, 61), 'feat1')
    feat2 = ctrl.Antecedent(np.linspace(-3, 3, 61), 'feat2')
    tumor_out = ctrl.Consequent(np.linspace(0, 1, 11), 'tumor')
    
    # Definir funciones de membresía para entradas
    feat1['bajo'] = fuzz.trimf(feat1.universe, [-3, -3, 0])
    feat1['alto'] = fuzz.trimf(feat1.universe, [0, 3, 3])
    feat2['bajo'] = fuzz.trimf(feat2.universe, [-3, -3, 0])
    feat2['alto'] = fuzz.trimf(feat2.universe, [0, 3, 3])
    
    # Funciones de membresía para salida
    tumor_out['no'] = fuzz.trimf(tumor_out.universe, [0, 0, 0.5])
    tumor_out['si'] = fuzz.trimf(tumor_out.universe, [0.5, 1, 1])
    
    # Reglas difusas
    rule1 = ctrl.Rule(feat1['alto'] & feat2['alto'], tumor_out['si'])
    rule2 = ctrl.Rule(feat1['bajo'] | feat2['bajo'], tumor_out['no'])
    
    sistema_ctrl = ctrl.ControlSystem([rule1, rule2])
    simulacion = ctrl.ControlSystemSimulation(sistema_ctrl)
    
    # Simulación para el primer ejemplo de características seleccionadas
    for i, sample_feat in enumerate(selected_features):
        # Tomamos las dos primeras características para cada muestra
        simulacion.input['feat1'] = sample_feat[0]
        simulacion.input['feat2'] = sample_feat[1]
        simulacion.compute()
        print(f"Imagen {i+1} - Probabilidad de tumor: {simulacion.output['tumor']}")
    
    # ============================================================
    # Visualización de preprocesamiento en una imagen de ejemplo
    # ============================================================
    for i, sample_path in enumerate(image_paths):
        preproc_example = preprocess_image_new(sample_path)
        plt.figure(figsize=(15, 4))
        plt.suptitle(f"Ejemplo de preprocesamiento - Imagen {i+1}")
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(preproc_example["enhanced"], cmap='gray')
        plt.title("Imagen mejorada (CLAHE)")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(preproc_example["segmentation_mask"], cmap='gray')
        plt.title("Máscara de segmentación")
        plt.axis("off")
        
        plt.show()

if __name__ == "__main__":
    main()
