## Preprocesamiento de imágenes y extracción de características GLCM ##

import os
import time
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler


def extract_glcm_features(i,image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Realiza la extracion de la mascara y recorta el ROI
    Extrae 6 características GLCM válidas.
    """

    enhanced = image["enhanced"]
    mask = image["segmentation_mask"]

    # Aplicar máscara y recortar ROI
    roi = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    x,y,w,h = cv2.boundingRect(mask)
    roi_cropped = roi[y:y+h, x:x+w]  # Enfocar solo en región segmentada

    #Visualizacion de prueba

    #if i % 5 == 0:  # Ejemplo: visualizar cada 100 imágenes
        #plt.imshow(roi_cropped, cmap='gray')
        #plt.show()
    


    if np.std(mask) == 0:
        raise ValueError("Imagen uniforme: no se puede calcular GLCM")
    

    # Calcular GLCM

    glcm = graycomatrix(roi_cropped, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    try:
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        mean = graycoprops(glcm, 'mean')[0, 0]
        entropy = graycoprops(glcm, 'entropy')[0, 0] 
        variance = graycoprops(glcm, 'variance')[0, 0]

    except Exception as e:
        raise ValueError(f"Error en GLCM: {e}")
    
    return np.array([contrast, asm, homogeneity, energy, mean, entropy, variance])



def preprocess_image(i,image_path):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Visualización (opcional)
    #if i % 5 == 0:  # Ejemplo: visualizar cada 100 imágenes
        
        #plt.imshow(blurred, cmap='gray')
        #plt.imshow(seg_mask, cmap='gray')
        #plt.imshow(seg_mask, cmap='gray')
        #plt.show()


    return {"enhanced": enhanced, "segmentation_mask": seg_mask}
    




def process_all_images(base_dir="./archive/test_3"):
    """Recorre imágenes en directorio con detección automática de prefijos"""
    # Intentar diferentes patrones de búsqueda
    patterns_to_try = [
        # Patrones para entrenamiento
        (os.path.join(base_dir, "meningioma", "Tr-me_*.jpg"), 
         os.path.join(base_dir, "notumor", "Tr-no_*.jpg")),
        
        # Patrones para prueba
        (os.path.join(base_dir, "meningioma", "Te-me_*.jpg"), 
         os.path.join(base_dir, "notumor", "Te-no_*.jpg")),
        
        # Patrón genérico como respaldo
        (os.path.join(base_dir, "meningioma", "*.jpg"), 
         os.path.join(base_dir, "notumor", "*.jpg")),
        
        # Incluir mayúsculas/minúsculas
        (os.path.join(base_dir, "meningioma", "*.[jJ][pP][gG]"), 
         os.path.join(base_dir, "notumor", "*.[jJ][pP][gG]"))
    ]
    
    image_paths = []
    for meningioma_pattern, notumor_pattern in patterns_to_try:
        if not image_paths:  # Solo si aún no hay rutas
            image_paths = glob.glob(meningioma_pattern) + glob.glob(notumor_pattern)
    
    # Mensaje informativo si no encuentra imágenes
    if len(image_paths) == 0:
        print(f"\n⚠️ ADVERTENCIA: No se encontraron imágenes en {base_dir}")
        print("Patrones intentados:")
        for i, (m, n) in enumerate(patterns_to_try):
            print(f"  {i+1}. Meningioma: {m}")
            print(f"     Notumor:   {n}")
        return np.zeros((0, 7)), []  # Retorno seguro

    features = []
    labels = []
    error_count = 0
    start_time = time.time()

    print("\nProcesando imágenes:")
    for i, path in enumerate(image_paths):
        try:
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\nImagen {i+1}/{len(image_paths)} - Tiempo: {elapsed:.2f}s")
                sys.stdout.flush()
            
            processed_img = preprocess_image(i, path)
            glcm_features = extract_glcm_features(i,processed_img)
            features.append(glcm_features)
            labels.append(1 if "meningioma" in path.lower() else 0)
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
    # (opcional) Mostrar ejemplo de características
    print("Ejemplo de características normalizadas:\n", normalized_features[0])
    
    return normalized_features, labels

if __name__ == "__main__":
    # Para probar de forma independiente
    X_train, y_train = process_all_images()