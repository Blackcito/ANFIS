# features/procesamiento_image.py - CORREGIDO

import os
import cv2
import numpy as np
import glob
import time
import sys
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# Importar configuración y caché
from config.configuracion import config
from utils.cache import sistema_cache

def save_step_image(output_dir, stage, filename, image, save_images=True):
    if not save_images or image is None:
        return
    out_dir = os.path.join(output_dir, stage)
    os.makedirs(out_dir, exist_ok=True)
    if image.dtype != np.uint8:
        image_to_save = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
    else:
        image_to_save = image
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, image_to_save)

def preprocess_image(i, image_path, save_dir=None, save_images=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    img_name = os.path.basename(image_path)
    if save_dir:
        save_step_image(save_dir, "01_original", img_name, img, save_images)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    if save_dir:
        save_step_image(save_dir, "02_enhanced", img_name, enhanced, save_images)

    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    if save_dir:
        save_step_image(save_dir, "03_blurred", img_name, blurred, save_images)

    seg_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if save_dir:
        save_step_image(save_dir, "04_mask", img_name, seg_mask, save_images)

    return {"enhanced": enhanced, "segmentation_mask": seg_mask}

def extract_glcm_features(i, image, save_dir=None, filename=None,
                          distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=256, save_images=False):
    if filename is None:
        filename = f"img_{i:06d}.png"

    enhanced = image["enhanced"]
    mask = image["segmentation_mask"]

    roi = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    x,y,w,h = cv2.boundingRect(mask)
    roi_cropped = roi[y:y+h, x:x+w]

    if save_dir:
        save_step_image(save_dir, "05_roi", filename, roi_cropped, save_images)

    if np.std(mask) == 0:
        raise ValueError("Imagen uniforme: no se puede calcular GLCM")

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

def process_all_images(tumor_dir=None, notumor_dir=None, save_dir=None, save_images=None):
    """
    Procesa imágenes con sistema de caché simple para características
    
    Args:
        tumor_dir: Directorio con las imágenes de tumor
        notumor_dir: Directorio con las imágenes de no tumor
        save_dir: Directorio para guardar imágenes de debug
        save_images: Si guarda imágenes intermedias
    """
    # CORRECCIÓN: Usar parámetros si se proporcionan, sino usar configuración
    if save_images is None:
        save_images = config.procesamiento.guardar_imagenes_intermedias

    print(f" Procesando imágenes con configuración:")
    print(f"   Directorio tumor: {tumor_dir}")
    print(f"   Directorio no-tumor: {notumor_dir}")
    print(f"   Guardar imágenes: {save_images}")

    # 2. Si no hay caché válido, procesar imágenes
    print(" Procesando imágenes desde disco...")
    
    image_paths = []
    labels = []

    # Buscar imágenes en directorio de tumor
    if tumor_dir and os.path.exists(tumor_dir):
        tumor_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        for pattern in tumor_patterns:
            tumor_paths = glob.glob(os.path.join(tumor_dir, pattern))
            for path in tumor_paths:
                image_paths.append(path)
                labels.append(1)  # 1 = tumor

    # Buscar imágenes en directorio de no-tumor
    if notumor_dir and os.path.exists(notumor_dir):
        notumor_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        for pattern in notumor_patterns:
            notumor_paths = glob.glob(os.path.join(notumor_dir, pattern))
            for path in notumor_paths:
                image_paths.append(path)
                labels.append(0)  # 0 = no tumor

    if len(image_paths) == 0:
        print("\n ADVERTENCIA: No se encontraron imágenes en los directorios especificados")
        return np.zeros((0, 7)), []

    # Ordenar paths para tener un orden consistente
    image_paths_with_labels = list(zip(image_paths, labels))
    image_paths_with_labels.sort(key=lambda x: x[0])  # Ordenar por path
    image_paths, labels = zip(*image_paths_with_labels)
    image_paths, labels = list(image_paths), list(labels)

    features = []
    error_count = 0
    start_time = time.time()

    print("\nProcesando imágenes:")
    for i, path in enumerate(sorted(image_paths)):
        try:
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f" Progreso: {i+1}/{len(image_paths)} imágenes ({timedelta(seconds=int(elapsed))})")
                sys.stdout.flush()

            processed_img = preprocess_image(i, path, save_dir=save_dir, save_images=save_images)
            fname = os.path.basename(path)
            glcm_features = extract_glcm_features(i, processed_img, save_dir=save_dir, filename=fname, save_images=save_images)
            features.append(glcm_features)
        except Exception as e:
            error_count += 1
            print(f" Error en imagen {i+1}: {os.path.basename(path)} - {str(e)}")
            continue

    print(f" Procesamiento completado: {len(features)} éxitos, {error_count} errores")
    total_time = time.time() - start_time
    print(f" Tiempo total: {timedelta(seconds=int(total_time))}")

    # Convertir a arrays numpy y verificar dimensiones
    features = np.array(features) if features else np.zeros((0,7))
    labels = np.array(labels)

    # Debug: imprimir dimensiones
    print(f"\nDimensiones de los datos:")
    print(f" - Features shape: {features.shape}")
    print(f" - Labels shape: {labels.shape}")
    print(f" - Tipos de datos: features={features.dtype}, labels={labels.dtype}")

    if len(features) != len(labels):
        raise ValueError(f"Dimensiones incompatibles: {len(features)} features vs {len(labels)} labels")

    return features, labels

if __name__ == "__main__":
    # Ejemplos de uso:
    # Con rutas explícitas para tumor y no-tumor:
    X, y = process_all_images(
        tumor_dir="./archive/test_2/tumor",
        notumor_dir="./archive/test_2/notumor"
    )