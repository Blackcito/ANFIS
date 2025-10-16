# --- sólo las funciones relevantes con flag para guardar ---
import os
import cv2
import numpy as np
import glob
import time
import sys
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

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


def process_all_images(base_dir="./archive/test_2", save_dir="./debug_images", 
                      save_images=False, normalize=False, use_cache=True):
    """
    Procesa imágenes con sistema de caché simple para características
    
    Args:
        base_dir: Directorio con las imágenes
        save_dir: Directorio para guardar imágenes de debug
        save_images: Si guarda imágenes intermedias
        normalize: Si normaliza características
        use_cache: Si usa caché para acelerar procesamiento
    """
    cache_dir = "./features_cache"
    # Crear nombre único basado en el directorio
    base_name = os.path.basename(os.path.normpath(base_dir))
    cache_file = os.path.join(cache_dir, f"{base_name}_features.npz")
    
    # 1. Intentar cargar desde caché
    if use_cache and os.path.exists(cache_file):
        print(f" Cargando características desde caché: {cache_file}")
        try:
            data = np.load(cache_file)
            features = data['features']
            labels = data['labels']
            
            # Verificación básica: el archivo no está vacío
            if len(features) > 0 and len(labels) > 0:
                print(f"   - {len(features)} muestras cargadas")
                
                if normalize:
                    scaler = StandardScaler()
                    features = scaler.fit_transform(features)
                    print("   - Características normalizadas")
                
                return features, labels
            else:
                print(" Caché vacío, reprocesando...")
        except Exception as e:
            print(f" Error cargando caché: {e}, reprocesando...")
    
    # 2. Si no hay caché válido, procesar imágenes
    print("🔄 Procesando imágenes desde disco...")
    
    patterns_to_try = [
        (os.path.join(base_dir, "meningioma", "Tr-me_*.jpg"),
         os.path.join(base_dir, "notumor", "Tr-no_*.jpg")),
        (os.path.join(base_dir, "meningioma", "Te-me_*.jpg"),
         os.path.join(base_dir, "notumor", "Te-no_*.jpg")),
        (os.path.join(base_dir, "meningioma", "*.jpg"),
         os.path.join(base_dir, "notumor", "*.jpg")),
        (os.path.join(base_dir, "meningioma", "*.[jJ][pP][gG]"),
         os.path.join(base_dir, "notumor", "*.[jJ][pP][gG]"))
    ]

    image_paths = []
    for meningioma_pattern, notumor_pattern in patterns_to_try:
        if not image_paths:
            image_paths = glob.glob(meningioma_pattern) + glob.glob(notumor_pattern)

    if len(image_paths) == 0:
        print(f"\n ADVERTENCIA: No se encontraron imágenes en {base_dir}")
        return np.zeros((0, 7)), []

    features = []
    labels = []
    error_count = 0
    start_time = time.time()

    print("\nProcesando imágenes:")
    for i, path in enumerate(sorted(image_paths)):
        try:
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Imagen {i+1}/{len(image_paths)} - {elapsed:.2f}s")
                sys.stdout.flush()

            processed_img = preprocess_image(i, path, save_dir=save_dir, save_images=save_images)
            fname = os.path.basename(path)
            glcm_features = extract_glcm_features(i, processed_img, save_dir=save_dir, filename=fname, save_images=save_images)
            features.append(glcm_features)
            labels.append(1 if "meningioma" in path.lower() else 0)
        except Exception as e:
            error_count += 1
            print(f"\nERROR en imagen {i+1}: {path}")
            print(f"  {str(e)}")
            print("  Saltando imagen...")
            continue

    print(f"\nProcesamiento completado. Errores: {error_count}/{len(image_paths)}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

    # Convertir a array
    features = np.array(features) if features else np.zeros((0,7))
    
    # 3. Guardar en caché si se solicita
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_file, features=features, labels=labels)
        print(f" Características guardadas en caché: {cache_file}")
        print(f"   - {len(features)} muestras")
        print(f"   - Tamaño: {os.path.getsize(cache_file) / 1024 / 1024:.2f} MB")
    
    # 4. Aplicar normalización si se solicita
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print("   - Características normalizadas")
    
    return features, labels


# Función para gestionar caché desde main.py
def gestionar_cache():
    """Gestiona archivos de caché de características"""
    cache_dir = "./features_cache"
    if not os.path.exists(cache_dir):
        print(" No hay archivos de caché")
        return
    
    archivos = os.listdir(cache_dir)
    print(f"\n Archivos en caché ({len(archivos)}):")
    for archivo in archivos:
        ruta = os.path.join(cache_dir, archivo)
        tamaño = os.path.getsize(ruta) / 1024 / 1024  # MB
        # Cargar metadata básica
        try:
            data = np.load(ruta)
            muestras = len(data['features'])
            print(f"  - {archivo} ({muestras} muestras, {tamaño:.2f} MB)")
        except:
            print(f"  - {archivo} (corrupto, {tamaño:.2f} MB)")
    
    if archivos:
        opcion = input("\n¿Limpiar caché? (s/n): ").strip().lower()
        if opcion == 's':
            for archivo in archivos:
                os.remove(os.path.join(cache_dir, archivo))
            print(" Caché limpiado")


if __name__ == "__main__":
    # Ejemplos de uso:
    # Con caché (recomendado):
    X, y = process_all_images(base_dir="./archive/test_2", use_cache=True)
    # Sin caché (forzar reprocesamiento):
    # X, y = process_all_images(base_dir="./archive/test_2", use_cache=False)