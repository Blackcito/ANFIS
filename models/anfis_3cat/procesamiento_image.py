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


def process_all_images(base_dir="./archive/categorica/", save_dir=None, save_images=False):
    """
    Lee todas las imágenes válidas en:
      base_dir/meningioma/*
      base_dir/notumor/*
      base_dir/pituitaria/*
    Devuelve (features_raw, labels) sin normalizar.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    men_dir = os.path.join(base_dir, "meningioma")
    no_dir  = os.path.join(base_dir, "notumor")
    pit_dir = os.path.join(base_dir, "pituitary")  # Nueva categoría

    # Patrones para buscar imágenes en diferentes formatos
    patterns_to_try = [
        (os.path.join(men_dir, "Tr-me_*.jpg"),
         os.path.join(no_dir, "Tr-no_*.jpg"),
         os.path.join(pit_dir, "Tr-pi_*.jpg")),
        (os.path.join(men_dir, "Te-me_*.jpg"),
         os.path.join(no_dir, "Te-no_*.jpg"),
         os.path.join(pit_dir, "Te-pi_*.jpg")),
        (os.path.join(men_dir, "*.jpg"),
         os.path.join(no_dir, "*.jpg"),
         os.path.join(pit_dir, "*.jpg")),
        (os.path.join(men_dir, "*.[jJ][pP][gG]"),
         os.path.join(no_dir, "*.[jJ][pP][gG]"),
         os.path.join(pit_dir, "*.[jJ][pP][gG]")),
        (os.path.join(men_dir, "*.[pP][nN][gG]"),
         os.path.join(no_dir, "*.[pP][nN][gG]"),
         os.path.join(pit_dir, "*.[pP][nN][gG]")),
        (os.path.join(men_dir, "*.[tT][iI][fF]"),
         os.path.join(no_dir, "*.[tT][iI][fF]"),
         os.path.join(pit_dir, "*.[tT][iI][fF]")),
        (os.path.join(men_dir, "*.[tT][iI][fF][fF]"),
         os.path.join(no_dir, "*.[tT][iI][fF][fF]"),
         os.path.join(pit_dir, "*.[tT][iI][fF][fF]"))
    ]

    image_paths = []
    for men_pattern, no_pattern, pit_pattern in patterns_to_try:
        if not image_paths:
            image_paths = glob.glob(men_pattern) + glob.glob(no_pattern) + glob.glob(pit_pattern)

    # Eliminar duplicados
    image_paths = list(set(image_paths))

    if len(image_paths) == 0:
        print(f"\n⚠️ ADVERTENCIA: No se encontraron imágenes en {base_dir}")
        print("Rutas inspeccionadas:")
        print(" ", men_dir)
        print(" ", no_dir)
        print(" ", pit_dir)
        return np.zeros((0, 7)), []

    features = []
    labels = []
    error_count = 0
    start_time = time.time()

    print(f"\nProcesando {len(image_paths)} imágenes en {base_dir}:")
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
            
            # Determinar la etiqueta según la carpeta
            if "meningioma" in path.lower():
                labels.append(0)  # Meningioma
            elif "notumor" in path.lower():
                labels.append(1)  # No tumor
            elif "pituitary" in path.lower():
                labels.append(2)  # Pituitaria
            else:
                # Intentar inferir por la ruta
                if 'meningioma' in path:
                    labels.append(0)
                elif 'notumor' in path:
                    labels.append(1)
                elif 'pituitaria' in path:
                    labels.append(2)
                else:
                    print(f"Advertencia: No se pudo determinar la clase de {path}. Se omite.")
                    continue
                
        except Exception as e:
            error_count += 1
            print(f"\nERROR en imagen {i+1}: {path}")
            print(f"  {str(e)}")
            print("  Saltando imagen...")
            continue

    print(f"\nProcesamiento completado. Errores: {error_count}/{len(image_paths)}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

    # Devolver características sin normalizar (la normalización se hará después)
    features = np.array(features, dtype=float)
    return features, labels


if __name__ == "__main__":
    # Ejemplos:
    # Guardar imágenes:
    # X, y = process_all_images(base_dir="./archive/test_2", save_dir="./debug_images", save_images=True)
    # No guardar:
    # X, y = process_all_images(base_dir="./archive/test_2", save_images=False)

    X_train, y_train = process_all_images(base_dir="./archive/categorica/test_1", save_dir="./debug_images", save_images=True)
