# predict_sugeno.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from procesamiento_image import process_all_images  # tu extractor GLCM
from Training_ANFIS import train_anfis
from anfis_sugeno import compute_weights, n_vars, reglas

def predict_sugeno(X, mf_params, theta, threshold=0.5):
    """
    Devuelve las salidas continuas y binarias (umbral > threshold)
    según el ANFIS-Sugeno entrenado.
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

if __name__ == "__main__":
    # 1) Preprocesar imágenes para obtener X_train, y_train
    X_train, y_train = process_all_images()

    # 2) Entrenar el ANFIS-Sugeno (PSO + LSE)
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)

    # 3) Predecir con el modelo entrenado
    y_cont, y_pred = predict_sugeno(X_train, mf_opt, theta_opt)

    # 4) Evaluar
    print("\nReporte de Clasificación:")
    print(classification_report(y_train, y_pred, target_names=['No Tumor','Tumor']))
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_train, y_pred))
