# pridiccion.py

import time
import sys
import skfuzzy as fuzz
from sklearn.metrics import classification_report, confusion_matrix

from procesamiento_image import process_all_images
from Training_ANFIS import train_anfis
from ANFIS_model import contraste, homogeneidad, entropia, asm, energia, varianza, media, simulador

def predict(params_opt, X_train):
    # Aplicar los parámetros óptimos a cada función de membresía
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
    media['bajo'].mf = fuzz.gaussmf(media.universe, params_opt[24], params_opt[25])
    media['alto'].mf = fuzz.gaussmf(media.universe, params_opt[26], params_opt[27])

    y_pred = []
    total_samples = len(X_train)
    start_pred = time.time()

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
        simulador.input['media'] = x[6]
        
        simulador.compute()
        y_pred.append(1 if simulador.output['diagnostico'] > 0.5 else 0)
        
    print(f"\nPredicción completada. Tiempo: {time.time() - start_pred:.2f}s")
    return y_pred

def main():
    # 1. Preprocesar imágenes y obtener datos de entrenamiento
    X_train, y_train = process_all_images()

    # 2. Entrenar el ANFIS con PSO
    params_opt = train_anfis(X_train, y_train)

    # 3. Realizar predicciones
    y_pred = predict(params_opt, X_train)

    # 4. Reporte de resultados
    print("\nReporte de Clasificación:")
    print(classification_report(y_train, y_pred, target_names=['No Tumor', 'Tumor']))
    
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_train, y_pred))

"""     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
 """
if __name__ == "__main__":
    main()
