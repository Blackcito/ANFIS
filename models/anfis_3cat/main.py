# ejemplo_uso_validacion.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from procesamiento_image import process_all_images
from Training_ANFIS import train_anfis_multiclass
from prediccion_analisis import (
    generar_reporte_completo,
    predict_sugeno,
    comparar_predicciones_por_clase,
    crear_visualizacion_reglas_discriminativas,
    predict_con_explicacion,
    predict_sugeno_multiclass,
    generar_reporte_completo_multiclass,
    predict_con_explicacion_multiclass
)
from analisis import AnalizadorReglasANFIS

def cargar_datos_train_test(train_dir="./archive/categorica/Training", test_dir="./archive/categorica/Testing"):
    """Carga características sin normalizar y aplica StandardScaler solo una vez usando train."""
    X_train, y_train = process_all_images(base_dir=train_dir)
    X_test,  y_test  = process_all_images(base_dir=test_dir)

    if X_train.size == 0:
        raise RuntimeError(f"No hay imágenes extraídas en {train_dir}. Revisa rutas y extensiones.")
    if len(y_train) == 0:
        raise RuntimeError(f"No hay etiquetas en {train_dir}.")

    # convertir etiquetas a arrays
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test, dtype=int) if len(y_test) > 0 else np.array([])

    # Normalización usando solo train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if X_test.size == 0:
        print(f"⚠️ ADVERTENCIA: No hay imágenes en {test_dir}. Se devuelve X_test vacío.")
        X_test = np.zeros((0, X_train.shape[1]))
    else:
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def entrenar_y_evaluar_multiclass():
    """Pipeline completo de entrenamiento y evaluación multiclase"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test()
    
    # Entrenar ANFIS multiclase
    mf_opt, theta_opt = train_anfis_multiclass(X_train, y_train, swarmsize=30, maxiter=10)
    
    # Evaluar sobre test
    y_pred_test, y_probs_test = predict_sugeno_multiclass(X_test, mf_opt, theta_opt)
    
    # Reporte completo sobre test
    resultados_test = generar_reporte_completo_multiclass(X_test, y_test, mf_opt, theta_opt, save_plots=True)
    
    return {
        "modelo": {"mf_params": mf_opt, "theta": theta_opt},
        "resultados_test": resultados_test,
        "y_pred_test": y_pred_test,
        "y_probs_test": y_probs_test
    }
    

    return {
        "modelo": {"mf_params": mf_opt, "theta": theta_opt},
        "resultados_test": resultados_test,
        "top_reglas_test": top_reglas_test,
        "comparacion_clases_test": comparacion
    }

def ejemplo_prediccion_individual_multiclass(idx=0):
    """Predicción explicada de un caso específico del test (multiclase)"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test()
    mf_opt, theta_opt = train_anfis_multiclass(X_train, y_train, swarmsize=30, maxiter=10)
    
    x_sample = X_test[idx]
    y_real = y_test[idx]
    
    explicacion = predict_con_explicacion_multiclass(x_sample, mf_opt, theta_opt, n_top_reglas=5)
    
    nombres_clases = ['Meningioma', 'No Tumor', 'Pituitaria']
    nombres_caracteristicas = ['Contraste','ASM','Homogeneidad','Energía','Media','Entropía','Varianza']
    
    print(f"\nMuestra #{idx}:")
    print(f"Etiqueta real: {nombres_clases[y_real]}")
    print(f"Predicción: {nombres_clases[explicacion['prediccion_binaria']]}")
    print(f"Probabilidades: {[f'{p:.4f}' for p in explicacion['prediccion_continua']]}")
    
    for clase_id, reglas_clase in enumerate(explicacion['top_reglas_activas']):
        print(f"\nTop reglas para {nombres_clases[clase_id]}:")
        for i, regla_activa in enumerate(reglas_clase):
            condiciones = [f"{nombres_caracteristicas[j]}={etq}" for j, etq in enumerate(regla_activa['regla'])]
            print(f"{i+1}. Regla {regla_activa['regla_idx']} -> {' & '.join(condiciones)} | "
                  f"Activación={regla_activa['activacion']:.3f} Contribución={regla_activa['contribucion']:.3f}")

def menu_interactivo():
    print("\n==== MENÚ ANFIS - VALIDACIÓN CON TEST ====")
    print("1. Entrenar y evaluar pipeline completo")
    print("2. Predicción explicada de un caso individual")
    print("3. Salir")
    
    opcion = input("Ingresa opción (1-3): ").strip()
    if opcion=="1":
        resultados = entrenar_y_evaluar_multiclass()
        print("\n✅ Pipeline completo ejecutado. Revisa las gráficas y reportes generados.")
    elif opcion=="2":
        idx = int(input("Ingresa índice de muestra del test: "))
        ejemplo_prediccion_individual_multiclass(idx)
    elif opcion=="3":
        print("👋 Saliendo...")
        return
    else:
        print("Opción inválida")
    menu_interactivo()

if __name__=="__main__":
    menu_interactivo()
