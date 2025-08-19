# ejemplo_uso_validacion.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from procesamiento_image import process_all_images
from Training_ANFIS import train_anfis
from prediccion_analisis import (
    generar_reporte_completo,
    predict_sugeno,
    comparar_predicciones_por_clase,
    crear_visualizacion_reglas_discriminativas,
    predict_con_explicacion
)
from analisis import AnalizadorReglasANFIS

def cargar_datos_train_test(train_dir="./archive/test_3", test_dir="./archive/train_1"):
    """Carga datos de train y test"""
    X_train, y_train = process_all_images(base_dir="./archive/test_2")
    X_test,  y_test  = process_all_images(base_dir="./archive/test_1")
    
    # Normalización usando solo train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def entrenar_y_evaluar():
    """Pipeline completo de entrenamiento y evaluación en test"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test()
    
    # Entrenar ANFIS
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    # Evaluar sobre test
    y_cont_test, y_bin_test = predict_sugeno(X_test, mf_opt, theta_opt)
    
    # Reporte completo sobre test
    resultados_test = generar_reporte_completo(X_test, y_test, mf_opt, theta_opt, save_plots=True)
    
    # Análisis de reglas
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_test, y_test)
    top_reglas_test = analizador.obtener_top_reglas(10)
    
    # Comparación de reglas por clase
    comparacion = comparar_predicciones_por_clase(X_test, y_test, mf_opt, theta_opt)
    

    return {
        "modelo": {"mf_params": mf_opt, "theta": theta_opt},
        "resultados_test": resultados_test,
        "top_reglas_test": top_reglas_test,
        "comparacion_clases_test": comparacion
    }

def ejemplo_prediccion_individual(idx=0):
    """Predicción explicada de un caso específico del test"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test()
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    x_sample = X_test[idx]
    y_real = y_test[idx]
    
    explicacion = predict_con_explicacion(x_sample, mf_opt, theta_opt, n_top_reglas=5)
    
    nombres_caracteristicas = ['Contraste','ASM','Homogeneidad','Energía','Media','Entropía','Varianza']
    print(f"\nMuestra #{idx}:")
    print(f"Etiqueta real: {'Tumor' if y_real==1 else 'No Tumor'}")
    print(f"Predicción: {'Tumor' if explicacion['prediccion_binaria']==1 else 'No Tumor'}")
    print(f"Confianza: {explicacion['prediccion_continua']:.4f}")
    
    for i, regla_activa in enumerate(explicacion['top_reglas_activas']):
        condiciones = [f"{n}={e}" for n, e in zip(nombres_caracteristicas, regla_activa['regla'])]
        print(f"{i+1}. Regla {regla_activa['regla_idx']} -> {' & '.join(condiciones)} | "
              f"Activación={regla_activa['activacion']:.3f} Contribución={regla_activa['contribucion']:.3f}")

def menu_interactivo():
    print("\n==== MENÚ ANFIS - VALIDACIÓN CON TEST ====")
    print("1. Entrenar y evaluar pipeline completo")
    print("2. Predicción explicada de un caso individual")
    print("3. Salir")
    
    opcion = input("Ingresa opción (1-3): ").strip()
    if opcion=="1":
        resultados = entrenar_y_evaluar()
        print("\n✅ Pipeline completo ejecutado. Revisa las gráficas y reportes generados.")
    elif opcion=="2":
        idx = int(input("Ingresa índice de muestra del test: "))
        ejemplo_prediccion_individual(idx)
    elif opcion=="3":
        print("👋 Saliendo...")
        return
    else:
        print("Opción inválida")
    menu_interactivo()

if __name__=="__main__":
    menu_interactivo()
