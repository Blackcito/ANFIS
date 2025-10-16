# main.py - VERDADERAMENTE delgado

import numpy as np
from sklearn.preprocessing import StandardScaler
from features.procesamiento_image import process_all_images, gestionar_cache

# Importaciones modularizadas
from core.training import train_anfis
from analysis.evaluador import EvaluadorANFIS
from analysis.explicador import ExplicadorANFIS
from analysis.analisis import AnalizadorReglasANFIS

def cargar_datos_train_test(train_dir="./archive/binaria/Training", test_dir="./archive/binaria/Testing", use_cache=True):
    """Carga características usando sistema de caché"""
    print(" Cargando datos de entrenamiento...")
    X_train, y_train = process_all_images(base_dir=train_dir, normalize=False, use_cache=use_cache)
    
    print(" Cargando datos de prueba...")
    X_test, y_test = process_all_images(base_dir=test_dir, normalize=False, use_cache=use_cache)

    if X_train.size == 0:
        raise RuntimeError(f"No hay imágenes extraídas en {train_dir}")
    
    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) if X_test.size > 0 else np.zeros((0, X_train.shape[1]))

    return X_train, y_train, X_test, y_test

def pipeline_completo(use_cache=True):
    """Pipeline completo usando módulos especializados"""
    # 1. Cargar datos
    X_train, y_train, X_test, y_test = cargar_datos_train_test(use_cache=use_cache)
    
    # 2. Entrenar modelo
    print(" Entrenando modelo ANFIS...")
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    modelo = {'mf_params': mf_opt, 'theta': theta_opt}
    
    # 3. Evaluar modelo
    print(" Evaluando modelo...")
    evaluador = EvaluadorANFIS(modelo, {'X': X_test, 'y': y_test})
    resultados_eval = evaluador.evaluar_modelo()
    
    # 4. Analizar reglas
    print(" Analizando reglas...")
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_test, y_test)
    analizador.generar_analisis_completo("analisis_reglas_anfis")
    
    # 5. Explicar predicciones
    print(" Generando explicaciones...")
    explicador = ExplicadorANFIS(modelo)
    comparacion_clases = explicador.comparar_por_clase(X_test, y_test)
    
    return {
        "modelo": modelo,
        "evaluacion": resultados_eval,
        "comparacion_clases": comparacion_clases
    }

def prediccion_individual(idx=0, use_cache=True):
    """Predicción explicada de un caso específico"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test(use_cache=use_cache)
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    modelo = {'mf_params': mf_opt, 'theta': theta_opt}
    
    explicador = ExplicadorANFIS(modelo)
    explicacion = explicador.explicar_prediccion(X_test[idx], n_top_reglas=5)
    
    return explicacion

def menu_interactivo():
    print("\n==== MENÚ ANFIS - ESTRUCTURA MODULAR ====")
    print("1. Pipeline completo (con caché)")
    print("2. Pipeline completo (sin caché)")
    print("3. Predicción individual explicada")
    print("4. Gestionar caché")
    print("5. Salir")
    
    opcion = input("Ingresa opción (1-5): ").strip()
    
    if opcion == "1":
        resultados = pipeline_completo(use_cache=True)
        print("\n✅ Pipeline ejecutado")
    elif opcion == "2":
        resultados = pipeline_completo(use_cache=False)
        print("\n✅ Pipeline ejecutado (sin caché)")
    elif opcion == "3":
        idx = int(input("Índice de muestra: "))
        explicacion = prediccion_individual(idx)
        print(f"Predicción: {explicacion['prediccion_binaria']}")
    elif opcion == "4":
        gestionar_cache()
    elif opcion == "5":
        return
    else:
        print("Opción inválida")
    
    menu_interactivo()

if __name__ == "__main__":
    menu_interactivo()