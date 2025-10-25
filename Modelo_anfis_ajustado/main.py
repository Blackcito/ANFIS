# main.py - ACTUALIZADO CON CONFIGURACIÓN CENTRALIZADA

import numpy as np
from sklearn.preprocessing import StandardScaler
from features.procesamiento_image import process_all_images
from core.training import train_anfis
from analysis.evaluador import EvaluadorANFIS
from analysis.analisis import AnalizadorReglasANFIS

# Importar el sistema de configuración y caché
from config.configuracion import config
from utils.cache import sistema_cache

def cargar_datos_train_test(train_dir=None, test_dir=None, use_cache=None):
    """Carga características usando sistema de caché centralizado"""
    
    # Usar valores de configuración si no se especifican
    train_dir = train_dir or config.directorio_entrenamiento
    test_dir = test_dir or config.directorio_prueba
    use_cache = use_cache if use_cache is not None else config.cache.usar_cache_caracteristicas
    
    print(f"Cargando datos de entrenamiento desde: {train_dir}")
    X_train, y_train = process_all_images(
        base_dir=train_dir, 
        normalize=config.procesamiento.normalizar_caracteristicas, 
        use_cache=use_cache,
        save_images=config.procesamiento.guardar_imagenes_intermedias,
        save_dir=config.procesamiento.directorio_imagenes_intermedias
    )
    
    print(f"Cargando datos de prueba desde: {test_dir}")
    X_test, y_test = process_all_images(
        base_dir=test_dir, 
        normalize=config.procesamiento.normalizar_caracteristicas,
        use_cache=use_cache,
        save_images=config.procesamiento.guardar_imagenes_intermedias,
        save_dir=config.procesamiento.directorio_imagenes_intermedias
    )

    if X_train.size == 0:
        raise RuntimeError(f"No hay imágenes extraídas en {train_dir}")
    
    # Convertir a arrays numpy
    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int) if len(y_test) > 0 else np.array([])

    # Normalización (si no se hizo en process_all_images)
    if not config.procesamiento.normalizar_caracteristicas:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) if X_test.size > 0 else np.zeros((0, X_train.shape[1]))

    print(f"Datos cargados: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, y_train, X_test, y_test

def pipeline_completo(use_cache=None, entrenar_nuevo=False):
    """Pipeline completo usando configuración centralizada"""
    # Usar configuración si no se especifica
    use_cache = use_cache if use_cache is not None else config.cache.usar_cache_caracteristicas
    
    # 1. Cargar datos
    X_train, y_train, X_test, y_test = cargar_datos_train_test(use_cache=use_cache)
    
    # 2. Entrenar o cargar modelo
    if entrenar_nuevo:
        print("Entrenando NUEVO modelo ANFIS...")
        mf_opt, theta_opt = train_anfis(
            X_train, y_train, 
            swarmsize=config.entrenamiento.tamano_enjambre,
            maxiter=config.entrenamiento.max_iteraciones,
            guardar_modelo=config.entrenamiento.guardar_modelo,
            nombre_modelo=config.entrenamiento.nombre_modelo
        )
    else:
        print("Intentando cargar modelo existente...")
        mf_opt, theta_opt, metadatos = sistema_cache.cargar_modelo()
        if mf_opt is not None:
            print("Modelo cargado exitosamente")
        else:
            print("No hay modelo guardado, entrenando nuevo...")
            mf_opt, theta_opt = train_anfis(
                X_train, y_train,
                swarmsize=config.entrenamiento.tamano_enjambre,
                maxiter=config.entrenamiento.max_iteraciones,
                guardar_modelo=config.entrenamiento.guardar_modelo,
                nombre_modelo=config.entrenamiento.nombre_modelo
            )
    
    modelo = {'mf_params': mf_opt, 'theta': theta_opt}
    
    # 3. Evaluar modelo en TEST
    print("Evaluando modelo en datos de TEST...")
    evaluador = EvaluadorANFIS(modelo, {'X': X_test, 'y': y_test})
    resultados_eval = evaluador.evaluar_modelo(save_plots=config.analisis.guardar_graficos)
    
    # 4. Analizar reglas
    print("Analizando reglas del modelo...")
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_test, y_test)
    analizador.generar_analisis_completo()  # Sin parámetro, usa configuración
    
    # 5. Mostrar resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL DEL MODELO")
    print("="*60)
    metricas = resultados_eval['metricas']
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Sensibilidad: {metricas['sensitivity']:.4f}")
    print(f"Especificidad: {metricas['specificity']:.4f}")
    print(f"F1-Score: {metricas['f1_score']:.4f}")
    if metricas['auc'] > 0:
        print(f"AUC-ROC: {metricas['auc']:.4f}")
    print("\n" + "="*60)
    # Mostrar estadísticas de caché
    stats = sistema_cache.obtener_estadisticas_cache()
    print(f"\nEstadisticas de cache:")
    print("\n" + "="*60)
    for tipo, datos in stats.items():
        print(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")
    print("\n" + "="*60)
    return {
        "modelo": modelo,
        "evaluacion": resultados_eval
    }

def pipeline_completo_con_modelo(nombre_modelo, use_cache=True):
    """Pipeline completo usando un modelo especifico"""
    X_train, y_train, X_test, y_test = cargar_datos_train_test(use_cache=use_cache)
    
    print(f"Cargando modelo especifico: {nombre_modelo}")
    mf_opt, theta_opt, metadatos = sistema_cache.cargar_modelo(nombre_modelo=nombre_modelo)
    
    if mf_opt is None:
        print(f"Error: No se pudo cargar el modelo {nombre_modelo}")
        return None
    
    modelo = {'mf_params': mf_opt, 'theta': theta_opt}
    
    # Evaluar modelo
    print("Evaluando modelo...")
    evaluador = EvaluadorANFIS(modelo, {'X': X_test, 'y': y_test})
    resultados_eval = evaluador.evaluar_modelo(save_plots=config.analisis.guardar_graficos)
    
    # Analizar reglas
    print("Analizando reglas del modelo...")
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_test, y_test)
    analizador.generar_analisis_completo()
    
    return {
        "modelo": modelo,
        "evaluacion": resultados_eval
    }

def usar_modelo_guardado(nombre_modelo=None):
    """Usa un modelo guardado especifico para evaluacion"""
    print(f"Cargando modelo: {nombre_modelo}")
    mf_opt, theta_opt, metadatos = sistema_cache.cargar_modelo(nombre_modelo=nombre_modelo)
    
    if mf_opt is None:
        print(f"No se pudo cargar el modelo: {nombre_modelo}")
        return None
    
    # Cargar datos de test
    _, _, X_test, y_test = cargar_datos_train_test(use_cache=True)
    
    if len(X_test) == 0:
        print("No hay datos de test disponibles")
        return None
    
    # Evaluar modelo cargado
    print("Evaluando modelo cargado...")
    modelo = {'mf_params': mf_opt, 'theta': theta_opt}
    evaluador = EvaluadorANFIS(modelo, {'X': X_test, 'y': y_test})
    resultados = evaluador.evaluar_modelo(save_plots=config.analisis.guardar_graficos)
    
    # Mostrar informacion del modelo
    print(f"Informacion del modelo:")
    print(f"  Entrenado: {metadatos.get('fecha_entrenamiento', 'Desconocido')}")
    print(f"  Muestras entrenamiento: {metadatos.get('info_entrenamiento', {}).get('n_muestras_entrenamiento', 'N/A')}")
    
    return resultados

def mostrar_configuracion_actual():
    """Muestra la configuración actual del sistema"""
    print("\nCONFIGURACION ACTUAL:")
    print("=" * 40)
    print(f"Directorio entrenamiento: {config.directorio_entrenamiento}")
    print(f"Directorio prueba: {config.directorio_prueba}")
    print(f"Tamano enjambre PSO: {config.entrenamiento.tamano_enjambre}")
    print(f"Max iteraciones: {config.entrenamiento.max_iteraciones}")
    print(f"Guardar imagenes: {config.procesamiento.guardar_imagenes_intermedias}")
    print(f"Top reglas mostrar: {config.analisis.top_reglas_mostrar}")

def menu_interactivo():
    print("\n==== MENU ANFIS - SISTEMA CONFIGURABLE ====")
    print("1. Pipeline completo (entrenar nuevo modelo)")
    print("2. Pipeline completo (usar modelo guardado)")
    print("3. Solo evaluacion (modelo guardado)")
    print("4. Listar modelos guardados")
    print("5. Estadisticas de cache")
    print("6. Gestionar cache")
    print("7. Configuracion del sistema")
    print("8. Mostrar configuracion actual")
    print("9. Salir")
    
    opcion = input("Ingresa opcion (1-9): ").strip()
    
    if opcion == "1":
        resultados = pipeline_completo(use_cache=True, entrenar_nuevo=True)
        print("\nNuevo modelo entrenado y guardado.")
    elif opcion == "2":
        resultados = pipeline_completo(use_cache=True, entrenar_nuevo=False)
        print("\nPipeline ejecutado con modelo guardado.")
    elif opcion == "3":
        usar_modelo_guardado()
    elif opcion == "4":
        sistema_cache.listar_modelos()
    elif opcion == "5":
        stats = sistema_cache.obtener_estadisticas_cache()
        print("\nEstadisticas de cache:")
        for tipo, datos in stats.items():
            print(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")
    elif opcion == "6":
        print("\nOpciones de gestion de cache:")
        print("1. Limpiar cache de caracteristicas")
        print("2. Limpiar cache de modelos")
        print("3. Limpiar cache de resultados")
        print("4. Limpiar todo el cache")
        sub_opcion = input("Seleccione (1-4): ").strip()
        if sub_opcion == "1":
            sistema_cache.limpiar_cache_caracteristicas()
        elif sub_opcion == "2":
            sistema_cache.limpiar_cache_modelos()
        elif sub_opcion == "3":
            sistema_cache.limpiar_cache_resultados()
        elif sub_opcion == "4":
            sistema_cache.limpiar_cache_caracteristicas()
            sistema_cache.limpiar_cache_modelos()
            sistema_cache.limpiar_cache_resultados()
    elif opcion == "7":
        from config.menu_configuracion import menu_configuracion
        menu_configuracion()
    elif opcion == "8":
        mostrar_configuracion_actual()
    elif opcion == "9":
        print("Saliendo...")
        return
    else:
        print("Opcion invalida")
    
    menu_interactivo()

def ejecutar_desde_interfaz():
    """Función para ejecutar desde la interfaz gráfica"""
    try:
        from interfaz.ventana_principal import VentanaPrincipal
        import tkinter as tk
        root = tk.Tk()
        app = VentanaPrincipal(root)
        root.mainloop()
        return True
    except Exception as e:
        print(f"Error al iniciar interfaz: {e}")
        return False

if __name__ == "__main__":
    # Cargar configuración al iniciar
    config.cargar_configuracion()
    
    print("ANFIS - Deteccion de Tumores Cerebrales")
    print("1. Interfaz Grafica")
    print("2. Menu Consola")
    
    opcion = input("Seleccione opcion (1-2): ").strip()
    
    
    
    if opcion == "1":
        if not ejecutar_desde_interfaz():
            print("No se pudo cargar la interfaz grafica, cargando menu consola...")
            menu_interactivo()
    else:
        menu_interactivo()