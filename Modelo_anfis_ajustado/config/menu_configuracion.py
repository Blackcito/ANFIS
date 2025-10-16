# config/menu_configuracion.py

import os
from config.configuracion import config

def menu_configuracion():
    """Menú interactivo para configurar el sistema desde consola"""
    while True:
        print("\n MENÚ DE CONFIGURACIÓN ANFIS")
        print("=" * 40)
        print("1. Mostrar configuración actual")
        print("2. Configurar directorios de datos")
        print("3. Configurar procesamiento de imágenes")
        print("4. Configurar entrenamiento")
        print("5. Configurar sistema de caché")
        print("6. Configurar análisis")
        print("7. Guardar configuración")
        print("8. Cargar configuración")
        print("9. Volver al menú principal")
        
        opcion = input("Seleccione opción (1-9): ").strip()
        
        if opcion == "1":
            config.cargar_configuracion()
        
        elif opcion == "2":
            print(f"\n CONFIGURAR DIRECTORIOS")
            print(f"Directorio actual entrenamiento: {config.directorio_entrenamiento}")
            nuevo_train = input("Nuevo directorio entrenamiento (Enter para mantener): ").strip()
            if nuevo_train:
                config.directorio_entrenamiento = nuevo_train
            
            print(f"Directorio actual prueba: {config.directorio_prueba}")
            nuevo_test = input("Nuevo directorio prueba (Enter para mantener): ").strip()
            if nuevo_test:
                config.directorio_prueba = nuevo_test
        
        elif opcion == "3":
            print(f"\n CONFIGURAR PROCESAMIENTO")
            config.procesamiento.guardar_imagenes_intermedias = _input_bool(
                "Guardar imágenes intermedias", 
                config.procesamiento.guardar_imagenes_intermedias
            )
            config.procesamiento.usar_cache_imagenes = _input_bool(
                "Usar caché de imágenes", 
                config.procesamiento.usar_cache_imagenes
            )
            config.procesamiento.normalizar_caracteristicas = _input_bool(
                "Normalizar características", 
                config.procesamiento.normalizar_caracteristicas
            )
        
        elif opcion == "4":
            print(f"\n CONFIGURAR ENTRENAMIENTO")
            config.entrenamiento.tamano_enjambre = _input_int(
                "Tamaño del enjambre PSO", 
                config.entrenamiento.tamano_enjambre
            )
            config.entrenamiento.max_iteraciones = _input_int(
                "Máximo de iteraciones", 
                config.entrenamiento.max_iteraciones
            )
            config.entrenamiento.guardar_modelo = _input_bool(
                "Guardar modelo automáticamente", 
                config.entrenamiento.guardar_modelo
            )
        
        elif opcion == "5":
            print(f"\n CONFIGURAR CACHÉ")
            config.cache.usar_cache_caracteristicas = _input_bool(
                "Usar caché de características", 
                config.cache.usar_cache_caracteristicas
            )
            config.cache.usar_cache_modelos = _input_bool(
                "Usar caché de modelos", 
                config.cache.usar_cache_modelos
            )
            config.cache.usar_cache_resultados = _input_bool(
                "Usar caché de resultados", 
                config.cache.usar_cache_resultados
            )
        
        elif opcion == "6":
            print(f"\n CONFIGURAR ANÁLISIS")
            config.analisis.top_reglas_mostrar = _input_int(
                "Número de reglas a mostrar", 
                config.analisis.top_reglas_mostrar
            )
            config.analisis.guardar_graficos = _input_bool(
                "Guardar gráficos automáticamente", 
                config.analisis.guardar_graficos
            )
            config.analisis.guardar_reportes = _input_bool(
                "Guardar reportes automáticamente", 
                config.analisis.guardar_reportes
            )
        
        elif opcion == "7":
            config.guardar_configuracion()
        
        elif opcion == "8":
            config.cargar_configuracion()
        
        elif opcion == "9":
            break
        
        else:
            print(" Opción inválida")

def _input_bool(prompt, valor_actual):
    """Solicita entrada booleana al usuario"""
    respuesta = input(f"{prompt} [{'Sí' if valor_actual else 'No'}] (s/n): ").strip().lower()
    if respuesta == 's':
        return True
    elif respuesta == 'n':
        return False
    else:
        return valor_actual

def _input_int(prompt, valor_actual):
    """Solicita entrada entera al usuario"""
    respuesta = input(f"{prompt} [{valor_actual}]: ").strip()
    if respuesta.isdigit():
        return int(respuesta)
    else:
        return valor_actual