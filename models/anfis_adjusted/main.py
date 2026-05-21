# main.py - ACTUALIZADO Y SIMPLIFICADO
# Obsoleto, se mantiene para compatibilidad con versiones anteriores y referencia.

from core.pipeline_anfis import pipeline_anfis
from config.configuracion import config
from utils.cache import sistema_cache

def menu_interactivo():
    """Menú interactivo actualizado"""
    while True:
        print("\n==== MENU ANFIS - SISTEMA OPTIMIZADO ====")
        print("1. Pipeline completo (entrenar nuevo)")
        print("2. Solo evaluación (modelo existente)") 
        print("3. Evaluar modelo específico")
        print("4. Configurar sistema")
        print("5. Gestión de caché")
        print("6. Salir")
        
        opcion = input("Ingresa opción (1-6): ").strip()
        
        if opcion == "1":
            # Entrenamiento completo con opciones personalizadas
            resultados = pipeline_anfis.ejecutar(
                entrenar_nuevo=True,
                forzar_reprocesamiento=False  # Usar caché si está disponible
            )
            
        elif opcion == "2":
            # Solo evaluación, muy rápido
            resultados = pipeline_anfis.ejecutar(
                entrenar_nuevo=False,
                forzar_reprocesamiento=False
            )
            
        elif opcion == "3":
            nombre = input("Nombre del modelo: ").strip()
            resultados = pipeline_anfis.ejecutar(
                entrenar_nuevo=False,
                nombre_modelo=nombre,
                forzar_reprocesamiento=False
            )
            
        elif opcion == "4":
            from config.menu_configuracion import menu_configuracion
            menu_configuracion()
            
        elif opcion == "5":
            gestionar_cache()
            
        elif opcion == "6":
            break
            
        else:
            print("Opción inválida")

def gestionar_cache():
    """Gestión del caché"""
    print("\n💾 GESTIÓN DE CACHÉ")
    print("1. Limpiar caché de características")
    print("2. Limpiar caché de modelos") 
    print("3. Limpiar caché de resultados")
    print("4. Limpiar todo")
    print("5. Estadísticas")
    
    opcion = input("Seleccione (1-5): ").strip()
    
    if opcion == "1":
        sistema_cache.limpiar_cache_caracteristicas()
    elif opcion == "2":
        sistema_cache.limpiar_cache_modelos()
    elif opcion == "3":
        sistema_cache.limpiar_cache_resultados()
    elif opcion == "4":
        sistema_cache.limpiar_cache_caracteristicas()
        sistema_cache.limpiar_cache_modelos()
        sistema_cache.limpiar_cache_resultados()
    elif opcion == "5":
        stats = sistema_cache.obtener_estadisticas_cache()
        for tipo, datos in stats.items():
            print(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")

if __name__ == "__main__":
    config.cargar_configuracion()
    print("🧠 ANFIS - Sistema Optimizado")
    menu_interactivo()