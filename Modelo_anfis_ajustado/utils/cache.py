import os
import numpy as np

def guardar_cache(cache_file, features, labels):
    """Guarda características y etiquetas en caché"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(cache_file, features=features, labels=labels)

def cargar_cache(cache_file):
    """Carga características y etiquetas desde caché"""
    try:
        data = np.load(cache_file)
        return data['features'], data['labels']
    except:
        return None, None

def gestionar_cache_global():
    """Gestiona todos los archivos de caché"""
    cache_dir = "./features_cache"
    if not os.path.exists(cache_dir):
        print(" No hay archivos de caché")
        return
    
    archivos = os.listdir(cache_dir)
    print(f"\n Archivos en caché ({len(archivos)}):")
    for archivo in archivos:
        ruta = os.path.join(cache_dir, archivo)
        tamaño = os.path.getsize(ruta) / 1024 / 1024  # MB
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