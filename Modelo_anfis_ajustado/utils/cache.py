# utils/cache.py

import os
import numpy as np
import json
from datetime import datetime
from config.configuracion import config

# Obtener el directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SistemaCache:
    def __init__(self):
        # Todas las rutas relativas al proyecto
        self.directorios = {
            'caracteristicas': os.path.join(BASE_DIR, "utils", "cache", "caracteristicas"),
            'modelos': os.path.join(BASE_DIR, "utils", "cache", "modelos"),
            'resultados': os.path.join(BASE_DIR, "utils", "cache", "resultados"),
            'graficos': os.path.join(BASE_DIR, "utils", "cache", "resultados", "graficos"),
            'reportes': os.path.join(BASE_DIR, "utils", "cache", "resultados", "reportes"),
            'metricas': os.path.join(BASE_DIR, "utils", "cache", "resultados", "metricas")
        }
        
        # Crear directorios si no existen
        for directorio in self.directorios.values():
            os.makedirs(directorio, exist_ok=True)
    
    # ===== CACHÉ DE CARACTERÍSTICAS =====
    def guardar_caracteristicas(self, base_dir, features, labels):
        """Guarda características GLCM en caché"""
        nombre_archivo = f"caracteristicas_{self._normalizar_nombre(base_dir)}.npz"
        ruta_archivo = os.path.join(self.directorios['caracteristicas'], nombre_archivo)
        
        np.savez(ruta_archivo, features=features, labels=labels)
        return ruta_archivo
    
    def cargar_caracteristicas(self, base_dir):
        """Carga características GLCM desde caché"""
        nombre_archivo = f"caracteristicas_{self._normalizar_nombre(base_dir)}.npz"
        ruta_archivo = os.path.join(self.directorios['caracteristicas'], nombre_archivo)
        
        if os.path.exists(ruta_archivo):
            try:
                data = np.load(ruta_archivo)
                return data['features'], data['labels']
            except Exception as e:
                print(f"❌ Error cargando caché de características: {e}")
        
        return None, None
    
    def listar_caracteristicas(self):
        """Lista todos los cachés de características disponibles"""
        caracteristicas_dir = self.directorios['caracteristicas']
        
        if not os.path.exists(caracteristicas_dir):
            return []
        
        archivos = [f for f in os.listdir(caracteristicas_dir) if f.endswith('.npz')]
        
        print(f"\nCaches de caracteristicas disponibles ({len(archivos)}):")
        for archivo in sorted(archivos):
            ruta_completa = os.path.join(caracteristicas_dir, archivo)
            tamaño = os.path.getsize(ruta_completa) / 1024  # KB
            
            try:
                data = np.load(ruta_completa)
                muestras = len(data['features'])
                print(f"  - {archivo} ({muestras} muestras, {tamaño:.1f} KB)")
            except:
                print(f"  - {archivo} (corrupto, {tamaño:.1f} KB)")
        
        return archivos

    def cargar_caracteristicas_especificas(self, nombre_archivo):
        """Carga características específicas por nombre de archivo"""
        ruta_archivo = os.path.join(self.directorios['caracteristicas'], nombre_archivo)
        
        if os.path.exists(ruta_archivo):
            try:
                data = np.load(ruta_archivo)
                return data['features'], data['labels']
            except Exception as e:
                print(f"Error cargando cache especifico: {e}")
        
        return None, None

    def eliminar_caracteristicas(self, nombre_archivo):
        """Elimina un cache de características específico"""
        ruta_archivo = os.path.join(self.directorios['caracteristicas'], nombre_archivo)
        
        try:
            if os.path.exists(ruta_archivo):
                os.remove(ruta_archivo)
                print(f"Cache de características {nombre_archivo} eliminado")
                return True
        except Exception as e:
            print(f"Error eliminando caracteristicas: {e}")
        
        return False

    # ===== CACHÉ DE MODELOS =====
    def guardar_modelo(self, mf_params, theta, metricas=None, info_entrenamiento=None, nombre="modelo_anfis"):
        """Guarda modelo ANFIS entrenado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base = f"{nombre}_{timestamp}"
        
        # Guardar parámetros
        ruta_modelo = os.path.join(self.directorios['modelos'], f"{nombre_base}.npz")
        np.savez(ruta_modelo, mf_params=mf_params, theta=theta)
        
        # Guardar metadatos
        metadatos = {
            'fecha_entrenamiento': timestamp,
            'nombre_modelo': nombre,
            'forma_mf_params': mf_params.shape,
            'forma_theta': theta.shape,
            'metricas': metricas or {},
            'info_entrenamiento': info_entrenamiento or {}
        }
        
        ruta_metadatos = os.path.join(self.directorios['modelos'], f"{nombre_base}_meta.json")
        with open(ruta_metadatos, 'w', encoding='utf-8') as f:
            json.dump(metadatos, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Modelo guardado: {ruta_modelo}")
        return ruta_modelo
    
    def cargar_modelo(self, ruta_modelo=None, nombre_modelo=None):
        """Carga modelo ANFIS entrenado"""
        modelos_dir = self.directorios['modelos']
        
        if not os.path.exists(modelos_dir):
            return None, None, None
        
        # Buscar archivos de modelo
        archivos_npz = [f for f in os.listdir(modelos_dir) if f.endswith('.npz')]
        
        if not archivos_npz:
            return None, None, None
        
        # Determinar qué modelo cargar
        if ruta_modelo:
            archivo_modelo = os.path.basename(ruta_modelo)
        elif nombre_modelo:
            modelos_filtrados = [f for f in archivos_npz if nombre_modelo in f]
            if not modelos_filtrados:
                return None, None, None
            archivo_modelo = sorted(modelos_filtrados)[-1]
        else:
            archivo_modelo = sorted(archivos_npz)[-1]
        
        # Cargar modelo
        ruta_completa = os.path.join(modelos_dir, archivo_modelo)
        try:
            data = np.load(ruta_completa)
            mf_params = data['mf_params']
            theta = data['theta']
            
            # Cargar metadatos
            ruta_metadatos = ruta_completa.replace('.npz', '_meta.json')
            with open(ruta_metadatos, 'r', encoding='utf-8') as f:
                metadatos = json.load(f)
            
            print(f" Modelo cargado: {archivo_modelo}")
            return mf_params, theta, metadatos
        
        except Exception as e:
            print(f" Error cargando modelo: {e}")
            return None, None, None
    
    def eliminar_modelo(self, nombre_modelo):
        """Elimina un modelo específico"""
        ruta_modelo = os.path.join(self.directorios['modelos'], nombre_modelo)
        ruta_metadatos = ruta_modelo.replace('.npz', '_meta.json')
        
        try:
            if os.path.exists(ruta_modelo):
                os.remove(ruta_modelo)
            if os.path.exists(ruta_metadatos):
                os.remove(ruta_metadatos)
            print(f"Modelo {nombre_modelo} eliminado")
            return True
        except Exception as e:
            print(f"Error eliminando modelo: {e}")
            return False
    # ===== CACHÉ DE RESULTADOS =====
    def guardar_grafico(self, nombre_grafico, figura=None, datos_bytes=None):
        """Guarda un gráfico en el caché de resultados"""
        if not config.cache.usar_cache_resultados:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_grafico}_{timestamp}.png"
        ruta_archivo = os.path.join(self.directorios['graficos'], nombre_archivo)
        
        try:
            if figura:
                figura.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
            elif datos_bytes:
                with open(ruta_archivo, 'wb') as f:
                    f.write(datos_bytes)
            
            print(f" Gráfico guardado en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando gráfico: {e}")
            return None
    
    def guardar_reporte(self, nombre_reporte, contenido):
        """Guarda un reporte de texto en el caché de resultados"""
        if not config.cache.usar_cache_resultados:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_reporte}_{timestamp}.txt"
        ruta_archivo = os.path.join(self.directorios['reportes'], nombre_archivo)
        
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido)
            
            print(f" Reporte guardado en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando reporte: {e}")
            return None
    
    def guardar_metricas(self, nombre_modelo, metricas):
        """Guarda métricas de evaluación en el caché"""
        if not config.cache.usar_cache_resultados:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"metricas_{nombre_modelo}_{timestamp}.json"
        ruta_archivo = os.path.join(self.directorios['metricas'], nombre_archivo)
        
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                json.dump(metricas, f, indent=2, ensure_ascii=False)
            
            print(f" Métricas guardadas en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando métricas: {e}")
            return None
    
    def cargar_metricas_recientes(self, nombre_modelo=None):
        """Carga las métricas más recientes"""
        metricas_dir = self.directorios['metricas']
        
        if not os.path.exists(metricas_dir):
            return None
        
        archivos_json = [f for f in os.listdir(metricas_dir) if f.endswith('.json')]
        
        if not archivos_json:
            return None
        
        # Filtrar por nombre de modelo si se especifica
        if nombre_modelo:
            archivos_filtrados = [f for f in archivos_json if nombre_modelo in f]
            if not archivos_filtrados:
                return None
            archivo_metricas = sorted(archivos_filtrados)[-1]
        else:
            archivo_metricas = sorted(archivos_json)[-1]
        
        ruta_completa = os.path.join(metricas_dir, archivo_metricas)
        try:
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                metricas = json.load(f)
            
            print(f" Métricas cargadas desde caché: {archivo_metricas}")
            return metricas
        except Exception as e:
            print(f" Error cargando métricas: {e}")
            return None
    
    def listar_resultados(self, tipo='todos'):
        """Lista resultados guardados en el caché"""
        tipos = {
            'graficos': self.directorios['graficos'],
            'reportes': self.directorios['reportes'],
            'metricas': self.directorios['metricas']
        }
        
        if tipo == 'todos':
            directorios = tipos.values()
        elif tipo in tipos:
            directorios = [tipos[tipo]]
        else:
            print(f"❌ Tipo no válido: {tipo}")
            return {}
        
        resultados = {}
        for nombre_tipo, directorio in tipos.items():
            if tipo == 'todos' or tipo == nombre_tipo:
                if os.path.exists(directorio):
                    archivos = os.listdir(directorio)
                    tamaño_total = sum(
                        os.path.getsize(os.path.join(directorio, f)) 
                        for f in archivos
                    ) / 1024 / 1024  # MB
                    
                    resultados[nombre_tipo] = {
                        'archivos': len(archivos),
                        'tamaño_mb': round(tamaño_total, 2),
                        'ejemplos': archivos[:3]  # Mostrar primeros 3 como ejemplo
                    }
                else:
                    resultados[nombre_tipo] = {'archivos': 0, 'tamaño_mb': 0, 'ejemplos': []}
        
        return resultados




    # ===== GESTIÓN DE CACHÉ =====
    def listar_modelos(self):
        """Lista todos los modelos guardados"""
        modelos_dir = self.directorios['modelos']
        
        if not os.path.exists(modelos_dir):
            print("No hay modelos guardados")
            return []
        
        archivos_npz = [f for f in os.listdir(modelos_dir) if f.endswith('.npz')]
        
        print(f"\n📂 Modelos entrenados guardados ({len(archivos_npz)}):")
        for archivo in sorted(archivos_npz):
            ruta_completa = os.path.join(modelos_dir, archivo)
            tamaño = os.path.getsize(ruta_completa) / 1024  # KB
            
            # Cargar metadatos
            ruta_metadatos = ruta_completa.replace('.npz', '_meta.json')
            try:
                with open(ruta_metadatos, 'r', encoding='utf-8') as f:
                    metadatos = json.load(f)
                fecha = metadatos.get('fecha_entrenamiento', 'Desconocida')
                metricas = metadatos.get('metricas', {})
                auc = metricas.get('auc', 'N/A')
                print(f"  - {archivo}")
                print(f"    Fecha: {fecha} | Tamaño: {tamaño:.1f} KB | AUC: {auc}")
            except:
                print(f"  - {archivo} (sin metadatos, {tamaño:.1f} KB)")
        
        return archivos_npz
    
    def limpiar_cache_caracteristicas(self):
        """Limpia el caché de características"""
        caracteristicas_dir = self.directorios['caracteristicas']
        
        if not os.path.exists(caracteristicas_dir):
            print("No hay caché de características")
            return
        
        archivos = os.listdir(caracteristicas_dir)
        if not archivos:
            print("No hay archivos en el caché de características")
            return
        
        print(f"Eliminando {len(archivos)} archivos de características...")
        for archivo in archivos:
            os.remove(os.path.join(caracteristicas_dir, archivo))
        print(" Caché de características limpiado")
    
    def limpiar_cache_modelos(self):
        """Limpia el caché de modelos"""
        modelos_dir = self.directorios['modelos']
        
        if not os.path.exists(modelos_dir):
            print("No hay modelos guardados")
            return
        
        archivos_npz = [f for f in os.listdir(modelos_dir) if f.endswith('.npz')]
        archivos_json = [f for f in os.listdir(modelos_dir) if f.endswith('.json')]
        
        print(f"Eliminando {len(archivos_npz)} modelos y {len(archivos_json)} metadatos...")
        
        for archivo in archivos_npz + archivos_json:
            os.remove(os.path.join(modelos_dir, archivo))
        
        print(" Caché de modelos limpiado")
    
    def limpiar_cache_resultados(self, tipo='todos'):
        """Limpia el caché de resultados"""
        if tipo == 'todos':
            directorios = ['graficos', 'reportes', 'metricas']
        elif tipo in ['graficos', 'reportes', 'metricas']:
            directorios = [tipo]
        else:
            print(f" Tipo no válido: {tipo}")
            return
        
        for dir_tipo in directorios:
            directorio = self.directorios[dir_tipo]
            if os.path.exists(directorio):
                archivos = os.listdir(directorio)
                print(f"Eliminando {len(archivos)} archivos de {dir_tipo}...")
                for archivo in archivos:
                    os.remove(os.path.join(directorio, archivo))
                print(f" Caché de {dir_tipo} limpiado")

    def obtener_estadisticas_cache(self):
        """Obtiene estadísticas del uso del caché"""
        stats = {}
        
        for nombre, directorio in self.directorios.items():
            if os.path.exists(directorio):
                archivos = os.listdir(directorio)
                tamaño_total = sum(
                    os.path.getsize(os.path.join(directorio, f)) 
                    for f in archivos
                ) / 1024 / 1024  # MB
                
                stats[nombre] = {
                    'archivos': len(archivos),
                    'tamaño_mb': round(tamaño_total, 2)
                }
            else:
                stats[nombre] = {'archivos': 0, 'tamaño_mb': 0}
        
        return stats
    
    def _normalizar_nombre(self, ruta):
        """Normaliza nombres de ruta para usar en nombres de archivo"""
        return ruta.replace('/', '_').replace('\\', '_').replace(':', '')

# Instancia global del sistema de caché
sistema_cache = SistemaCache()