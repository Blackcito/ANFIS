# utils/cache.py - ACTUALIZADO PARA USAR RUTAS PERSISTENTES

import os
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from config.rutas import sistema_rutas
from config.configuracion import config

class SistemaCache:
    def __init__(self):
        # Usar sistema centralizado de rutas
        self.directorios = {
            'caracteristicas': sistema_rutas.cache_dir / "caracteristicas",
            'modelos': sistema_rutas.cache_dir / "modelos", 
            'resultados': sistema_rutas.cache_dir / "resultados",
            'graficos': sistema_rutas.cache_dir / "resultados" / "graficos",
            'reportes': sistema_rutas.cache_dir / "resultados" / "reportes",
            'metricas': sistema_rutas.cache_dir / "resultados" / "metricas",
            'datos_reglas': sistema_rutas.cache_dir / "resultados" / "datos_reglas"
        }
        
        # Los directorios ya están creados por sistema_rutas
        print(f" Sistema de caché inicializado en: {sistema_rutas.cache_dir}")
    
    # ===== CACHÉ DE CARACTERÍSTICAS =====
    def guardar_caracteristicas(self, base_dir, features, labels):
        """Guarda características GLCM en caché SI ESTÁ CONFIGURADO"""
        # SIEMPRE guardar si está configurado, independientemente de si se usó caché para cargar
        if not config.cache.guardar_cache_caracteristicas:
            return None
            
        nombre_archivo = f"caracteristicas_{self._normalizar_nombre(base_dir)}.npz"
        ruta_archivo = self.directorios['caracteristicas'] / nombre_archivo
        
        np.savez(ruta_archivo, features=features, labels=labels)
        print(f" Características guardadas en caché: {nombre_archivo}")
        return ruta_archivo
    
    def cargar_caracteristicas(self, base_dir):
        """Carga características GLCM desde caché - depende de la opción de USO"""
        # El USO del caché se controla desde las opciones de ejecución, no desde configuración
        nombre_archivo = f"caracteristicas_{self._normalizar_nombre(base_dir)}.npz"
        ruta_archivo = self.directorios['caracteristicas'] / nombre_archivo
        
        if os.path.exists(ruta_archivo):
            try:
                data = np.load(ruta_archivo)
                print(f" Características cargadas desde caché: {nombre_archivo}")
                return data['features'], data['labels']
            except Exception as e:
                print(f" Error cargando caché de características: {e}")
        
        return None, None
    
    def listar_caracteristicas(self):
        """Lista todos los cachés de características disponibles - DEVUELVE LISTA"""
        caracteristicas_dir = self.directorios['caracteristicas']
        
        if not caracteristicas_dir.exists():
            return []
        
        archivos = [f.name for f in caracteristicas_dir.glob("*.npz")]
        

        return sorted(archivos)  # Devuelve lista ordenada

    def cargar_caracteristicas_especificas(self, nombre_archivo):
        """Carga características específicas por nombre de archivo"""
        # Verificar si el caché global está activado

            
        ruta_archivo = self.directorios['caracteristicas'] / nombre_archivo
        
        if ruta_archivo.exists():
            try:
                data = np.load(ruta_archivo)
                return data['features'], data['labels']
            except Exception as e:
                print(f"Error cargando cache especifico: {e}")
        
        return None, None

    def eliminar_caracteristicas(self, nombre_archivo):
        """Elimina un cache de características específico"""
        ruta_archivo = self.directorios['caracteristicas'] / nombre_archivo
        
        try:
            if ruta_archivo.exists():
                ruta_archivo.unlink()
                print(f"Cache de características {nombre_archivo} eliminado")
                return True
        except Exception as e:
            print(f"Error eliminando caracteristicas: {e}")
        
        return False

    # ===== CACHÉ DE MODELOS =====
    def guardar_modelo(self, mf_params, theta, metricas=None, info_entrenamiento=None, nombre="modelo_anfis"):
        """Guarda modelo ANFIS entrenado SI ESTÁ CONFIGURADO"""
        if not config.cache.guardar_cache_modelos:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base = f"{nombre}_{timestamp}"
        
        # Guardar parámetros
        ruta_modelo = self.directorios['modelos'] / f"{nombre_base}.npz"
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
        
        ruta_metadatos = self.directorios['modelos'] / f"{nombre_base}_meta.json"
        with open(ruta_metadatos, 'w', encoding='utf-8') as f:
            json.dump(metadatos, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Modelo guardado: {ruta_modelo}")
        return ruta_modelo
    
    def listar_modelos(self):
        """Lista todos los modelos guardados - DEVUELVE LISTA"""
        modelos_dir = self.directorios['modelos']
        
        if not modelos_dir.exists():
            return []
        
        archivos_npz = [f.name for f in modelos_dir.glob("*.npz")]
               
        return sorted(archivos_npz)  # Devuelve lista ordenada
    
    def cargar_modelo(self, ruta_modelo=None, nombre_modelo=None):
        """Carga modelo ANFIS entrenado"""
        # Verificar si el caché de modelos está activado
        #if not config.cache.cache_modelos:
        #    return None, None, None
            
        modelos_dir = self.directorios['modelos']
        
        if not modelos_dir.exists():
            return None, None, None
        
        # Buscar archivos de modelo
        archivos_npz = [f.name for f in modelos_dir.glob("*.npz")]
        
        if not archivos_npz:
            return None, None, None
        
        # Determinar qué modelo cargar
        if ruta_modelo:
            archivo_modelo = Path(ruta_modelo).name
        elif nombre_modelo:
            modelos_filtrados = [f for f in archivos_npz if nombre_modelo in f]
            if not modelos_filtrados:
                return None, None, None
            archivo_modelo = sorted(modelos_filtrados)[-1]
        else:
            archivo_modelo = sorted(archivos_npz)[-1]
        
        # Cargar modelo
        ruta_completa = self.directorios['modelos'] / archivo_modelo
        try:
            data = np.load(ruta_completa)
            mf_params = data['mf_params']
            theta = data['theta']
            
            # Cargar metadatos
            ruta_metadatos = ruta_completa.with_name(ruta_completa.stem + "_meta.json")
            with open(ruta_metadatos, 'r', encoding='utf-8') as f:
                metadatos = json.load(f)
            
            print(f" Modelo cargado: {archivo_modelo}")
            return mf_params, theta, metadatos
        
        except Exception as e:
            print(f" Error cargando modelo: {e}")
            return None, None, None
    
    def eliminar_modelo(self, nombre_modelo):
        """Elimina un modelo específico"""
        ruta_modelo = self.directorios['modelos'] / nombre_modelo
        ruta_metadatos = ruta_modelo.with_name(ruta_modelo.stem + "_meta.json")
        
        try:
            if ruta_modelo.exists():
                ruta_modelo.unlink()
            if ruta_metadatos.exists():
                ruta_metadatos.unlink()
            print(f"Modelo {nombre_modelo} eliminado")
            return True
        except Exception as e:
            print(f"Error eliminando modelo: {e}")
            return False
    
    # ===== CACHÉ DE RESULTADOS =====
    def guardar_grafico(self, nombre_grafico, figura=None, datos_bytes=None):
        """Guarda un gráfico en el caché de resultados SI ESTÁ CONFIGURADO"""
        if not config.cache.guardar_cache_graficos:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_grafico}_{timestamp}.png"
        ruta_archivo = self.directorios['graficos'] / nombre_archivo
        
        try:
            if figura:
                figura.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
            elif datos_bytes:
                with open(ruta_archivo, 'wb') as f:
                    f.write(datos_bytes)
            
            #print(f" Gráfico guardado en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando gráfico: {e}")
            return None
    
    def guardar_reporte(self, nombre_reporte, contenido):
        """Guarda un reporte de texto en el caché de resultados"""
        # Verificar si el caché de resultados está activado
        if not config.cache.guardar_cache_reportes:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_reporte}_{timestamp}.txt"
        ruta_archivo = self.directorios['reportes'] / nombre_archivo
        
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido)
            
            #print(f" Reporte guardado en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando reporte: {e}")
            return None
    
    def guardar_datos_reglas(self, nombre_modelo, datos_csv, datos_txt):
        """Guarda los datos de las reglas en el caché"""
        if not config.cache.guardar_cache_datos_reglas:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base = f"datos_reglas_{nombre_modelo}_{timestamp}"
        
        # Guardar archivos CSV y TXT
        ruta_csv = self.directorios['datos_reglas'] / f"{nombre_base}.csv"
        ruta_txt = self.directorios['datos_reglas'] / f"{nombre_base}.txt"
        
        try:
            with open(ruta_csv, 'w', encoding='utf-8') as f:
                f.write(datos_csv)
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write(datos_txt)
            
            #print(f" Datos de reglas guardados en caché: {nombre_base}")
            return {'csv': ruta_csv, 'txt': ruta_txt}
        except Exception as e:
            print(f" Error guardando datos de reglas: {e}")
            return None
            
    def guardar_metricas(self, nombre_modelo, metricas):
        """Guarda métricas de evaluación en el caché"""
        # Verificar si el caché de resultados está activado
        if not config.cache.guardar_cache_metricas:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"metricas_{nombre_modelo}_{timestamp}.json"
        ruta_archivo = self.directorios['metricas'] / nombre_archivo
        
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                json.dump(metricas, f, indent=2, ensure_ascii=False)
            
            #print(f" Métricas guardadas en caché: {ruta_archivo}")
            return ruta_archivo
        except Exception as e:
            print(f" Error guardando métricas: {e}")
            return None
    
    def cargar_metricas_recientes(self, nombre_modelo=None):
        """Carga las métricas más recientes"""
        # Verificar si el caché de resultados está activado
        #if not config.cache.cache_resultados:
        #    return None
            
        metricas_dir = self.directorios['metricas']
        
        if not metricas_dir.exists():
            return None
        
        archivos_json = [f.name for f in metricas_dir.glob("*.json")]
        
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
        
        ruta_completa = self.directorios['metricas'] / archivo_metricas
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
                if directorio.exists():
                    archivos = [f.name for f in directorio.iterdir()]
                    tamaño_total = sum(
                        f.stat().st_size for f in directorio.iterdir()
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
    def limpiar_cache_caracteristicas(self):
        """Limpia el caché de características"""
        caracteristicas_dir = self.directorios['caracteristicas']
        
        if not caracteristicas_dir.exists():
            print("No hay caché de características")
            return
        
        archivos = [f for f in caracteristicas_dir.iterdir()]
        if not archivos:
            print("No hay archivos en el caché de características")
            return
        
        print(f"Eliminando {len(archivos)} archivos de características...")
        for archivo in archivos:
            archivo.unlink()
        print(" Caché de características limpiado")
    
    def limpiar_cache_modelos(self):
        """Limpia el caché de modelos"""
        modelos_dir = self.directorios['modelos']
        
        if not modelos_dir.exists():
            print("No hay modelos guardados")
            return
        
        archivos_npz = [f for f in modelos_dir.glob("*.npz")]
        archivos_json = [f for f in modelos_dir.glob("*.json")]
        
        print(f"Eliminando {len(archivos_npz)} modelos y {len(archivos_json)} metadatos...")
        
        for archivo in archivos_npz + archivos_json:
            archivo.unlink()
        
        print(" Caché de modelos limpiado")
    
    def limpiar_cache_resultados(self, tipo='todos'):
        """Limpia el caché de resultados"""
        if tipo == 'todos':
            directorios = ['graficos', 'reportes', 'metricas', 'datos_reglas']
        elif tipo in ['graficos', 'reportes', 'metricas', 'datos_reglas']:
            directorios = [tipo]
        else:
            print(f" Tipo no válido: {tipo}")
            return
        
        for dir_tipo in directorios:
            directorio = self.directorios[dir_tipo]
            if directorio.exists():
                archivos = [f for f in directorio.iterdir()]
                #print(f"Eliminando {len(archivos)} archivos de {dir_tipo}...")
                for archivo in archivos:
                    archivo.unlink()
                #print(f" Caché de {dir_tipo} limpiado")

    def obtener_estadisticas_cache(self):
        """Obtiene estadísticas del uso del caché"""
        stats = {}
        
        for nombre, directorio in self.directorios.items():
            if directorio.exists():
                archivos = [f for f in directorio.iterdir()]
                tamaño_total = sum(
                    f.stat().st_size for f in archivos
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
        return str(ruta).replace('/', '_').replace('\\', '_').replace(':', '')

# Instancia global del sistema de caché
sistema_cache = SistemaCache()