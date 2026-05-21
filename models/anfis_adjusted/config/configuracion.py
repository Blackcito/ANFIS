# config/configuracion.py - VERSIÓN SIMPLIFICADA

import json
from dataclasses import dataclass, asdict
from config.rutas import sistema_rutas

@dataclass
class ConfiguracionCache:
    guardar_cache_caracteristicas: bool = True
    guardar_cache_modelos: bool = True
    guardar_cache_graficos: bool = True
    guardar_cache_metricas: bool = True
    guardar_cache_reportes: bool = True
    guardar_cache_datos_reglas: bool = True

@dataclass 
class ConfiguracionProcesamiento:
    guardar_imagenes_intermedias: bool = False
    directorio_imagenes_intermedias: str = str(sistema_rutas.persist_dir / "imagenes_intermedias")
    normalizar_caracteristicas: bool = False

@dataclass
class ConfiguracionEntrenamiento:
    tamano_enjambre: int = 30
    max_iteraciones: int = 10
    guardar_modelo: bool = True
    nombre_modelo: str = "modelo_anfis"

@dataclass
class ConfiguracionAnalisis:
    top_reglas_mostrar: int = 15
    guardar_metricas: bool = True
    guardar_reportes: bool = True
    guardar_datos_reglas: bool = True
    guardar_graficos_analisis: bool = True
    directorio_analisis: str = str(sistema_rutas.persist_dir / "analisis")

class ConfiguracionGlobal:
    def __init__(self):
        # Directorios de datos (configurables por usuario)
        self.directorio_entrenamiento_tumor = ""
        self.directorio_entrenamiento_notumor = ""
        self.directorio_prueba_tumor = ""
        self.directorio_prueba_notumor = ""
        
        # Módulos de configuración
        self.procesamiento = ConfiguracionProcesamiento()
        self.entrenamiento = ConfiguracionEntrenamiento()
        self.cache = ConfiguracionCache()
        self.analisis = ConfiguracionAnalisis()
        
        # Archivo de configuración
        self.archivo_config = sistema_rutas.obtener_ruta_configuracion()
        
        # Cargar configuración existente
        self.cargar_configuracion()
    
    def guardar_configuracion(self):
        """Guarda la configuración en archivo persistente"""
        try:
            config_dict = {
                'directorios': {
                    'entrenamiento_tumor': self.directorio_entrenamiento_tumor,
                    'entrenamiento_notumor': self.directorio_entrenamiento_notumor,
                    'prueba_tumor': self.directorio_prueba_tumor,
                    'prueba_notumor': self.directorio_prueba_notumor,
                },
                'procesamiento': asdict(self.procesamiento),
                'entrenamiento': asdict(self.entrenamiento),
                'cache': asdict(self.cache),
                'analisis': asdict(self.analisis)
            }
            
            with open(self.archivo_config, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f" Configuración guardada en: {self.archivo_config}")
            return True
            
        except Exception as e:
            print(f" Error guardando configuración: {e}")
            return False
    
    def cargar_configuracion(self):
        """Carga la configuración desde archivo persistente"""
        if self.archivo_config.exists():
            try:
                with open(self.archivo_config, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # Cargar directorios
                if 'directorios' in config_dict:
                    dirs = config_dict['directorios']
                    self.directorio_entrenamiento_tumor = dirs.get('entrenamiento_tumor', self.directorio_entrenamiento_tumor)
                    self.directorio_entrenamiento_notumor = dirs.get('entrenamiento_notumor', self.directorio_entrenamiento_notumor)
                    self.directorio_prueba_tumor = dirs.get('prueba_tumor', self.directorio_prueba_tumor)
                    self.directorio_prueba_notumor = dirs.get('prueba_notumor', self.directorio_prueba_notumor)
                
                # Cargar módulos
                for modulo, clase in [
                    ('procesamiento', self.procesamiento),
                    ('entrenamiento', self.entrenamiento), 
                    ('cache', self.cache),
                    ('analisis', self.analisis)
                ]:
                    if modulo in config_dict:
                        for key, value in config_dict[modulo].items():
                            if hasattr(clase, key):
                                setattr(clase, key, value)
                
                print(f" Configuración cargada desde: {self.archivo_config}")
                return True
                
            except Exception as e:
                print(f" Error cargando configuración: {e}")
        
        print("No se encontró archivo de configuración, usando valores por defecto")
        return False

# Instancia global
config = ConfiguracionGlobal()