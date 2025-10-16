# config/configuracion.py - AJUSTADO

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

# Obtener el directorio base del proyecto (Modelo_anfis_ajustado)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ConfiguracionProcesamiento:
    """Configuración para el procesamiento de imágenes"""
    guardar_imagenes_intermedias: bool = False
    directorio_imagenes_intermedias: str = os.path.join(BASE_DIR, "debug_images")
    usar_cache_imagenes: bool = True
    normalizar_caracteristicas: bool = True

@dataclass
class ConfiguracionEntrenamiento:
    """Configuración para el entrenamiento del modelo"""
    tamano_enjambre: int = 30
    max_iteraciones: int = 10
    guardar_modelo: bool = True
    nombre_modelo: str = "modelo_anfis"

@dataclass
class ConfiguracionCache:
    """Configuración para el sistema de caché"""
    usar_cache_caracteristicas: bool = True
    usar_cache_modelos: bool = True
    usar_cache_resultados: bool = True
    limpiar_cache_automatico: bool = False

@dataclass
class ConfiguracionAnalisis:
    """Configuración para el análisis de resultados"""
    top_reglas_mostrar: int = 15
    guardar_graficos: bool = True
    guardar_reportes: bool = True
    directorio_analisis: str = os.path.join(BASE_DIR, "analisis_reglas_anfis")

class ConfiguracionGlobal:
    """Configuración global del sistema ANFIS"""
    
    def __init__(self):
        # Rutas relativas al directorio del proyecto
        self.directorio_entrenamiento = os.path.join(BASE_DIR, "../archive/binaria/test_1")
        self.directorio_prueba = os.path.join(BASE_DIR, "../archive/binaria/test_1")
        
        # Módulos de configuración
        self.procesamiento = ConfiguracionProcesamiento()
        self.entrenamiento = ConfiguracionEntrenamiento()
        self.cache = ConfiguracionCache()
        self.analisis = ConfiguracionAnalisis()
        
        # Archivo de configuración dentro del proyecto
        self.archivo_config = os.path.join(BASE_DIR, "config", "configuracion.json")
    
    def guardar_configuracion(self):
        """Guarda la configuración actual en un archivo JSON"""
        os.makedirs(os.path.dirname(self.archivo_config), exist_ok=True)
        
        config_dict = {
            'directorios': {
                'entrenamiento': self.directorio_entrenamiento,
                'prueba': self.directorio_prueba
            },
            'procesamiento': asdict(self.procesamiento),
            'entrenamiento': asdict(self.entrenamiento),
            'cache': asdict(self.cache),
            'analisis': asdict(self.analisis)
        }
        
        with open(self.archivo_config, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuración guardada en: {self.archivo_config}")
    
    def cargar_configuracion(self):
        """Carga la configuración desde un archivo JSON"""
        if os.path.exists(self.archivo_config):
            try:
                with open(self.archivo_config, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # Cargar directorios
                if 'directorios' in config_dict:
                    self.directorio_entrenamiento = config_dict['directorios'].get('entrenamiento', self.directorio_entrenamiento)
                    self.directorio_prueba = config_dict['directorios'].get('prueba', self.directorio_prueba)
                
                # Cargar módulos
                if 'procesamiento' in config_dict:
                    for key, value in config_dict['procesamiento'].items():
                        if hasattr(self.procesamiento, key):
                            setattr(self.procesamiento, key, value)
                
                if 'entrenamiento' in config_dict:
                    for key, value in config_dict['entrenamiento'].items():
                        if hasattr(self.entrenamiento, key):
                            setattr(self.entrenamiento, key, value)
                
                if 'cache' in config_dict:
                    for key, value in config_dict['cache'].items():
                        if hasattr(self.cache, key):
                            setattr(self.cache, key, value)
                
                if 'analisis' in config_dict:
                    for key, value in config_dict['analisis'].items():
                        if hasattr(self.analisis, key):
                            setattr(self.analisis, key, value)
                
                print(f"✅ Configuración cargada desde: {self.archivo_config}")
                return True
                
            except Exception as e:
                print(f"❌ Error cargando configuración: {e}")
        
        print("ℹ️ No se encontró archivo de configuración, usando valores por defecto")
        return False

# Instancia global de configuración
config = ConfiguracionGlobal()