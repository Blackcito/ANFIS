"""
config/rutas_portables.py - SISTEMA CENTRALIZADO DE RUTAS
"""

import sys
import os
from pathlib import Path
import platform

class SistemaRutas:
    def __init__(self):
        self._base_dir = None
        self._persist_dir = None
        self._configurar_rutas()
    
    def _configurar_rutas(self):
        """Configura todas las rutas de manera centralizada"""
        # Determinar directorio base según modo
        if getattr(sys, 'frozen', False):
            self._base_dir = Path(sys.executable).parent
            self._modo = "ejecutable"
        else:
            self._base_dir = Path(__file__).parent.parent
            self._modo = "desarrollo"
        
        # Determinar directorio persistente según SO
        sistema = platform.system().lower()
        if sistema == "windows":
            self._persist_dir = Path.home() / "AppData" / "Roaming" / "ANFIS_Tumor_Cerebral"
        elif sistema == "darwin":
            self._persist_dir = Path.home() / "Library" / "Application Support" / "ANFIS_Tumor_Cerebral"
        else:  # Linux y otros Unix
            self._persist_dir = Path.home() / ".local" / "share" / "anfis-tumor-cerebral"
        
        # Crear directorios persistentes
        self._crear_estructura_persistente()
        
        #print(f" Modo: {self._modo}")
        #print(f" Base: {self._base_dir}")
        #print(f" Persistente: {self._persist_dir}")
    
    def _crear_estructura_persistente(self):
        """Crea toda la estructura de directorios persistentes"""
        directorios = [
            self._persist_dir,
            self._persist_dir / "config",
            self._persist_dir / "cache",
            self._persist_dir / "cache" / "caracteristicas",
            self._persist_dir / "cache" / "modelos", 
            self._persist_dir / "cache" / "resultados",
            self._persist_dir / "cache" / "resultados" / "graficos",
            self._persist_dir / "cache" / "resultados" / "reportes",
            self._persist_dir / "cache" / "resultados" / "metricas",
            self._persist_dir / "cache" / "resultados" / "datos_reglas",
            self._persist_dir / "imagenes_intermedias",
            self._persist_dir / "analisis",
        ]
        
        
        for directorio in directorios:
            try:
                directorio.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                print(f" Error de permisos: {directorio} - {e}")
                # En Linux, podríamos intentar con ~/.local/share si falla
                if "linux" in sys.platform:
                    fallback = Path.home() / ".anfis-tumor-cerebral"
                    fallback.mkdir(parents=True, exist_ok=True)
                    self._persist_dir = fallback
                    break
            except Exception as e:
                print(f" Error creando directorio {directorio}: {e}")
    
    # PROPIEDADES PÚBLICAS (SOLO LECTURA)
    @property
    def base_dir(self):
        """Directorio base de la aplicación (ejecutable o desarrollo)"""
        return self._base_dir
    
    @property
    def persist_dir(self):
        """Directorio persistente para datos del usuario"""
        return self._persist_dir
    
    @property
    def config_dir(self):
        """Directorio de configuración persistente"""
        return self._persist_dir / "config"
    
    @property
    def cache_dir(self):
        """Directorio principal de caché"""
        return self._persist_dir / "cache"
    

    

    

    
    def obtener_ruta_configuracion(self):
        """Ruta del archivo de configuración"""
        return self.config_dir / "configuracion.json"
    
    def obtener_rutas_importantes(self):
        """Diccionario con todas las rutas importantes (para logging)"""
        return {
            'modo': self._modo,
            'base': str(self.base_dir),
            'persistente': str(self.persist_dir),
            'config': str(self.config_dir),
            'cache': str(self.cache_dir)
        }

# Instancia global del sistema de rutas
sistema_rutas = SistemaRutas()