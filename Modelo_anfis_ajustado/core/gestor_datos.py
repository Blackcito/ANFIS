# core/gestor_datos.py - SEPARADO CLARAMENTE

import numpy as np
from sklearn.preprocessing import StandardScaler
from features.procesamiento_image import process_all_images
from config.configuracion import config
from utils.cache import sistema_cache

class GestorDatos:
    """Gestor centralizado para carga y procesamiento de datos"""
    
    def __init__(self):
        self.scaler = None
    
# core/gestor_datos.py - ACTUALIZADO PARA USAR SELECCIÓN ESPECÍFICA

    def cargar_datos_entrenamiento(self, tumor_dir=None, notumor_dir=None, usar_cache=False, cache_especifico=None, forzar_reprocesamiento=False):
        """
        Carga datos de entrenamiento
        
        Args:
            tumor_dir: Directorio con imágenes de tumor para entrenamiento
            notumor_dir: Directorio con imágenes de no-tumor para entrenamiento
            usar_cache: Si debe INTENTAR CARGAR desde caché existente
            cache_especifico: Nombre específico del archivo de cache a cargar
            forzar_reprocesamiento: Si debe IGNORAR caché y reprocesar
        """
        # Si se fuerza reprocesamiento, ignorar cache
        if forzar_reprocesamiento:
            usar_cache = False
        
        # INTENTAR CARGAR desde caché si está permitido
        if usar_cache:
            if cache_especifico:
                # Cargar cache específico
                features, labels = sistema_cache.cargar_caracteristicas_especificas(cache_especifico)
                if features is not None:
                    print(f" Datos de entrenamiento cargados desde caché específico: {cache_especifico}")
                    return self._procesar_caracteristicas(features, labels, entrenamiento=True)
            else:
                # Cargar cache automático por directorio
                cache_dir = f"{tumor_dir}_{notumor_dir}" if tumor_dir and notumor_dir else None
                if cache_dir:
                    features, labels = sistema_cache.cargar_caracteristicas(cache_dir)
                    if features is not None:
                        print(" Datos de entrenamiento cargados desde caché automático")
                        return self._procesar_caracteristicas(features, labels, entrenamiento=True)
        
        # Procesar desde disco
        print(" Procesando datos de entrenamiento desde disco...")
        features, labels = process_all_images(
            tumor_dir=tumor_dir,
            notumor_dir=notumor_dir,
            save_dir=config.procesamiento.directorio_imagenes_intermedias,
            save_images=config.procesamiento.guardar_imagenes_intermedias
        )
        
        # GUARDAR en caché SIEMPRE si está configurado
        if tumor_dir and notumor_dir:
            cache_dir = f"{tumor_dir}_{notumor_dir}"
            sistema_cache.guardar_caracteristicas(cache_dir, features, labels)
        
        return self._procesar_caracteristicas(features, labels, entrenamiento=True)
    
    def cargar_datos_prueba(self, tumor_dir=None, notumor_dir=None, usar_cache=False, cache_especifico=None, forzar_reprocesamiento=False):
        """
        Carga datos de prueba
        
        Args:
            tumor_dir: Directorio con imágenes de tumor para prueba
            notumor_dir: Directorio con imágenes de no-tumor para prueba
            usar_cache: Si debe INTENTAR CARGAR desde caché existente  
            cache_especifico: Nombre específico del archivo de cache a cargar
            forzar_reprocesamiento: Si debe IGNORAR caché y reprocesar
        """
        # Si se fuerza reprocesamiento, ignorar cache
        if forzar_reprocesamiento:
            usar_cache = False
        
        # INTENTAR CARGAR desde caché si está permitido
        if usar_cache:
            if cache_especifico:
                # Cargar cache específico
                features, labels = sistema_cache.cargar_caracteristicas_especificas(cache_especifico)
                if features is not None:
                    print(f" Datos de prueba cargados desde caché específico: {cache_especifico}")
                    return self._procesar_caracteristicas(features, labels, entrenamiento=False)
            else:
                # Cargar cache automático por directorio
                cache_dir = f"{tumor_dir}_{notumor_dir}" if tumor_dir and notumor_dir else None
                if cache_dir:
                    features, labels = sistema_cache.cargar_caracteristicas(cache_dir)
                    if features is not None:
                        print(" Datos de prueba cargados desde caché automático")
                        return self._procesar_caracteristicas(features, labels, entrenamiento=False)
        
        # Procesar desde disco
        print(" Procesando datos de prueba desde disco...")
        features, labels = process_all_images(
            tumor_dir=tumor_dir,
            notumor_dir=notumor_dir,
            save_dir=config.procesamiento.directorio_imagenes_intermedias,
            save_images=config.procesamiento.guardar_imagenes_intermedias
        )
        
        # GUARDAR en caché SIEMPRE si está configurado
        if tumor_dir and notumor_dir:
            cache_dir = f"{tumor_dir}_{notumor_dir}"
            sistema_cache.guardar_caracteristicas(cache_dir, features, labels)
        
        return self._procesar_caracteristicas(features, labels, entrenamiento=False)
    
    def _procesar_caracteristicas(self, features, labels, entrenamiento=True):
        """Procesa características (normalización)"""
        if features.size == 0:
            raise RuntimeError("No hay características para procesar")
        
        labels = np.array(labels, dtype=int)
        
        # Normalización según configuración global
        if config.procesamiento.normalizar_caracteristicas:
            if entrenamiento:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(features)
            elif self.scaler is not None:
                features = self.scaler.transform(features)
        
        return features, labels

# Instancia global
gestor_datos = GestorDatos()