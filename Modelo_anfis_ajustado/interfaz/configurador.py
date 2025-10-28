# interfaz/configurador.py - COMPLETAMENTE CORREGIDO

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.configuracion import config

class Configurador:
    def __init__(self, root):
        self.root = root
        self.root.title("Configurador del Sistema ANFIS")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.config_modificada = False
        self.inicializar_variables()
        self.crear_interfaz()
        self.cargar_configuracion_actual()
    
    def inicializar_variables(self):
        """Inicializa SOLO las variables de configuración actuales"""
        # Directorios
        self.dir_entrenamiento_tumor = tk.StringVar()
        self.dir_entrenamiento_notumor = tk.StringVar()
        self.dir_prueba_tumor = tk.StringVar()
        self.dir_prueba_notumor = tk.StringVar()
        
        # Procesamiento
        self.guardar_imagenes = tk.BooleanVar()
        self.dir_imagenes = tk.StringVar()
        self.normalizar_caracteristicas = tk.BooleanVar()
        
        # Entrenamiento
        self.tamano_enjambre = tk.IntVar()
        self.max_iteraciones = tk.IntVar()
        self.nombre_modelo = tk.StringVar()
        
        # Cache - SOLO configuración de GUARDADO (nueva estructura)
        self.guardar_cache_caracteristicas = tk.BooleanVar()
        self.guardar_cache_modelos = tk.BooleanVar()
        self.guardar_cache_graficos = tk.BooleanVar()
        self.guardar_cache_metricas = tk.BooleanVar()
        self.guardar_cache_reportes = tk.BooleanVar()
        self.guardar_cache_datos_reglas = tk.BooleanVar()
        
        # Analisis
        self.top_reglas = tk.IntVar()
        self.guardar_metricas = tk.BooleanVar()
        self.guardar_reportes = tk.BooleanVar()
        self.guardar_datos_reglas = tk.BooleanVar()
        self.guardar_graficos_analisis = tk.BooleanVar()
        self.dir_analisis = tk.StringVar()
    
    def crear_interfaz(self):
        # Frame principal con paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Navegacion
        left_frame = ttk.Frame(main_paned, width=200)
        main_paned.add(left_frame, weight=0)
        
        # Panel derecho - Contenido
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # ===== PANEL IZQUIERDO - NAVEGACION =====
        ttk.Label(left_frame, text="Categorías", font=('Arial', 11, 'bold')).pack(pady=(0, 10))
        
        self.categorias = [
            ("Directorios", self.mostrar_directorios),
            ("Procesamiento", self.mostrar_procesamiento),
            ("Entrenamiento", self.mostrar_entrenamiento),
            ("Sistema de Cache", self.mostrar_cache),
            ("Análisis", self.mostrar_analisis)
        ]
        
        self.botones_categorias = []
        for i, (nombre, comando) in enumerate(self.categorias):
            btn = ttk.Button(left_frame, text=nombre, command=comando, width=15)
            btn.pack(pady=5, padx=5, fill=tk.X)
            self.botones_categorias.append(btn)
        
        # Botones de accion
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="Guardar", command=self.guardar_configuracion).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Cargar", command=self.cargar_configuracion).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Valores por Defecto", command=self.restaurar_valores_defecto).pack(pady=5, fill=tk.X)
        
        # ===== PANEL DERECHO - CONTENIDO =====
        self.contenido_frame = ttk.Frame(right_frame)
        self.contenido_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Estado
        self.estado = ttk.Label(right_frame, text="Configuración cargada", foreground='green')
        self.estado.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Mostrar primera categoria por defecto
        self.mostrar_directorios()
    
    def crear_campo_entrada(self, frame, fila, etiqueta, variable, tipo='texto', opciones=None):
        """Crea un campo de entrada estandarizado"""
        ttk.Label(frame, text=etiqueta).grid(row=fila, column=0, sticky=tk.W, pady=5, padx=5)
        
        if tipo == 'texto':
            entry = ttk.Entry(frame, textvariable=variable, width=40)
            entry.grid(row=fila, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
            entry.bind('<KeyRelease>', lambda e: self.marcar_modificada())
            return entry
        
        elif tipo == 'booleano':
            check = ttk.Checkbutton(frame, variable=variable, 
                                  command=self.marcar_modificada)
            check.grid(row=fila, column=1, sticky=tk.W, pady=5, padx=5)
            return check
        
        elif tipo == 'numerico':
            spin = ttk.Spinbox(frame, from_=opciones['min'], to=opciones['max'], 
                              textvariable=variable, width=10)
            spin.grid(row=fila, column=1, sticky=tk.W, pady=5, padx=5)
            spin.bind('<KeyRelease>', lambda e: self.marcar_modificada())
            return spin
        
        elif tipo == 'directorio':
            subframe = ttk.Frame(frame)
            subframe.grid(row=fila, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
            
            entry = ttk.Entry(subframe, textvariable=variable, width=35)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entry.bind('<KeyRelease>', lambda e: self.marcar_modificada())
            
            ttk.Button(subframe, text="Examinar", 
                      command=lambda: self.seleccionar_directorio(variable)).pack(side=tk.RIGHT, padx=(5, 0))
            return entry
    
    def mostrar_directorios(self):
        """Configuración de directorios"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuración de Directorios", 
                font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        # Sección de entrenamiento
        ttk.Label(frame, text="Directorios de Entrenamiento", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.crear_campo_entrada(frame, 1, "Directorio de Tumor:", 
                                self.dir_entrenamiento_tumor, 'directorio')
        self.crear_campo_entrada(frame, 2, "Directorio de No-Tumor:", 
                                self.dir_entrenamiento_notumor, 'directorio')

        # Sección de pruebas
        ttk.Label(frame, text="Directorios de Prueba", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        self.crear_campo_entrada(frame, 4, "Directorio de Tumor:", 
                                self.dir_prueba_tumor, 'directorio')
        self.crear_campo_entrada(frame, 5, "Directorio de No-Tumor:", 
                                self.dir_prueba_notumor, 'directorio')
    
    def mostrar_procesamiento(self):
        """Configuración de procesamiento"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuración de Procesamiento", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        self.crear_campo_entrada(frame, 0, "Guardar Imágenes Intermedias:", 
                                self.guardar_imagenes, 'booleano')
        self.crear_campo_entrada(frame, 1, "Directorio Imágenes Intermedias:", 
                                self.dir_imagenes, 'directorio')
        self.crear_campo_entrada(frame, 2, "Normalizar Características:", 
                                self.normalizar_caracteristicas, 'booleano')
    
    def mostrar_entrenamiento(self):
        """Configuración de entrenamiento"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuración de Entrenamiento", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        self.crear_campo_entrada(frame, 0, "Tamaño del Enjambre PSO:", 
                                self.tamano_enjambre, 'numerico', {'min': 10, 'max': 100})
        self.crear_campo_entrada(frame, 1, "Máximo de Iteraciones:", 
                                self.max_iteraciones, 'numerico', {'min': 5, 'max': 50})
        self.crear_campo_entrada(frame, 3, "Nombre del Modelo:", 
                                self.nombre_modelo, 'texto')
    
    def mostrar_cache(self):
        """Configuración del sistema de caché - SOLO GUARDADO"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuración del Sistema de Caché", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        # SOLO configuración de GUARDADO
        ttk.Label(frame, text="¿Qué se debe GUARDAR en caché?", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(10, 5))
        
        self.crear_campo_entrada(frame, 1, "Guardar Características en Caché:", 
                                self.guardar_cache_caracteristicas, 'booleano')
        self.crear_campo_entrada(frame, 2, "Guardar Modelos en Caché:", 
                                self.guardar_cache_modelos, 'booleano')
        self.crear_campo_entrada(frame, 3, "Guardar Graficos en Caché:", 
                                self.guardar_cache_graficos, 'booleano')
        self.crear_campo_entrada(frame, 4, "Guardar Métricas en Caché:", 
                                self.guardar_cache_metricas, 'booleano')
        self.crear_campo_entrada(frame, 5, "Guardar Reportes en Caché:", 
                                self.guardar_cache_reportes, 'booleano')
        self.crear_campo_entrada(frame, 6, "Guardar Datos de Reglas en Caché:", 
                                self.guardar_cache_datos_reglas, 'booleano')
        
        # Información
        ttk.Label(frame, text="Nota: El USO de caché existente se controla desde la ventana principal durante la ejecución.", 
                foreground='gray', font=('Arial', 9)).grid(row=7, column=0, columnspan=2, pady=10)
    
    def mostrar_analisis(self):
        """Configuración de análisis"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuración de Análisis", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        self.crear_campo_entrada(frame, 0, "Número de Reglas a Mostrar:", 
                                self.top_reglas, 'numerico', {'min': 5, 'max': 50})
        self.crear_campo_entrada(frame, 1, "Guardar Metricas:", 
                                self.guardar_metricas, 'booleano')
        self.crear_campo_entrada(frame, 2, "Guardar Reportes:", 
                                self.guardar_reportes, 'booleano')
        self.crear_campo_entrada(frame, 3, "Guardar Datos de Reglas:", 
                                self.guardar_datos_reglas, 'booleano')
        self.crear_campo_entrada(frame, 4, "Guardar Gráficos de Análisis:", 
                                self.guardar_graficos_analisis, 'booleano')
        self.crear_campo_entrada(frame, 5, "Directorio de Análisis:", 
                                self.dir_analisis, 'directorio')
    
    def limpiar_contenido(self):
        """Limpia el frame de contenido"""
        for widget in self.contenido_frame.winfo_children():
            widget.destroy()
    
    def seleccionar_directorio(self, variable):
        """Abre dialogo para seleccionar directorio"""
        directorio = filedialog.askdirectory(title="Seleccionar directorio")
        if directorio:
            variable.set(directorio)
            self.marcar_modificada()
    
    def cargar_configuracion_actual(self):
        """Carga la configuración actual en las variables - CORREGIDO"""
        # Directorios
        self.dir_entrenamiento_tumor.set(config.directorio_entrenamiento_tumor)
        self.dir_entrenamiento_notumor.set(config.directorio_entrenamiento_notumor)
        self.dir_prueba_tumor.set(config.directorio_prueba_tumor)
        self.dir_prueba_notumor.set(config.directorio_prueba_notumor)
        
        # Procesamiento
        self.guardar_imagenes.set(config.procesamiento.guardar_imagenes_intermedias)
        self.dir_imagenes.set(config.procesamiento.directorio_imagenes_intermedias)
        self.normalizar_caracteristicas.set(config.procesamiento.normalizar_caracteristicas)
        
        # Entrenamiento
        self.tamano_enjambre.set(config.entrenamiento.tamano_enjambre)
        self.max_iteraciones.set(config.entrenamiento.max_iteraciones)
        self.nombre_modelo.set(config.entrenamiento.nombre_modelo)
        
        # Cache - SOLO configuración de GUARDADO (nueva estructura)
        self.guardar_cache_caracteristicas.set(config.cache.guardar_cache_caracteristicas)
        self.guardar_cache_modelos.set(config.cache.guardar_cache_modelos)
        self.guardar_cache_graficos.set(config.cache.guardar_cache_graficos)
        self.guardar_cache_metricas.set(config.cache.guardar_cache_metricas)
        self.guardar_cache_reportes.set(config.cache.guardar_cache_reportes)
        self.guardar_cache_datos_reglas.set(config.cache.guardar_cache_datos_reglas)
        
        # Analisis
        self.top_reglas.set(config.analisis.top_reglas_mostrar)
        self.guardar_metricas.set(config.analisis.guardar_metricas)
        self.guardar_reportes.set(config.analisis.guardar_reportes)
        self.guardar_datos_reglas.set(config.analisis.guardar_datos_reglas)
        self.guardar_graficos_analisis.set(config.analisis.guardar_graficos_analisis)
        self.dir_analisis.set(config.analisis.directorio_analisis)
        
        self.config_modificada = False
        self.actualizar_estado("Configuración cargada")
    
    def guardar_configuracion(self):
        """Guarda la configuración desde las variables a la configuración global - CORREGIDO"""
        # Directorios
        config.directorio_entrenamiento_tumor = self.dir_entrenamiento_tumor.get()
        config.directorio_entrenamiento_notumor = self.dir_entrenamiento_notumor.get()
        config.directorio_prueba_tumor = self.dir_prueba_tumor.get()
        config.directorio_prueba_notumor = self.dir_prueba_notumor.get()
        
        # Procesamiento
        config.procesamiento.guardar_imagenes_intermedias = self.guardar_imagenes.get()
        config.procesamiento.directorio_imagenes_intermedias = self.dir_imagenes.get()
        config.procesamiento.normalizar_caracteristicas = self.normalizar_caracteristicas.get()
        
        # Entrenamiento
        config.entrenamiento.tamano_enjambre = self.tamano_enjambre.get()
        config.entrenamiento.max_iteraciones = self.max_iteraciones.get()
        config.entrenamiento.nombre_modelo = self.nombre_modelo.get()
        
        # Cache - SOLO configuración de GUARDADO (nueva estructura)
        config.cache.guardar_cache_caracteristicas = self.guardar_cache_caracteristicas.get()
        config.cache.guardar_cache_modelos = self.guardar_cache_modelos.get()
        config.cache.guardar_cache_graficos = self.guardar_cache_graficos.get()
        config.cache.guardar_cache_metricas = self.guardar_cache_metricas.get()
        config.cache.guardar_cache_reportes = self.guardar_cache_reportes.get()
        config.cache.guardar_cache_datos_reglas = self.guardar_cache_datos_reglas.get()
        
        # Analisis
        config.analisis.top_reglas_mostrar = self.top_reglas.get()
        config.analisis.guardar_metricas = self.guardar_metricas.get()
        config.analisis.guardar_reportes = self.guardar_reportes.get()
        config.analisis.guardar_datos_reglas = self.guardar_datos_reglas.get()
        config.analisis.guardar_graficos_analisis = self.guardar_graficos_analisis.get()
        config.analisis.directorio_analisis = self.dir_analisis.get()
        
        # Guardar en archivo
        config.guardar_configuracion()
        self.config_modificada = False
        self.actualizar_estado("Configuración guardada exitosamente")
        messagebox.showinfo("Configuración", "Configuración guardada correctamente.")
    
    def cargar_configuracion(self):
        """Carga la configuración desde archivo"""
        if self.config_modificada:
            if not messagebox.askyesno("Confirmar", "Hay cambios sin guardar. ¿Desea cargar de todos modos?"):
                return
        
        config.cargar_configuracion()
        self.cargar_configuracion_actual()
        self.actualizar_estado("Configuración cargada desde archivo")
    
    def restaurar_valores_defecto(self):
        """Restaura los valores por defecto"""
        if not messagebox.askyesno("Confirmar", "¿Restaurar valores por defecto? Se perderán los cambios no guardados."):
            return
        
        from config.configuracion import ConfiguracionGlobal
        config_defecto = ConfiguracionGlobal()
        
        config.directorio_entrenamiento_tumor = config_defecto.directorio_entrenamiento_tumor
        config.directorio_entrenamiento_notumor = config_defecto.directorio_entrenamiento_notumor
        config.directorio_prueba_tumor = config_defecto.directorio_prueba_tumor
        config.directorio_prueba_notumor = config_defecto.directorio_prueba_notumor
        config.procesamiento = config_defecto.procesamiento
        config.entrenamiento = config_defecto.entrenamiento
        config.cache = config_defecto.cache
        config.analisis = config_defecto.analisis
        
        self.cargar_configuracion_actual()
        self.actualizar_estado("Valores por defecto restaurados")
    
    def marcar_modificada(self):
        """Marca la configuración como modificada"""
        self.config_modificada = True
        self.actualizar_estado("Configuración modificada - No guardada")
    
    def actualizar_estado(self, mensaje):
        """Actualiza el mensaje de estado"""
        self.estado.config(text=mensaje)
        if "modificada" in mensaje.lower():
            self.estado.config(foreground='orange')
        elif "error" in mensaje.lower():
            self.estado.config(foreground='red')
        else:
            self.estado.config(foreground='green')

def main():
    root = tk.Tk()
    app = Configurador(root)
    root.mainloop()

if __name__ == "__main__":
    main()