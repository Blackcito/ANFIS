# interfaz/configurador.py - CORREGIDO

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
        
        # INICIALIZAR TODAS LAS VARIABLES AL PRINCIPIO
        self.inicializar_variables()
        
        self.crear_interfaz()
        self.cargar_configuracion_actual()
    
    def inicializar_variables(self):
        """Inicializa todas las variables de configuración"""
        # Directorios
        self.dir_entrenamiento = tk.StringVar()
        self.dir_prueba = tk.StringVar()
        
        # Procesamiento
        self.guardar_imagenes = tk.BooleanVar()
        self.dir_imagenes = tk.StringVar()
        self.usar_cache_imagenes = tk.BooleanVar()
        self.normalizar_caracteristicas = tk.BooleanVar()
        
        # Entrenamiento
        self.tamano_enjambre = tk.IntVar()
        self.max_iteraciones = tk.IntVar()
        self.guardar_modelo = tk.BooleanVar()
        self.nombre_modelo = tk.StringVar()
        
        # Cache
        self.usar_cache_caracteristicas = tk.BooleanVar()
        self.usar_cache_modelos = tk.BooleanVar()
        self.usar_cache_resultados = tk.BooleanVar()
        self.limpiar_cache_auto = tk.BooleanVar()
        
        # Analisis
        self.top_reglas = tk.IntVar()
        self.guardar_graficos = tk.BooleanVar()
        self.guardar_reportes = tk.BooleanVar()
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
        ttk.Label(left_frame, text="Categorias", font=('Arial', 11, 'bold')).pack(pady=(0, 10))
        
        self.categorias = [
            ("Directorios", self.mostrar_directorios),
            ("Entrenamiento", self.mostrar_entrenamiento),
            ("Analisis", self.mostrar_analisis)
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
        self.estado = ttk.Label(right_frame, text="Configuracion cargada", foreground='green')
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
        """Solo directorios, sin opciones de cache"""
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuracion de Directorios", 
                font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        self.crear_campo_entrada(frame, 0, "Directorio de Entrenamiento:", 
                                self.dir_entrenamiento, 'directorio')
        self.crear_campo_entrada(frame, 1, "Directorio de Prueba:", 
                                self.dir_prueba, 'directorio')
        
        # INFO: Cache se controla desde ventana principal
        ttk.Label(frame, text="Nota: El uso de cache se controla desde la ventana principal", 
                foreground='gray', font=('Arial', 9)).grid(row=2, column=0, columnspan=2, pady=10)
    
    def mostrar_procesamiento(self):
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuracion de Procesamiento", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        # Usar variables ya inicializadas
        self.crear_campo_entrada(frame, 0, "Guardar Imagenes Intermedias:", 
                                self.guardar_imagenes, 'booleano')
        self.crear_campo_entrada(frame, 1, "Directorio Imagenes Intermedias:", 
                                self.dir_imagenes, 'directorio')
        self.crear_campo_entrada(frame, 2, "Usar Cache de Imagenes:", 
                                self.usar_cache_imagenes, 'booleano')
        self.crear_campo_entrada(frame, 3, "Normalizar Caracteristicas:", 
                                self.normalizar_caracteristicas, 'booleano')
    
    def mostrar_entrenamiento(self):
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuracion de Entrenamiento", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        # Usar variables ya inicializadas
        self.crear_campo_entrada(frame, 0, "Tamano del Enjambre PSO:", 
                                self.tamano_enjambre, 'numerico', {'min': 10, 'max': 100})
        self.crear_campo_entrada(frame, 1, "Maximo de Iteraciones:", 
                                self.max_iteraciones, 'numerico', {'min': 5, 'max': 50})
        self.crear_campo_entrada(frame, 2, "Guardar Modelo Automaticamente:", 
                                self.guardar_modelo, 'booleano')
        self.crear_campo_entrada(frame, 3, "Nombre del Modelo:", 
                                self.nombre_modelo, 'texto')
    
    def mostrar_cache(self):
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuracion de Cache", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        # Usar variables ya inicializadas
        self.crear_campo_entrada(frame, 0, "Usar Cache de Caracteristicas:", 
                                self.usar_cache_caracteristicas, 'booleano')
        self.crear_campo_entrada(frame, 1, "Usar Cache de Modelos:", 
                                self.usar_cache_modelos, 'booleano')
        self.crear_campo_entrada(frame, 2, "Usar Cache de Resultados:", 
                                self.usar_cache_resultados, 'booleano')
        self.crear_campo_entrada(frame, 3, "Limpieza Automatica de Cache:", 
                                self.limpiar_cache_auto, 'booleano')
    
    def mostrar_analisis(self):
        self.limpiar_contenido()
        ttk.Label(self.contenido_frame, text="Configuracion de Analisis", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 15))
        
        frame = ttk.Frame(self.contenido_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        
        # Usar variables ya inicializadas
        self.crear_campo_entrada(frame, 0, "Numero de Reglas a Mostrar:", 
                                self.top_reglas, 'numerico', {'min': 5, 'max': 50})
        self.crear_campo_entrada(frame, 1, "Guardar Graficos:", 
                                self.guardar_graficos, 'booleano')
        self.crear_campo_entrada(frame, 2, "Guardar Reportes:", 
                                self.guardar_reportes, 'booleano')
        self.crear_campo_entrada(frame, 3, "Directorio de Analisis:", 
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
        """Carga la configuracion actual en las variables (que ya están inicializadas)"""
        # Directorios
        self.dir_entrenamiento.set(config.directorio_entrenamiento)
        self.dir_prueba.set(config.directorio_prueba)
        
        # Procesamiento
        self.guardar_imagenes.set(config.procesamiento.guardar_imagenes_intermedias)
        self.dir_imagenes.set(config.procesamiento.directorio_imagenes_intermedias)
        self.usar_cache_imagenes.set(config.procesamiento.usar_cache_imagenes)
        self.normalizar_caracteristicas.set(config.procesamiento.normalizar_caracteristicas)
        
        # Entrenamiento
        self.tamano_enjambre.set(config.entrenamiento.tamano_enjambre)
        self.max_iteraciones.set(config.entrenamiento.max_iteraciones)
        self.guardar_modelo.set(config.entrenamiento.guardar_modelo)
        self.nombre_modelo.set(config.entrenamiento.nombre_modelo)
        
        # Cache
        self.usar_cache_caracteristicas.set(config.cache.usar_cache_caracteristicas)
        self.usar_cache_modelos.set(config.cache.usar_cache_modelos)
        self.usar_cache_resultados.set(config.cache.usar_cache_resultados)
        self.limpiar_cache_auto.set(config.cache.limpiar_cache_automatico)
        
        # Analisis
        self.top_reglas.set(config.analisis.top_reglas_mostrar)
        self.guardar_graficos.set(config.analisis.guardar_graficos)
        self.guardar_reportes.set(config.analisis.guardar_reportes)
        self.dir_analisis.set(config.analisis.directorio_analisis)
        
        self.config_modificada = False
        self.actualizar_estado("Configuracion cargada")
    
    def guardar_configuracion(self):
        """Guarda la configuracion desde las variables a la configuracion global"""
        # Directorios
        config.directorio_entrenamiento = self.dir_entrenamiento.get()
        config.directorio_prueba = self.dir_prueba.get()
        
        # Procesamiento
        config.procesamiento.guardar_imagenes_intermedias = self.guardar_imagenes.get()
        config.procesamiento.directorio_imagenes_intermedias = self.dir_imagenes.get()
        config.procesamiento.usar_cache_imagenes = self.usar_cache_imagenes.get()
        config.procesamiento.normalizar_caracteristicas = self.normalizar_caracteristicas.get()
        
        # Entrenamiento
        config.entrenamiento.tamano_enjambre = self.tamano_enjambre.get()
        config.entrenamiento.max_iteraciones = self.max_iteraciones.get()
        config.entrenamiento.guardar_modelo = self.guardar_modelo.get()
        config.entrenamiento.nombre_modelo = self.nombre_modelo.get()
        
        # Cache
        config.cache.usar_cache_caracteristicas = self.usar_cache_caracteristicas.get()
        config.cache.usar_cache_modelos = self.usar_cache_modelos.get()
        config.cache.usar_cache_resultados = self.usar_cache_resultados.get()
        config.cache.limpiar_cache_automatico = self.limpiar_cache_auto.get()
        
        # Analisis
        config.analisis.top_reglas_mostrar = self.top_reglas.get()
        config.analisis.guardar_graficos = self.guardar_graficos.get()
        config.analisis.guardar_reportes = self.guardar_reportes.get()
        config.analisis.directorio_analisis = self.dir_analisis.get()
        
        # Guardar en archivo
        config.guardar_configuracion()
        self.config_modificada = False
        self.actualizar_estado("Configuracion guardada exitosamente")
        messagebox.showinfo("Configuracion", "Configuracion guardada correctamente.")
    
    def cargar_configuracion(self):
        """Carga la configuracion desde archivo"""
        if self.config_modificada:
            if not messagebox.askyesno("Confirmar", "Hay cambios sin guardar. ¿Desea cargar de todos modos?"):
                return
        
        config.cargar_configuracion()
        self.cargar_configuracion_actual()
        self.actualizar_estado("Configuracion cargada desde archivo")
    
    def restaurar_valores_defecto(self):
        """Restaura los valores por defecto"""
        if not messagebox.askyesno("Confirmar", "¿Restaurar valores por defecto? Se perderan los cambios no guardados."):
            return
        
        # Crear nueva instancia para obtener valores por defecto
        from config.configuracion import ConfiguracionGlobal
        config_defecto = ConfiguracionGlobal()
        
        # Actualizar configuracion actual
        config.directorio_entrenamiento = config_defecto.directorio_entrenamiento
        config.directorio_prueba = config_defecto.directorio_prueba
        config.procesamiento = config_defecto.procesamiento
        config.entrenamiento = config_defecto.entrenamiento
        config.cache = config_defecto.cache
        config.analisis = config_defecto.analisis
        
        self.cargar_configuracion_actual()
        self.actualizar_estado("Valores por defecto restaurados")
    
    def marcar_modificada(self):
        """Marca la configuracion como modificada"""
        self.config_modificada = True
        self.actualizar_estado("Configuracion modificada - No guardada")
    
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