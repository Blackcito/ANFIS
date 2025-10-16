# interfaz/ventana_principal.py - ACTUALIZADO

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import sys

# Agregar el directorio raíz al path para importar nuestros módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import pipeline_completo, usar_modelo_guardado
from utils.cache import sistema_cache
from config.configuracion import config

class VentanaPrincipal:
    def __init__(self, root):
        self.root = root
        self.root.title("ANFIS - Deteccion de Tumores Cerebrales")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables para almacenar rutas y selecciones
        self.train_dir = tk.StringVar(value=config.directorio_entrenamiento)
        self.test_dir = tk.StringVar(value=config.directorio_prueba)
        self.modelo_seleccionado = tk.StringVar()
        self.cache_seleccionado = tk.StringVar()
        
        self.crear_interfaz()
        self.actualizar_listas()
        
    def crear_interfaz(self):
        # Frame principal con paned window para mejor distribucion
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Panel derecho - Resultados
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # ===== PANEL IZQUIERDO - CONTROLES =====
        # Titulo
        titulo = ttk.Label(left_frame, 
                          text="ANFIS - Sistema de Deteccion de Tumores Cerebrales", 
                          font=('Arial', 14, 'bold'))
        titulo.pack(pady=(0, 20))
        
        # Frame de configuracion de datos
        config_frame = ttk.LabelFrame(left_frame, text="Configuracion de Datos", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(config_frame, text="Carpeta de Entrenamiento:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.train_dir, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(config_frame, text="Seleccionar", 
                  command=self.seleccionar_train_dir).grid(row=0, column=2, pady=5)
        
        ttk.Label(config_frame, text="Carpeta de Prueba:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(config_frame, textvariable=self.test_dir, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(config_frame, text="Seleccionar", 
                  command=self.seleccionar_test_dir).grid(row=1, column=2, pady=5)
        
        config_frame.columnconfigure(1, weight=1)
        
        # Frame de seleccion de modelos
        modelo_frame = ttk.LabelFrame(left_frame, text="Seleccion de Modelo Entrenado", padding="10")
        modelo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(modelo_frame, text="Modelo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.combo_modelos = ttk.Combobox(modelo_frame, textvariable=self.modelo_seleccionado, width=30)
        self.combo_modelos.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(modelo_frame, text="Actualizar", 
                  command=lambda: self.actualizar_listas('modelos')).grid(row=0, column=2, pady=5)
        
        ttk.Button(modelo_frame, text="Eliminar Seleccionado", 
                  command=self.eliminar_modelo_seleccionado).grid(row=1, column=2, pady=5)
        
        modelo_frame.columnconfigure(1, weight=1)
        
        # Frame de seleccion de cache
        cache_frame = ttk.LabelFrame(left_frame, text="Seleccion de Cache de Caracteristicas", padding="10")
        cache_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(cache_frame, text="Cache:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.combo_caches = ttk.Combobox(cache_frame, textvariable=self.cache_seleccionado, width=30)
        self.combo_caches.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(cache_frame, text="Actualizar", 
                  command=lambda: self.actualizar_listas('caches')).grid(row=0, column=2, pady=5)
        
        ttk.Button(cache_frame, text="Eliminar Seleccionado", 
                  command=self.eliminar_cache_seleccionado).grid(row=1, column=2, pady=5)
        
        cache_frame.columnconfigure(1, weight=1)
        
        # Frame de operaciones principales
        ops_frame = ttk.LabelFrame(left_frame, text="Operaciones Principales", padding="10")
        ops_frame.pack(fill=tk.X, pady=(0, 10))
        
        botones_principales = [
            ("Pipeline Completo (Nuevo Modelo)", self.ejecutar_pipeline_nuevo),
            ("Pipeline Completo (Modelo Guardado)", self.ejecutar_pipeline_guardado),
            ("Solo Evaluacion (Modelo Guardado)", self.ejecutar_solo_evaluacion),
            ("Usar Cache Especifico", self.usar_cache_especifico)
        ]
        
        for i, (texto, comando) in enumerate(botones_principales):
            ttk.Button(ops_frame, text=texto, command=comando,
                      style='Accent.TButton').grid(row=i, column=0, padx=5, pady=5, sticky='ew')
        
        # Frame de utilidades
        util_frame = ttk.LabelFrame(left_frame, text="Utilidades", padding="10")
        util_frame.pack(fill=tk.X, pady=(0, 10))
        
        botones_util = [
            ("Configuracion del Sistema", self.abrir_configuracion),
            ("Ver Graficos", self.mostrar_graficos),
            ("Limpiar Cache Modelos", self.limpiar_cache_modelos),
            ("Limpiar Cache Caracteristicas", self.limpiar_cache_caracteristicas),
            ("Estadisticas de Cache", self.mostrar_estadisticas_cache)
        ]
        
        for i, (texto, comando) in enumerate(botones_util):
            ttk.Button(util_frame, text=texto, command=comando).grid(row=i, column=0, padx=5, pady=5, sticky='ew')
        
        # ===== PANEL DERECHO - RESULTADOS =====
        # Area de resultados
        resultados_frame = ttk.LabelFrame(right_frame, text="Resultados y Logs", padding="10")
        resultados_frame.pack(fill=tk.BOTH, expand=True)
        
        self.texto_resultados = scrolledtext.ScrolledText(resultados_frame, width=80, height=30, font=('Consolas', 10))
        self.texto_resultados.pack(fill=tk.BOTH, expand=True)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Estado
        self.estado = ttk.Label(right_frame, text="Listo", foreground='green')
        self.estado.pack(side=tk.LEFT)
        
        # Configurar estilo
        self.configurar_estilos()
        
        self.log("Sistema ANFIS inicializado. Seleccione una operacion para comenzar.")
    
    def configurar_estilos(self):
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    def actualizar_listas(self, tipo=None):
        """Actualiza las listas de modelos y caches disponibles"""
        if tipo is None or tipo == 'modelos':
            modelos = sistema_cache.listar_modelos()
            self.combo_modelos['values'] = modelos
            if modelos:
                self.modelo_seleccionado.set(modelos[0])
        
        if tipo is None or tipo == 'caches':
            caches = sistema_cache.listar_caracteristicas()
            self.combo_caches['values'] = caches
            if caches:
                self.cache_seleccionado.set(caches[0])
    
    def seleccionar_train_dir(self):
        directorio = filedialog.askdirectory(title="Seleccionar carpeta de entrenamiento")
        if directorio:
            self.train_dir.set(directorio)
            self.log(f"Carpeta de entrenamiento seleccionada: {directorio}")
    
    def seleccionar_test_dir(self):
        directorio = filedialog.askdirectory(title="Seleccionar carpeta de prueba")
        if directorio:
            self.test_dir.set(directorio)
            self.log(f"Carpeta de prueba seleccionada: {directorio}")
    
    def eliminar_modelo_seleccionado(self):
        modelo = self.modelo_seleccionado.get()
        if modelo and messagebox.askyesno("Confirmar", f"¿Eliminar el modelo {modelo}?"):
            if sistema_cache.eliminar_modelo(modelo):
                self.log(f"Modelo eliminado: {modelo}")
                self.actualizar_listas('modelos')
            else:
                self.log(f"Error eliminando modelo: {modelo}")
    
    def eliminar_cache_seleccionado(self):
        cache = self.cache_seleccionado.get()
        if cache and messagebox.askyesno("Confirmar", f"¿Eliminar el cache {cache}?"):
            if sistema_cache.eliminar_caracteristicas(cache):
                self.log(f"Cache eliminado: {cache}")
                self.actualizar_listas('caches')
            else:
                self.log(f"Error eliminando cache: {cache}")
    
    def ejecutar_pipeline_nuevo(self):
        self.log("Iniciando pipeline completo con nuevo modelo...")
        threading.Thread(target=self._ejecutar_pipeline, args=(True, None), daemon=True).start()
    
    def ejecutar_pipeline_guardado(self):
        modelo = self.modelo_seleccionado.get()
        if not modelo:
            messagebox.showwarning("Advertencia", "Seleccione un modelo primero")
            return
        
        self.log(f"Iniciando pipeline completo con modelo: {modelo}")
        threading.Thread(target=self._ejecutar_pipeline, args=(False, modelo), daemon=True).start()
    
    def _ejecutar_pipeline(self, entrenar_nuevo, nombre_modelo=None):
        self.mostrar_progreso(True)
        try:
            # Actualizar configuracion con las rutas seleccionadas
            config.directorio_entrenamiento = self.train_dir.get()
            config.directorio_prueba = self.test_dir.get()
            config.guardar_configuracion()
            
            if entrenar_nuevo:
                resultado = pipeline_completo(use_cache=True, entrenar_nuevo=True)
            else:
                # Usar modelo especifico
                from main import pipeline_completo_con_modelo
                resultado = pipeline_completo_con_modelo(nombre_modelo, use_cache=True)
            
            self.log("Pipeline completado exitosamente!")
            
            if resultado and 'evaluacion' in resultado:
                metricas = resultado['evaluacion']['metricas']
                self.log("Metricas obtenidas:")
                self.log(f"  - Precision: {metricas['precision']:.4f}")
                self.log(f"  - Sensibilidad: {metricas['sensitivity']:.4f}")
                self.log(f"  - Especificidad: {metricas['specificity']:.4f}")
                self.log(f"  - F1-Score: {metricas['f1_score']:.4f}")
                if metricas['auc']:
                    self.log(f"  - AUC-ROC: {metricas['auc']:.4f}")
            
            # Actualizar listas despues de ejecucion
            self.actualizar_listas()
                    
        except Exception as e:
            self.log(f"Error en pipeline: {str(e)}")
            messagebox.showerror("Error", f"Error durante la ejecucion:\n{str(e)}")
        finally:
            self.mostrar_progreso(False)
    
    def ejecutar_solo_evaluacion(self):
        modelo = self.modelo_seleccionado.get()
        if not modelo:
            messagebox.showwarning("Advertencia", "Seleccione un modelo primero")
            return
        
        self.log(f"Evaluando modelo: {modelo}")
        threading.Thread(target=self._ejecutar_evaluacion, args=(modelo,), daemon=True).start()
    
    def _ejecutar_evaluacion(self, nombre_modelo):
        self.mostrar_progreso(True)
        try:
            config.directorio_entrenamiento = self.train_dir.get()
            config.directorio_prueba = self.test_dir.get()
            config.guardar_configuracion()
            
            resultado = usar_modelo_guardado(nombre_modelo=nombre_modelo)
            if resultado:
                self.log("Evaluacion completada exitosamente")
            
        except Exception as e:
            self.log(f"Error en evaluacion: {str(e)}")
        finally:
            self.mostrar_progreso(False)
    
    def usar_cache_especifico(self):
        cache = self.cache_seleccionado.get()
        if not cache:
            messagebox.showwarning("Advertencia", "Seleccione un cache primero")
            return
        
        self.log(f"Usando cache especifico: {cache}")
        # Aqui implementarias la logica para usar un cache especifico
        # Esto requeriria modificar main.py para aceptar un parametro de cache
    
    def abrir_configuracion(self):
        self.log("Abriendo configurador del sistema...")
        try:
            from interfaz.configurador import Configurador
            ventana_config = tk.Toplevel(self.root)
            Configurador(ventana_config)
        except Exception as e:
            self.log(f"Error al abrir configurador: {str(e)}")
    
    def mostrar_graficos(self):
        self.log("Abriendo visualizador de graficos...")
        threading.Thread(target=self._abrir_visualizador, daemon=True).start()
    
    def _abrir_visualizador(self):
        try:
            from interfaz.visualizador_graficos import VentanaGraficos
            ventana_graficos = tk.Toplevel(self.root)
            VentanaGraficos(ventana_graficos)
        except Exception as e:
            self.log(f"Error al abrir visualizador: {str(e)}")
    
    def limpiar_cache_modelos(self):
        if messagebox.askyesno("Confirmar", "¿Está seguro de limpiar todo el cache de modelos?"):
            sistema_cache.limpiar_cache_modelos()
            self.log("Cache de modelos limpiado")
            self.actualizar_listas('modelos')
    
    def limpiar_cache_caracteristicas(self):
        if messagebox.askyesno("Confirmar", "¿Está seguro de limpiar todo el cache de características?"):
            sistema_cache.limpiar_cache_caracteristicas()
            self.log("Cache de características limpiado")
            self.actualizar_listas('caches')
    
    def mostrar_estadisticas_cache(self):
        stats = sistema_cache.obtener_estadisticas_cache()
        self.log("Estadisticas de cache:")
        for tipo, datos in stats.items():
            self.log(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")
    
    def log(self, mensaje):
        """Agregar mensaje al area de texto"""
        self.texto_resultados.insert(tk.END, f"{mensaje}\n")
        self.texto_resultados.see(tk.END)
        self.estado.config(text=mensaje)
        self.root.update_idletasks()
    
    def mostrar_progreso(self, activo):
        """Mostrar u ocultar barra de progreso"""
        if activo:
            self.progress.start()
            self.estado.config(text="Procesando...", foreground='orange')
        else:
            self.progress.stop()
            self.estado.config(text="Listo", foreground='green')

def main():
    root = tk.Tk()
    app = VentanaPrincipal(root)
    root.mainloop()

if __name__ == "__main__":
    main()