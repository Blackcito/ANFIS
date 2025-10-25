# interfaz/ventana_principal.py - SISTEMA UNIFICADO

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import sys
import io

# Agregar el directorio raíz al path para importar nuestros módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import pipeline_completo, usar_modelo_guardado
from utils.cache import sistema_cache
from config.configuracion import config

class OutputRedirector(io.StringIO):
    """Redirige stdout a la ventana principal"""
    def __init__(self, ventana):
        super().__init__()
        self.ventana = ventana
    
    def write(self, text):
        # Enviar a la ventana principal
        if text.strip():  # Solo texto no vacío
            self.ventana.log(text.strip())
        return len(text)
    
    def flush(self):
        pass

class VentanaPrincipal:
    def __init__(self, root):
        self.root = root
        self.root.title("ANFIS - Detección de Tumores Cerebrales")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables para almacenar rutas y selecciones
        self.train_dir = tk.StringVar(value=config.directorio_entrenamiento)
        self.test_dir = tk.StringVar(value=config.directorio_prueba)
        
        # NUEVAS VARIABLES PARA EL SISTEMA UNIFICADO
        self.usar_modelo = tk.BooleanVar(value=False)
        self.usar_cache_entrenamiento = tk.BooleanVar(value=False)
        self.usar_cache_prueba = tk.BooleanVar(value=False)
        self.modelo_seleccionado = tk.StringVar()
        self.cache_entrenamiento_seleccionado = tk.StringVar()
        self.cache_prueba_seleccionado = tk.StringVar()
        
        # REDIRECCIÓN DE OUTPUT
        self.output_redirector = OutputRedirector(self)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector
        
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
                          text="ANFIS - Sistema de Detección de Tumores Cerebrales", 
                          font=('Arial', 14, 'bold'))
        titulo.pack(pady=(0, 20))
        
        # Frame de configuracion de datos
        config_frame = ttk.LabelFrame(left_frame, text="Configuración de Datos", padding="10")
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
        
        # ===== NUEVA SECCIÓN: CONFIGURACIÓN DE RECURSOS =====
        recursos_frame = ttk.LabelFrame(left_frame, text="Configuración de Recursos", padding="10")
        recursos_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Modelo
        ttk.Label(recursos_frame, text="Modelo:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.combo_modelos = ttk.Combobox(recursos_frame, textvariable=self.modelo_seleccionado, width=25)
        self.combo_modelos.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.chk_usar_modelo = ttk.Checkbutton(recursos_frame, text="Usar", variable=self.usar_modelo)
        self.chk_usar_modelo.grid(row=0, column=2, pady=2)
        
        # Cache Entrenamiento
        ttk.Label(recursos_frame, text="Cache Entrenamiento:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.combo_cache_entrenamiento = ttk.Combobox(recursos_frame, textvariable=self.cache_entrenamiento_seleccionado, width=25)
        self.combo_cache_entrenamiento.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.chk_usar_cache_entrenamiento = ttk.Checkbutton(recursos_frame, text="Usar", 
                                                          variable=self.usar_cache_entrenamiento)
        self.chk_usar_cache_entrenamiento.grid(row=1, column=2, pady=2)
        
        # Cache Prueba
        ttk.Label(recursos_frame, text="Cache Prueba:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.combo_cache_prueba = ttk.Combobox(recursos_frame, textvariable=self.cache_prueba_seleccionado, width=25)
        self.combo_cache_prueba.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.chk_usar_cache_prueba = ttk.Checkbutton(recursos_frame, text="Usar", 
                                                   variable=self.usar_cache_prueba)
        self.chk_usar_cache_prueba.grid(row=2, column=2, pady=2)
        
        # Botones de actualización
        ttk.Button(recursos_frame, text="Actualizar Listas", 
                  command=self.actualizar_listas).grid(row=3, column=0, columnspan=3, pady=5)
        
        recursos_frame.columnconfigure(1, weight=1)
        
        # ===== BOTÓN PRINCIPAL UNIFICADO =====
        acciones_frame = ttk.LabelFrame(left_frame, text="Acción Principal", padding="10")
        acciones_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(acciones_frame, text="🚀 EJECUTAR PIPELINE INTELIGENTE", 
                  command=self.ejecutar_pipeline_inteligente,
                  style='Accent.TButton').pack(fill=tk.X, pady=5)
        
        # Etiqueta de estado del pipeline
        self.estado_pipeline = ttk.Label(acciones_frame, text="Listo para ejecutar", 
                                        foreground='green', font=('Arial', 9))
        self.estado_pipeline.pack(fill=tk.X)
        
        # Frame de utilidades
        util_frame = ttk.LabelFrame(left_frame, text="Utilidades", padding="10")
        util_frame.pack(fill=tk.X, pady=(0, 10))
        
        botones_util = [
            ("Configuración del Sistema", self.abrir_configuracion),
            ("Ver Gráficos", self.mostrar_graficos),
            ("Limpiar Cache Modelos", self.limpiar_cache_modelos),
            ("Limpiar Cache Características", self.limpiar_cache_caracteristicas),
            ("Estadísticas de Cache", self.mostrar_estadisticas_cache)
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
        
        self.log("Sistema ANFIS inicializado. Seleccione una operación para comenzar.")
    
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
            self.combo_cache_entrenamiento['values'] = caches
            self.combo_cache_prueba['values'] = caches
            if caches:
                self.cache_entrenamiento_seleccionado.set(caches[0])
                self.cache_prueba_seleccionado.set(caches[0])
    
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
    
    def eliminar_cache_seleccionado(self, tipo):
        """Elimina cache de entrenamiento o prueba según el tipo"""
        if tipo == 'entrenamiento':
            cache = self.cache_entrenamiento_seleccionado.get()
        else:
            cache = self.cache_prueba_seleccionado.get()
            
        if cache and messagebox.askyesno("Confirmar", f"¿Eliminar el cache {cache}?"):
            if sistema_cache.eliminar_caracteristicas(cache):
                self.log(f"Cache eliminado: {cache}")
                self.actualizar_listas('caches')
            else:
                self.log(f"Error eliminando cache: {cache}")
    
    # ===== NUEVO SISTEMA DE PIPELINE INTELIGENTE =====
    
    def ejecutar_pipeline_inteligente(self):
        """Ejecuta el pipeline basado en las selecciones del usuario"""
        self.log("🤖 INICIANDO PIPELINE INTELIGENTE")
        self.log("=" * 50)
        
        # Determinar qué vamos a hacer
        config_accion = self.analizar_configuracion()
        self.log(f"📋 Acción detectada: {config_accion['tipo']}")
        
        threading.Thread(target=self._ejecutar_pipeline_inteligente, 
                        args=(config_accion,), daemon=True).start()
    
    def analizar_configuracion(self):
        """Analiza las selecciones y determina qué hacer"""
        config = {
            'tipo': '',
            'usar_modelo': self.usar_modelo.get(),
            'usar_cache_entrenamiento': self.usar_cache_entrenamiento.get(),
            'usar_cache_prueba': self.usar_cache_prueba.get(),
            'modelo_seleccionado': self.modelo_seleccionado.get(),
            'cache_entrenamiento': self.cache_entrenamiento_seleccionado.get(),
            'cache_prueba': self.cache_prueba_seleccionado.get(),
            'carpeta_entrenamiento': self.train_dir.get(),
            'carpeta_prueba': self.test_dir.get()
        }
        
        # Determinar el tipo de operación
        if config['usar_modelo']:
            config['tipo'] = 'EVALUAR_MODELO'
        else:
            if config['usar_cache_entrenamiento'] or config['usar_cache_prueba']:
                config['tipo'] = 'ENTRENAR_CON_CACHE'
            else:
                config['tipo'] = 'ENTRENAR_NUEVO'
        
        return config
    
    def _ejecutar_pipeline_inteligente(self, config_accion):
        """Ejecuta el pipeline basado en la configuración"""
        self.mostrar_progreso(True)
        try:
            # Actualizar rutas en configuración
            config.directorio_entrenamiento = self.train_dir.get()
            config.directorio_prueba = self.test_dir.get()
            config.guardar_configuracion()
            
            if config_accion['tipo'] == 'ENTRENAR_NUEVO':
                self._ejecutar_entrenamiento_nuevo(config_accion)
            elif config_accion['tipo'] == 'ENTRENAR_CON_CACHE':
                self._ejecutar_entrenamiento_cache(config_accion)
            elif config_accion['tipo'] == 'EVALUAR_MODELO':
                self._ejecutar_evaluacion_modelo(config_accion)
            else:
                self.log("❌ Configuración no válida")
                
        except Exception as e:
            self.log(f"❌ Error en pipeline: {str(e)}")
            messagebox.showerror("Error", f"Error durante la ejecución:\n{str(e)}")
        finally:
            self.mostrar_progreso(False)
            self.actualizar_listas()
    
    def _cargar_datos_inteligente(self, config_accion):
        """
        Carga datos de manera inteligente según la configuración
        Solo procesa las carpetas que realmente se necesitan
        """
        from features.procesamiento_image import process_all_images
        
        X_train, y_train, X_test, y_test = None, None, None, None
        
        # LÓGICA PARA ENTRENAMIENTO
        if config_accion['usar_cache_entrenamiento'] and config_accion['cache_entrenamiento']:
            self.log(f"📁 Cargando cache de entrenamiento: {config_accion['cache_entrenamiento']}")
            X_train, y_train = sistema_cache.cargar_caracteristicas_especificas(
                config_accion['cache_entrenamiento'])
            if X_train is None:
                self.log("❌ No se pudo cargar el cache de entrenamiento, procesando desde carpeta...")
                X_train, y_train = process_all_images(
                    base_dir=config_accion['carpeta_entrenamiento'],
                    use_cache=False
                )
        else:
            self.log(f"📁 Procesando imágenes de entrenamiento desde: {config_accion['carpeta_entrenamiento']}")
            X_train, y_train = process_all_images(
                base_dir=config_accion['carpeta_entrenamiento'],
                use_cache=not config_accion['usar_cache_entrenamiento']  # Usar cache solo si no estamos forzando procesamiento
            )
        
        # LÓGICA PARA PRUEBA
        if config_accion['usar_cache_prueba'] and config_accion['cache_prueba']:
            self.log(f"📁 Cargando cache de prueba: {config_accion['cache_prueba']}")
            X_test, y_test = sistema_cache.cargar_caracteristicas_especificas(
                config_accion['cache_prueba'])
            if X_test is None:
                self.log("❌ No se pudo cargar el cache de prueba, procesando desde carpeta...")
                X_test, y_test = process_all_images(
                    base_dir=config_accion['carpeta_prueba'],
                    use_cache=False
                )
        else:
            self.log(f"📁 Procesando imágenes de prueba desde: {config_accion['carpeta_prueba']}")
            X_test, y_test = process_all_images(
                base_dir=config_accion['carpeta_prueba'],
                use_cache=not config_accion['usar_cache_prueba']  # Usar cache solo si no estamos forzando procesamiento
            )
        
        return X_train, y_train, X_test, y_test
    
    def _ejecutar_entrenamiento_nuevo(self, config_accion):
        """Entrena un nuevo modelo procesando solo lo necesario"""
        self.log("🎯 ENTRENANDO NUEVO MODELO")
        
        # Cargar datos de manera inteligente
        X_train, y_train, X_test, y_test = self._cargar_datos_inteligente(config_accion)
        
        # Verificar que tenemos datos de entrenamiento
        if X_train is None or len(X_train) == 0:
            self.log("❌ Error: No hay datos de entrenamiento disponibles")
            return
        
        # Si no hay datos de prueba, usar los de entrenamiento para evaluación
        if X_test is None or len(X_test) == 0:
            self.log("⚠️  No hay datos de prueba, usando datos de entrenamiento para evaluación")
            X_test, y_test = X_train, y_train
        
        # Entrenar modelo
        from main import pipeline_completo
        resultado = pipeline_completo(
            use_cache=False,  # Ya cargamos los datos manualmente
            entrenar_nuevo=True,
            datos_entrenamiento=(X_train, y_train),
            datos_prueba=(X_test, y_test)
        )
        
        if resultado and 'evaluacion' in resultado:
            self._mostrar_resultados_evaluacion(resultado['evaluacion'])
        
        self.log("✅ Entrenamiento completado")
    
    def _ejecutar_entrenamiento_cache(self, config_accion):
        """Entrena modelo usando caché existente - procesando solo lo necesario"""
        self.log("🎯 ENTRENANDO NUEVO MODELO (con caché)")
        
        # Cargar datos de manera inteligente (misma lógica que arriba)
        X_train, y_train, X_test, y_test = self._cargar_datos_inteligente(config_accion)
        
        # Verificar que tenemos datos de entrenamiento
        if X_train is None or len(X_train) == 0:
            self.log("❌ Error: No hay datos de entrenamiento disponibles")
            return
        
        # Si no hay datos de prueba, usar los de entrenamiento para evaluación
        if X_test is None or len(X_test) == 0:
            self.log("⚠️  No hay datos de prueba, usando datos de entrenamiento para evaluación")
            X_test, y_test = X_train, y_train
        
        # Entrenar modelo
        from main import pipeline_completo
        resultado = pipeline_completo(
            use_cache=False,  # Ya cargamos los datos manualmente
            entrenar_nuevo=True,
            datos_entrenamiento=(X_train, y_train),
            datos_prueba=(X_test, y_test)
        )
        
        if resultado and 'evaluacion' in resultado:
            self._mostrar_resultados_evaluacion(resultado['evaluacion'])
        
        self.log("✅ Entrenamiento con caché completado")
    
    def _ejecutar_evaluacion_modelo(self, config_accion):
        """Evalúa un modelo existente - procesando solo lo necesario"""
        self.log("🎯 EVALUANDO MODELO EXISTENTE")
        
        if not config_accion['modelo_seleccionado']:
            raise ValueError("Debe seleccionar un modelo para evaluar")
        
        # Solo necesitamos datos de prueba para evaluación
        if config_accion['usar_cache_prueba'] and config_accion['cache_prueba']:
            self.log(f"📁 Cargando cache de prueba: {config_accion['cache_prueba']}")
            X_test, y_test = sistema_cache.cargar_caracteristicas_especificas(
                config_accion['cache_prueba'])
            if X_test is None:
                self.log("❌ No se pudo cargar el cache de prueba, procesando desde carpeta...")
                X_test, y_test = process_all_images(
                    base_dir=config_accion['carpeta_prueba'],
                    use_cache=False
                )
        else:
            self.log(f"📁 Procesando imágenes de prueba desde: {config_accion['carpeta_prueba']}")
            X_test, y_test = process_all_images(
                base_dir=config_accion['carpeta_prueba'],
                use_cache=not config_accion['usar_cache_prueba']
            )
        
        # Verificar que tenemos datos de prueba
        if X_test is None or len(X_test) == 0:
            self.log("❌ Error: No hay datos de prueba disponibles")
            return
        
        from main import usar_modelo_guardado
        resultado = usar_modelo_guardado(
            nombre_modelo=config_accion['modelo_seleccionado'],
            datos_prueba=(X_test, y_test)
        )
        
        if resultado:
            self._mostrar_resultados_evaluacion(resultado)
        
        self.log("✅ Evaluación completada")
    
    def _ejecutar_evaluacion_modelo(self, config_accion):
        """Evalúa un modelo existente"""
        self.log("🎯 EVALUANDO MODELO EXISTENTE")
        
        if not config_accion['modelo_seleccionado']:
            raise ValueError("Debe seleccionar un modelo para evaluar")
        
        self.log(f"🤖 Modelo: {config_accion['modelo_seleccionado']}")
        
        if config_accion['usar_cache_prueba']:
            self.log(f"📁 Cache prueba: {config_accion['cache_prueba']}")
        else:
            self.log(f"📁 Carpeta prueba: {config_accion['carpeta_prueba']}")
        
        resultado = usar_modelo_guardado(nombre_modelo=config_accion['modelo_seleccionado'])
        
        if resultado:
            self._mostrar_resultados_evaluacion(resultado)
        
        self.log("✅ Evaluación completada")
    
    def _mostrar_resultados_evaluacion(self, evaluacion):
        """Muestra los resultados de la evaluación"""
        if 'metricas' in evaluacion:
            metricas = evaluacion['metricas']
            self.log("📊 Métricas obtenidas:")
            self.log(f"  - Precisión: {metricas['precision']:.4f}")
            self.log(f"  - Sensibilidad: {metricas['sensitivity']:.4f}")
            self.log(f"  - Especificidad: {metricas['specificity']:.4f}")
            self.log(f"  - F1-Score: {metricas['f1_score']:.4f}")
            if metricas.get('auc', 0) > 0:
                self.log(f"  - AUC-ROC: {metricas['auc']:.4f}")
    
    # ===== FUNCIONES DE UTILIDAD (mantenidas) =====
    
    def abrir_configuracion(self):
        self.log("Abriendo configurador del sistema...")
        try:
            from interfaz.configurador import Configurador
            ventana_config = tk.Toplevel(self.root)
            Configurador(ventana_config)
        except Exception as e:
            self.log(f"Error al abrir configurador: {str(e)}")
    
    def mostrar_graficos(self):
        self.log("Abriendo visualizador de gráficos...")
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
        self.log("Estadísticas de cache:")
        for tipo, datos in stats.items():
            self.log(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")
    
    def log(self, mensaje):
        """Agregar mensaje al area de texto - MEJORADA"""
        # Limpiar mensajes de progreso repetitivos
        if "Imagen" in mensaje and "procesando" in mensaje.lower():
            # Actualizar última línea en lugar de agregar nueva
            lines = self.texto_resultados.get("1.0", tk.END).split('\n')
            if len(lines) > 1 and "Imagen" in lines[-2]:
                # Reemplazar última línea
                self.texto_resultados.delete("end-2l", "end-1l")
        
        # Agregar timestamp opcional
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        mensaje_formateado = f"[{timestamp}] {mensaje}\n"
        
        self.texto_resultados.insert(tk.END, mensaje_formateado)
        self.texto_resultados.see(tk.END)
        self.estado.config(text=mensaje[:50] + "..." if len(mensaje) > 50 else mensaje)
        self.root.update_idletasks()
    
    def restaurar_output_original(self):
        """Restaura la salida estándar al cerrar la ventana"""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def __del__(self):
        self.restaurar_output_original()
    
    def mostrar_progreso(self, activo):
        """Mostrar u ocultar barra de progreso"""
        if activo:
            self.progress.start()
            self.estado.config(text="Procesando...", foreground='orange')
            self.estado_pipeline.config(text="Ejecutando pipeline...", foreground='orange')
        else:
            self.progress.stop()
            self.estado.config(text="Listo", foreground='green')
            self.estado_pipeline.config(text="Listo para ejecutar", foreground='green')

def main():
    root = tk.Tk()
    app = VentanaPrincipal(root)
    root.mainloop()

if __name__ == "__main__":
    main()