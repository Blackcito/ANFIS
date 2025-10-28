# interfaz/ventana_principal_ttkbootstrap.py
# Versión completa de la interfaz usando ttkbootstrap (tema: morph)
# Mantiene la lógica original (métodos, nombres y flujo) y solo actualiza la UI.

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
from datetime import datetime
import threading
import os
import sys
import io
from pathlib import Path
import shutil

# Intento de import original (igual que tu versión)
try:
    from config.configuracion import config
    from config.rutas import sistema_rutas
    from core.pipeline_anfis import pipeline_anfis
    from utils.cache import sistema_cache

    rutas_info = sistema_rutas.obtener_rutas_importantes()
    print(f" Sistema inicializado en modo: {rutas_info['modo']}")
    print(f" Directorio base: {rutas_info['base']}")
    print(f" Directorio persistente: {rutas_info['persistente']}")

except ImportError as e:
    print(f" Error de importación: {e}")
    print(f"sys.path: {sys.path}")
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent.parent

    rutas_emergencia = [
        str(base_dir / 'config'),
        str(base_dir / 'core'),
        str(base_dir / 'utils'),
        str(base_dir)
    ]

    for ruta in rutas_emergencia:
        if ruta not in sys.path:
            sys.path.insert(0, ruta)

    try:
        from config.rutas import sistema_rutas
        from core.pipeline_anfis import pipeline_anfis
        from config.configuracion import config
        from utils.cache import sistema_cache
        print(" Imports recuperados en modo emergencia")
    except ImportError as e2:
        print(f" Error crítico de importación: {e2}")
        raise


class OutputRedirector(io.StringIO):
    def __init__(self, ventana):
        super().__init__()
        self.ventana = ventana

    def write(self, text):
        if text.strip():
            # Mantener newline handling
            for line in text.rstrip().splitlines():
                self.ventana.log(line)
        return len(text)

    def flush(self):
        pass


class VentanaPrincipal:
    def __init__(self, root):
        # root es ttk.Window de ttkbootstrap
        self.root = root
        self.root.title("ANFIS - Detección de Tumores Cerebrales")
        self.root.geometry("1400x900")
        # Variables de rutas inicializadas desde config (si existe)
        self.train_tumor_dir = tk.StringVar(value=getattr(config, 'directorio_entrenamiento_tumor', ''))
        self.train_notumor_dir = tk.StringVar(value=getattr(config, 'directorio_entrenamiento_notumor', ''))
        self.test_tumor_dir = tk.StringVar(value=getattr(config, 'directorio_prueba_tumor', ''))
        self.test_notumor_dir = tk.StringVar(value=getattr(config, 'directorio_prueba_notumor', ''))

        # Variables opciones principales
        self.usar_modelo_existente = tk.BooleanVar(value=False)
        self.usar_cache_entrenamiento_existente = tk.BooleanVar(value=False)
        self.usar_cache_prueba_existente = tk.BooleanVar(value=False)

        self.guardar_graficos = tk.BooleanVar(value=getattr(config.analisis, 'guardar_graficos_analisis', False))
        self.visualizar_graficos = tk.BooleanVar(value=True)

        self.modelo_seleccionado = tk.StringVar()
        self.cache_entrenamiento_seleccionado = tk.StringVar()
        self.cache_prueba_seleccionado = tk.StringVar()

        # Redirección de output
        self.output_redirector = OutputRedirector(self)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector

        # crear interfaz
        self.crear_interfaz()
        # actualizar listas (usa sistema_cache)
        try:
            self.actualizar_listas()
        except Exception:
            pass

        # bind combobox events (se crean en crear_interfaz)
        try:
            self.combo_modelos.bind('<<ComboboxSelected>>', self._guardar_seleccion_modelo)
            self.combo_cache_train.bind('<<ComboboxSelected>>', self._guardar_seleccion_cache_train)
            self.combo_cache_test.bind('<<ComboboxSelected>>', self._guardar_seleccion_cache_test)
        except Exception:
            pass

        self.log("Sistema ANFIS inicializado. Configure las opciones y ejecute el pipeline.")

    # ---- eventos de guardado de selección ----
    def _guardar_seleccion_modelo(self, event):
        seleccion = self.combo_modelos.get()
        if seleccion:
            self.log(f" Modelo seleccionado: {seleccion}")

    def _guardar_seleccion_cache_train(self, event):
        seleccion = self.combo_cache_train.get()
        if seleccion:
            self.log(f" Cache entrenamiento seleccionado: {seleccion}")

    def _guardar_seleccion_cache_test(self, event):
        seleccion = self.combo_cache_test.get()
        if seleccion:
            self.log(f" Cache prueba seleccionado: {seleccion}")

    # ---- construcción UI ----
    def crear_interfaz(self):
        # Estilos y tema ya definidos por root = ttk.Window(themename='morph')
        self.configurar_estilos()

        # Layout principal: una columna izquierda (pestañas) y derecha para contenido (se mantiene minimal),
        # pero logs ahora en panel inferior (footer)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=12, pady=10)

        # columnas: left controls + right canvas (opcionales). Mantendremos layout con left controls y espacio para futuro contenido.
        left_col = ttk.Frame(main_frame)
        left_col.pack(side=LEFT, fill=Y, padx=(0,10))

        center_col = ttk.Frame(main_frame)
        center_col.pack(side=LEFT, fill=BOTH, expand=True)

        # ---------- Left column: Notebook con pestañas ----------
        notebook = ttk.Notebook(left_col, bootstyle="primary")
        notebook.pack(fill=Y, expand=False, ipadx=10, ipady=10)

        # Pestaña Datos
        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text="Datos")

        # Pestaña Ejecución
        tab_ejecucion = ttk.Frame(notebook)
        notebook.add(tab_ejecucion, text="Ejecución")

        # Pestaña Herramientas
        tab_utils = ttk.Frame(notebook)
        notebook.add(tab_utils, text="Herramientas")

        # ---------- Tab Datos (configuración de rutas) ----------
        config_frame = ttk.Labelframe(tab_datos, text="Configuración de Datos", padding=10)
        config_frame.pack(fill=BOTH, expand=False, padx=6, pady=6)

        # Training subframe
        train_subframe = ttk.Frame(config_frame)
        train_subframe.pack(fill=X, pady=4)
        ttk.Label(train_subframe, text="Carpeta Tumor (Entrenamiento):").grid(row=0, column=0, sticky=W, pady=2)
        ttk.Entry(train_subframe, textvariable=self.train_tumor_dir, width=40).grid(row=0, column=1, padx=6)
        ttk.Button(train_subframe, text="Seleccionar", command=lambda: self.seleccionar_directorio(self.train_tumor_dir, "tumor (entrenamiento)"), bootstyle="secondary-outline").grid(row=0, column=2)

        ttk.Label(train_subframe, text="Carpeta No-Tumor (Entrenamiento):").grid(row=1, column=0, sticky=W, pady=2)
        ttk.Entry(train_subframe, textvariable=self.train_notumor_dir, width=40).grid(row=1, column=1, padx=6)
        ttk.Button(train_subframe, text="Seleccionar", command=lambda: self.seleccionar_directorio(self.train_notumor_dir, "no-tumor (entrenamiento)"), bootstyle="secondary-outline").grid(row=1, column=2)

        # Test subframe
        test_subframe = ttk.Frame(config_frame)
        test_subframe.pack(fill=X, pady=6)
        ttk.Label(test_subframe, text="Carpeta Tumor (Prueba):").grid(row=0, column=0, sticky=W, pady=2)
        ttk.Entry(test_subframe, textvariable=self.test_tumor_dir, width=40).grid(row=0, column=1, padx=6)
        ttk.Button(test_subframe, text="Seleccionar", command=lambda: self.seleccionar_directorio(self.test_tumor_dir, "tumor (prueba)"), bootstyle="secondary-outline").grid(row=0, column=2)

        ttk.Label(test_subframe, text="Carpeta No-Tumor (Prueba):").grid(row=1, column=0, sticky=W, pady=2)
        ttk.Entry(test_subframe, textvariable=self.test_notumor_dir, width=40).grid(row=1, column=1, padx=6)
        ttk.Button(test_subframe, text="Seleccionar", command=lambda: self.seleccionar_directorio(self.test_notumor_dir, "no-tumor (prueba)"), bootstyle="secondary-outline").grid(row=1, column=2)

        # ---------- Tab Ejecución (opciones y acción principal) ----------
        opciones_frame = ttk.Labelframe(tab_ejecucion, text="Opciones de Ejecución", padding=10)
        opciones_frame.pack(fill=BOTH, expand=False, padx=6, pady=6)

        # modelo existente
        modelo_frame = ttk.Frame(opciones_frame)
        modelo_frame.pack(fill=X, pady=4)
        ttk.Checkbutton(modelo_frame, text="Usar Modelo Existente", variable=self.usar_modelo_existente, command=self.actualizar_interfaz_opciones, bootstyle="round-toggle").pack(side=LEFT)
        self.combo_modelos = ttk.Combobox(modelo_frame, textvariable=self.modelo_seleccionado, width=28, state="disabled")
        self.combo_modelos.pack(side=RIGHT)

        # cache entrenamiento
        cache_train_frame = ttk.Frame(opciones_frame)
        cache_train_frame.pack(fill=X, pady=4)
        ttk.Checkbutton(cache_train_frame, text="Usar Cache Entrenamiento Existente", variable=self.usar_cache_entrenamiento_existente, command=self.actualizar_interfaz_opciones, bootstyle="round-toggle").pack(side=LEFT)
        self.combo_cache_train = ttk.Combobox(cache_train_frame, textvariable=self.cache_entrenamiento_seleccionado, width=28, state="disabled")
        self.combo_cache_train.pack(side=RIGHT)

        # cache prueba
        cache_test_frame = ttk.Frame(opciones_frame)
        cache_test_frame.pack(fill=X, pady=4)
        ttk.Checkbutton(cache_test_frame, text="Usar Cache Prueba Existente", variable=self.usar_cache_prueba_existente, command=self.actualizar_interfaz_opciones, bootstyle="round-toggle").pack(side=LEFT)
        self.combo_cache_test = ttk.Combobox(cache_test_frame, textvariable=self.cache_prueba_seleccionado, width=28, state="disabled")
        self.combo_cache_test.pack(side=RIGHT)

        # visualizar graficos
        ttk.Checkbutton(opciones_frame, text="Visualizar Gráficos (mostrar ventanas)", variable=self.visualizar_graficos, bootstyle="round-toggle").pack(anchor=W, pady=(6,0))

        # boton actualizar / reprocesar
        gestion_frame = ttk.Frame(opciones_frame)
        gestion_frame.pack(fill=X, pady=(8,0))
        ttk.Button(gestion_frame, text="Actualizar Listas", command=self.actualizar_listas, bootstyle="primary-outline").pack(side=LEFT, padx=(0,4))
        ttk.Button(gestion_frame, text="Forzar Reprocesamiento", command=self.ejecutar_sin_cache, bootstyle="danger-outline").pack(side=LEFT)

        # accion principal destacada
        acciones_frame = ttk.Frame(tab_ejecucion)
        acciones_frame.pack(fill=X, padx=6, pady=(8,6))
        ttk.Button(acciones_frame, text="EJECUTAR PIPELINE", command=self.ejecutar_pipeline, style='Accent.TButton').pack(fill=tk.X, pady=5)
        self.estado_pipeline = ttk.Label(acciones_frame, text="Listo para ejecutar", bootstyle="success")
        self.estado_pipeline.pack()

        # ---------- Tab Herramientas ----------
        util_frame = ttk.Labelframe(tab_utils, text="Utilidades", padding=10)
        util_frame.pack(fill=BOTH, expand=False, padx=6, pady=6)

        botones_util = [
            ("Configuración del Sistema", self.abrir_configuracion),
            ("Ver Gráficos", self.mostrar_graficos),
            ("Limpiar Cache Modelos", self.limpiar_cache_modelos),
            ("Limpiar Cache Características", self.limpiar_cache_caracteristicas),
            ("Limpiar Cache Resultados", self.limpiar_cache_resultados),
            ("Estadísticas de Cache", self.mostrar_estadisticas_cache),
            ("Eliminar todo el cache", self.limpiar_todo_cache),
            ("Eliminar imágenes intermedias", self.limpiar_imagenes_intermedias)
        ]

        for i, (texto, comando) in enumerate(botones_util):
            btn = ttk.Button(util_frame, text=texto, command=comando, bootstyle="secondary")
            btn.grid(row=i//2, column=i%2, padx=6, pady=6, sticky="ew")

        util_frame.columnconfigure(0, weight=1)
        util_frame.columnconfigure(1, weight=1)

        # ---------- Center column: resultados / visualización ----------
        resultados_frame = ttk.Labelframe(center_col, text="Resultados y Logs", padding=10)
        resultados_frame.pack(fill=BOTH, expand=True)

        # ScrolledText para logs (mantener monoespaciado)
        self.texto_resultados = scrolledtext.ScrolledText(resultados_frame, width=90, height=30, font=('Consolas', 10))
        self.texto_resultados.pack(fill=BOTH, expand=True)

        # Colorear tags básicos (info / error)
        try:
            self.texto_resultados.tag_config("error", foreground="red")
            self.texto_resultados.tag_config("info", foreground="#00bfff")
            self.texto_resultados.tag_config("warning", foreground="orange")
        except Exception:
            pass

        # ---------- Footer: barra de progreso y estado ----------
        footer = ttk.Frame(self.root)
        footer.pack(fill=X, side=BOTTOM, padx=6, pady=6)

        self.progress = ttk.Progressbar(footer, bootstyle="info", mode='indeterminate')
        self.progress.pack(fill=X, side=LEFT, expand=True, padx=(0,10))

        self.estado = ttk.Label(footer, text="Listo", bootstyle="success")
        self.estado.pack(side=RIGHT)

        # Inicializar estado UI
        self.actualizar_interfaz_opciones()

    def configurar_estilos(self):
        style = ttk.Style()
        # Bootstyle ya aplica tema morph; ajustar fuentes y tamaños
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TEntry', padding=4)

    # ---- lógica de UI ----
    def actualizar_interfaz_opciones(self):
        if self.usar_modelo_existente.get():
            self.combo_modelos.config(state="readonly")
            try:
                self.actualizar_listas()
            except Exception:
                pass
        else:
            self.combo_modelos.config(state="disabled")
            self.modelo_seleccionado.set("")

        if self.usar_cache_entrenamiento_existente.get():
            self.combo_cache_train.config(state="readonly")
            try:
                self.actualizar_listas()
            except Exception:
                pass
        else:
            self.combo_cache_train.config(state="disabled")
            self.cache_entrenamiento_seleccionado.set("")

        if self.usar_cache_prueba_existente.get():
            self.combo_cache_test.config(state="readonly")
            try:
                self.actualizar_listas()
            except Exception:
                pass
        else:
            self.combo_cache_test.config(state="disabled")
            self.cache_prueba_seleccionado.set("")

    def actualizar_listas(self):
        seleccion_modelo_actual = self.modelo_seleccionado.get()
        seleccion_cache_train_actual = self.cache_entrenamiento_seleccionado.get()
        seleccion_cache_test_actual = self.cache_prueba_seleccionado.get()

        modelos = sistema_cache.listar_modelos()
        self.combo_modelos['values'] = modelos

        caches = sistema_cache.listar_caracteristicas()
        self.combo_cache_train['values'] = caches
        self.combo_cache_test['values'] = caches

        if modelos:
            if seleccion_modelo_actual in modelos and self.usar_modelo_existente.get():
                self.modelo_seleccionado.set(seleccion_modelo_actual)
            elif self.usar_modelo_existente.get():
                self.modelo_seleccionado.set(modelos[0])
            else:
                self.modelo_seleccionado.set("")

        if caches:
            if seleccion_cache_train_actual in caches and self.usar_cache_entrenamiento_existente.get():
                self.cache_entrenamiento_seleccionado.set(seleccion_cache_train_actual)
            elif self.usar_cache_entrenamiento_existente.get():
                self.cache_entrenamiento_seleccionado.set(caches[0])
            else:
                self.cache_entrenamiento_seleccionado.set("")

            if seleccion_cache_test_actual in caches and self.usar_cache_prueba_existente.get():
                self.cache_prueba_seleccionado.set(seleccion_cache_test_actual)
            elif self.usar_cache_prueba_existente.get():
                self.cache_prueba_seleccionado.set(caches[0])
            else:
                self.cache_prueba_seleccionado.set("")

    def seleccionar_directorio(self, var, tipo):
        directorio = filedialog.askdirectory(title=f"Seleccionar carpeta {tipo}")
        if directorio:
            var.set(directorio)
            self.log(f"Carpeta {tipo} seleccionada: {directorio}")

    # ---- ejecución pipeline (misma lógica) ----
    def ejecutar_pipeline(self):
        self.log(" INICIANDO PIPELINE ANFIS")
        self.log("=" * 50)

        opciones = {
            'entrenar_nuevo': not self.usar_modelo_existente.get(),
            'nombre_modelo': self.modelo_seleccionado.get() if self.usar_modelo_existente.get() else None,
            'usar_cache_entrenamiento': self.usar_cache_entrenamiento_existente.get(),
            'usar_cache_prueba': self.usar_cache_prueba_existente.get(),
            'cache_entrenamiento_especifico': self.cache_entrenamiento_seleccionado.get() if self.usar_cache_entrenamiento_existente.get() else None,
            'cache_prueba_especifico': self.cache_prueba_seleccionado.get() if self.usar_cache_prueba_existente.get() else None,
            'guardar_graficos': getattr(config.analisis, 'guardar_graficos_analisis', self.guardar_graficos.get()),
            'visualizar_graficos': self.visualizar_graficos.get(),
            'forzar_reprocesamiento': False,
        }

        self.mostrar_configuracion_ejecucion(opciones)

        if not self.validar_configuracion(opciones):
            return

        threading.Thread(target=self._ejecutar_pipeline, args=(opciones,), daemon=True).start()

    def _ejecutar_pipeline(self, opciones):
        self.mostrar_progreso(True)
        try:
            opciones.update({
                'train_tumor_dir': self.train_tumor_dir.get(),
                'train_notumor_dir': self.train_notumor_dir.get(),
                'test_tumor_dir': self.test_tumor_dir.get(),
                'test_notumor_dir': self.test_notumor_dir.get()
            })

            resultado = pipeline_anfis.ejecutar(**opciones)

            if resultado:
                self.log(" Pipeline ejecutado exitosamente")
                if 'evaluacion' in resultado:
                    self.mostrar_resultados_evaluacion(resultado['evaluacion'])
            else:
                self.log(" El pipeline no produjo resultados")

        except Exception as e:
            self.log(f" Error en pipeline: {str(e)}")
            messagebox.showerror("Error", f"Error durante la ejecución:\n{str(e)}")
        finally:
            self.mostrar_progreso(False)
            try:
                self.actualizar_listas()
            except Exception:
                pass

    def mostrar_configuracion_ejecucion(self, opciones):
        self.log(" CONFIGURACIÓN DE EJECUCIÓN:")
        self.log(f"   - Entrenar nuevo modelo: {opciones['entrenar_nuevo']}")
        self.log(f"   - Usar cache entrenamiento existente: {opciones['usar_cache_entrenamiento']}")
        self.log(f"   - Usar cache prueba existente: {opciones['usar_cache_prueba']}")
        self.log(f"   - Guardar gráficos de análisis: {getattr(config.analisis, 'guardar_graficos_analisis', False)}")

        if opciones['nombre_modelo']:
            self.log(f"   - Modelo seleccionado: {opciones['nombre_modelo']}")
        if self.usar_cache_entrenamiento_existente.get() and self.cache_entrenamiento_seleccionado.get():
            self.log(f"   - Cache entrenamiento: {self.cache_entrenamiento_seleccionado.get()}")
        if self.usar_cache_prueba_existente.get() and self.cache_prueba_seleccionado.get():
            self.log(f"   - Cache prueba: {self.cache_prueba_seleccionado.get()}")

        self.log(" CONFIGURACIÓN DE GUARDADO (desde Configuración del Sistema):")
        self.log(f"   - Guardar cache características: {config.cache.guardar_cache_caracteristicas}")
        self.log(f"   - Guardar cache modelos: {config.cache.guardar_cache_modelos}")
        self.log(f"   - Guardar cache gráficos: {config.cache.guardar_cache_graficos}")
        self.log(f"   - Guardar cache métricas: {config.cache.guardar_cache_metricas}")
        self.log(f"   - Guardar métricas análisis: {config.analisis.guardar_metricas}")
        self.log(f"   - Guardar reportes análisis: {config.analisis.guardar_reportes}")

    def validar_configuracion(self, opciones):
        if opciones['entrenar_nuevo']:
            train_tumor = Path(self.train_tumor_dir.get())
            train_notumor = Path(self.train_notumor_dir.get())

            if not train_tumor.exists():
                self.log(" Error: No existe la carpeta de tumor (entrenamiento)")
                messagebox.showerror("Error", f"No existe la carpeta: {train_tumor}")
                return False

            if not train_notumor.exists():
                self.log(" Error: No existe la carpeta de no-tumor (entrenamiento)")
                messagebox.showerror("Error", f"No existe la carpeta: {train_notumor}")
                return False

        test_tumor = Path(self.test_tumor_dir.get())
        test_notumor = Path(self.test_notumor_dir.get())

        if not test_tumor.exists():
            self.log(" Error: No existe la carpeta de tumor (prueba)")
            messagebox.showerror("Error", f"No existe la carpeta: {test_tumor}")
            return False

        if not test_notumor.exists():
            self.log(" Error: No existe la carpeta de no-tumor (prueba)")
            messagebox.showerror("Error", f"No existe la carpeta: {test_notumor}")
            return False

        if not opciones['entrenar_nuevo'] and not opciones['nombre_modelo']:
            self.log(" Error: Debe seleccionar un modelo para evaluación")
            messagebox.showerror("Error", "Debe seleccionar un modelo para evaluación")
            return False

        if opciones['usar_cache_entrenamiento'] and self.cache_entrenamiento_seleccionado.get():
            cache_path = sistema_rutas.cache_dir / "caracteristicas" / self.cache_entrenamiento_seleccionado.get()
            if not cache_path.exists():
                self.log(f" Error: No existe el cache de entrenamiento seleccionado")
                messagebox.showerror("Error", f"No existe el cache de entrenamiento: {self.cache_entrenamiento_seleccionado.get()}")
                return False

        if opciones['usar_cache_prueba'] and self.cache_prueba_seleccionado.get():
            cache_path = sistema_rutas.cache_dir / "caracteristicas" / self.cache_prueba_seleccionado.get()
            if not cache_path.exists():
                self.log(f" Error: No existe el cache de prueba seleccionado")
                messagebox.showerror("Error", f"No existe el cache de prueba: {self.cache_prueba_seleccionado.get()}")
                return False

        return True

    def ejecutar_sin_cache(self):
        self.log(" Ejecutando con reprocesamiento forzado...")
        opciones = {
            'entrenar_nuevo': True,
            'usar_cache_entrenamiento': False,
            'usar_cache_prueba': False,
            'forzar_reprocesamiento': True
        }
        threading.Thread(target=self._ejecutar_pipeline, args=(opciones,), daemon=True).start()

    def mostrar_resultados_evaluacion(self, evaluacion):
        if 'metricas' in evaluacion:
            metricas = evaluacion['metricas'].get('clasificacion', {})
            self.log(" RESULTADOS OBTENIDOS:")
            self.log("=" * 30)
            self.log(f"Precisión: {metricas.get('precision', 0):.4f}")
            self.log(f"Sensibilidad: {metricas.get('sensitivity', 0):.4f}")
            self.log(f"Especificidad: {metricas.get('specificity', 0):.4f}")
            self.log(f"F1-Score: {metricas.get('f1_score', 0):.4f}")
            if metricas.get('auc', 0) > 0:
                self.log(f"AUC-ROC: {metricas.get('auc', 0):.4f}")

    # UTILIDADES (mismas funciones)
    def abrir_configuracion(self):
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
            self.actualizar_listas()

    def limpiar_cache_caracteristicas(self):
        if messagebox.askyesno("Confirmar", "¿Está seguro de limpiar todo el cache de características?"):
            sistema_cache.limpiar_cache_caracteristicas()
            self.log("Cache de características limpiado")
            self.actualizar_listas()

    def limpiar_cache_resultados(self):
        if messagebox.askyesno("Confirmar", "¿Está seguro de limpiar todo el cache de resultados?"):
            sistema_cache.limpiar_cache_resultados()
            self.log("Cache de características limpiado")
            self.actualizar_listas()

    def limpiar_todo_cache(self):
        """Elimina todo el cache (modelos, características y resultados) con confirmación."""
        if not messagebox.askyesno("Confirmar", "¿Eliminar todo el caché (modelos, características y resultados)? Esto no se puede deshacer."):
            return
        try:
            self.log("Iniciando limpieza completa del caché...")
            sistema_cache.limpiar_cache_modelos()
            sistema_cache.limpiar_cache_caracteristicas()
            sistema_cache.limpiar_cache_resultados()
            self.log("Limpieza completa del caché finalizada.")
            try:
                self.actualizar_listas()
            except Exception:
                pass
            messagebox.showinfo("Completado", "Se ha eliminado todo el caché correctamente.")
        except Exception as e:
            self.log(f"Error limpiando todo el caché: {e}")
            messagebox.showerror("Error", f"No se pudo limpiar todo el caché:\n{e}")

    def limpiar_imagenes_intermedias(self):
        """Elimina las imágenes intermedias usadas durante el procesamiento.

        Usa la ruta configurada en config.procesamiento.directorio_imagenes_intermedias.
        Se pide confirmación y se recrea el directorio vacío si existía.
        """
        ruta_str = getattr(config.procesamiento, 'directorio_imagenes_intermedias', None)
        if not ruta_str:
            self.log("No hay una ruta configurada para imágenes intermedias en la configuración.")
            messagebox.showwarning("Aviso", "No hay ruta configurada para imágenes intermedias.")
            return

        ruta = Path(ruta_str)
        if not ruta.exists():
            self.log(f"La carpeta de imágenes intermedias no existe: {ruta}")
            messagebox.showinfo("Info", "La carpeta de imágenes intermedias no existe.")
            return

        if not messagebox.askyesno("Confirmar", f"¿Eliminar todas las imágenes intermedias en:\n{ruta}? Esta acción eliminará archivos y subcarpetas."):
            return

        try:
            # Mover/Eliminar de forma segura
            self.log(f"Eliminando imágenes intermedias en: {ruta}")
            # Usar shutil.rmtree para eliminar contenido y recrear carpeta vacía
            shutil.rmtree(ruta)
            ruta.mkdir(parents=True, exist_ok=True)
            self.log("Imágenes intermedias eliminadas correctamente.")
            messagebox.showinfo("Completado", "Imágenes intermedias eliminadas correctamente.")
        except Exception as e:
            self.log(f"Error eliminando imágenes intermedias: {e}")
            messagebox.showerror("Error", f"No se pudieron eliminar las imágenes intermedias:\n{e}")

    def mostrar_estadisticas_cache(self):
        stats = sistema_cache.obtener_estadisticas_cache()
        self.log(" ESTADÍSTICAS DE CACHÉ:")
        for tipo, datos in stats.items():
            self.log(f"  {tipo}: {datos['archivos']} archivos ({datos['tamaño_mb']} MB)")

    def log(self, mensaje):
        timestamp = datetime.now().strftime("%H:%M:%S")
        mensaje_formateado = f"[{timestamp}] {mensaje}\n"
        try:
            self.texto_resultados.insert(tk.END, mensaje_formateado)
            self.texto_resultados.see(tk.END)
            # resaltar si contiene keywords
            lower = mensaje.lower()
            if 'error' in lower or 'exception' in lower:
                self.texto_resultados.tag_add("error", f"end-2l linestart", "end-1l")
            elif 'proces' in lower or 'progreso' in lower:
                self.texto_resultados.tag_add("info", f"end-2l linestart", "end-1l")
        except Exception:
            pass
        try:
            self.estado.config(text=mensaje[:50] + "..." if len(mensaje) > 50 else mensaje)
        except Exception:
            pass
        self.root.update_idletasks()

    def mostrar_progreso(self, activo):
        if activo:
            self.progress.start()
            try:
                self.estado.config(text="Procesando...", bootstyle="warning")
                self.estado_pipeline.config(text="Ejecutando pipeline...", bootstyle="warning")
            except Exception:
                self.estado.config(text="Procesando...")
                self.estado_pipeline.config(text="Ejecutando pipeline...")
        else:
            self.progress.stop()
            try:
                self.estado.config(text="Listo", bootstyle="success")
                self.estado_pipeline.config(text="Listo para ejecutar", bootstyle="success")
            except Exception:
                self.estado.config(text="Listo")
                self.estado_pipeline.config(text="Listo para ejecutar")

    def restaurar_output_original(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def __del__(self):
        try:
            self.restaurar_output_original()
        except Exception:
            pass


def main():
    # Crear ventana con tema 'morph'
    root = ttk.Window(themename="morph")
    app = VentanaPrincipal(root)

    def on_closing():
        app.restaurar_output_original()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
