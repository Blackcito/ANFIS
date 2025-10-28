# interfaz/visualizador_graficos.py - VERSIÓN MEJORADA (SOPORTA TXT Y MÁS)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import matplotlib
# Configurar backend antes de importar pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import glob
import sys
from pathlib import Path
import numpy as np
import json
import pandas as pd

#  USAR SISTEMA CENTRALIZADO DE RUTAS
try:
    from config.rutas import sistema_rutas
    from config.configuracion import config
except ImportError as e:
    print(f" Error de importación: {e}")
    # Configuración de emergencia
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent.parent
    
    # Reintentar imports con rutas de emergencia
    sys.path.insert(0, str(base_dir / 'core'))
    sys.path.insert(0, str(base_dir / 'config'))
    try:
        from config.rutas import sistema_rutas
        from config.configuracion import config
    except ImportError:
        print(" Error crítico: No se pudo cargar el sistema de rutas")
        sistema_rutas = None
        config = None

class VentanaGraficos:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Resultados - ANFIS")
        self.root.geometry("1200x800")
        
        # Variables para tipos de archivo a mostrar
        self.show_imagenes = tk.BooleanVar(value=True)
        self.show_textos = tk.BooleanVar(value=True)
        self.show_datos = tk.BooleanVar(value=True)
        
        # Flags para controlar las fuentes que se buscarán
        self.show_cache_graficos = tk.BooleanVar(value=True)
        self.show_cache_reportes = tk.BooleanVar(value=True)
        self.show_cache_metricas = tk.BooleanVar(value=True)
        self.show_analisis = tk.BooleanVar(value=True)
        self.show_imagenes_intermedias = tk.BooleanVar(value=True)

        # Sub-checks para carpetas internas dentro de directorio_imagenes_intermedias
        self.show_img_original = tk.BooleanVar(value=True)
        self.show_img_enhanced = tk.BooleanVar(value=True)
        self.show_img_blurred = tk.BooleanVar(value=True)
        self.show_img_mask = tk.BooleanVar(value=True)
        self.show_img_roi = tk.BooleanVar(value=True)

        self.archivos_disponibles = []
        self.crear_interfaz()
        self.actualizar_lista()
    
    def obtener_rutas_busqueda(self):
        """Obtiene las rutas donde buscar archivos usando sistema centralizado"""
        rutas_busqueda = []
        
        if not sistema_rutas:
            print(" Sistema de rutas no disponible")
            return [os.getcwd()]
        
        # 1. Directorio de cache persistente para gráficos
        try:
            cache_graficos = sistema_rutas.cache_dir / "resultados" / "graficos"
            if self.show_cache_graficos.get() and cache_graficos.exists():
                rutas_busqueda.append(str(cache_graficos))
        except Exception:
            pass

        # 2. Directorio de cache para reportes
        try:
            cache_reportes = sistema_rutas.cache_dir / "resultados" / "reportes"
            if self.show_cache_reportes.get() and cache_reportes.exists():
                rutas_busqueda.append(str(cache_reportes))
        except Exception:
            pass

        # 3. Directorio de cache para métricas
        try:
            cache_metricas = sistema_rutas.cache_dir / "resultados" / "metricas"
            if self.show_cache_metricas.get() and cache_metricas.exists():
                rutas_busqueda.append(str(cache_metricas))
        except Exception:
            pass

        # 4. Directorio de cache para datos de reglas
        try:
            cache_datos_reglas = sistema_rutas.cache_dir / "resultados" / "datos_reglas"
            if cache_datos_reglas.exists():
                rutas_busqueda.append(str(cache_datos_reglas))
        except Exception:
            pass

        # 5. Directorio de análisis persistente
        try:
            if self.show_analisis.get() and config and hasattr(config.analisis, 'directorio_analisis'):
                analisis_dir = Path(config.analisis.directorio_analisis)
                if analisis_dir.exists():
                    rutas_busqueda.append(str(analisis_dir))
        except Exception:
            pass

        # 6. Directorio de imágenes intermedias
        try:
            if self.show_imagenes_intermedias.get() and config and hasattr(config.procesamiento, 'directorio_imagenes_intermedias'):
                imagenes_dir = Path(config.procesamiento.directorio_imagenes_intermedias)
                subdirs = [
                    (self.show_img_original, 'original'),
                    (self.show_img_enhanced, 'enhanced'),
                    (self.show_img_blurred, 'blurred'),
                    (self.show_img_mask, 'mask'),
                    (self.show_img_roi, 'roi'),
                ]

                for var, sub in subdirs:
                    try:
                        if not var.get():
                            continue
                        ruta_sub = imagenes_dir / sub
                        if ruta_sub.exists():
                            rutas_busqueda.append(str(ruta_sub))
                            continue
                        # Fallback: buscar subcarpetas por nombre
                        for child in imagenes_dir.iterdir():
                            try:
                                if child.is_dir() and sub in child.name.lower():
                                    rutas_busqueda.append(str(child))
                                    break
                            except Exception:
                                continue
                    except Exception:
                        continue
        except Exception:
            pass
        
        # 7. Directorios de desarrollo/backup (solo en modo desarrollo)
        if sistema_rutas and getattr(sistema_rutas, '_modo', None) == "desarrollo":
            ruta_analisis = sistema_rutas.base_dir / "analisis_reglas_anfis"
            if ruta_analisis.exists() and str(ruta_analisis) not in rutas_busqueda:
                rutas_busqueda.append(str(ruta_analisis))
        
        # Filtrar solo rutas que existen y eliminar duplicados
        rutas_validas = []
        for ruta in rutas_busqueda:
            if os.path.exists(ruta) and ruta not in rutas_validas:
                rutas_validas.append(ruta)
        
        # Si no hay rutas válidas, crear directorio de análisis como respaldo
        if not rutas_validas and sistema_rutas and config and hasattr(config.analisis, 'directorio_analisis'):
            analisis_dir = Path(config.analisis.directorio_analisis)
            analisis_dir.mkdir(parents=True, exist_ok=True)
            rutas_validas.append(str(analisis_dir))
        
        return rutas_validas
    
    def buscar_archivos(self):
        """Buscar archivos en múltiples ubicaciones según los tipos seleccionados"""
        carpetas_busqueda = self.obtener_rutas_busqueda()
        archivos = []
        
        extensiones_imagen = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        extensiones_texto = ['.txt', '.log', '.md', '.rst']
        extensiones_datos = ['.csv', '.json', '.xml', '.yaml', '.yml']
        
        for carpeta in carpetas_busqueda:
            try:
                carpeta_path = Path(carpeta)
                if carpeta_path.exists():
                    # Buscar todos los archivos
                    for archivo in carpeta_path.rglob("*"):
                        if archivo.is_file():
                            extension = archivo.suffix.lower()
                            
                            # Filtrar por tipo seleccionado
                            if (self.show_imagenes.get() and extension in extensiones_imagen) or \
                               (self.show_textos.get() and extension in extensiones_texto) or \
                               (self.show_datos.get() and extension in extensiones_datos):
                                
                                ruta_str = str(archivo)
                                if ruta_str not in archivos:
                                    archivos.append(ruta_str)
            except Exception as e:
                print(f" Error buscando en {carpeta}: {e}")
        
        # Ordenar por fecha de modificación (más recientes primero)
        archivos_ordenados = []
        for archivo in archivos:
            try:
                archivo_path = Path(archivo)
                if archivo_path.exists():
                    archivos_ordenados.append(archivo)
            except Exception as e:
                print(f" Error accediendo a {archivo}: {e}")
        
        archivos_ordenados.sort(key=lambda x: Path(x).stat().st_mtime if Path(x).exists() else 0, reverse=True)
        
        return archivos_ordenados
    
    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controles superiores
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="Seleccionar archivo:").pack(side=tk.LEFT, padx=(0, 10))

        # Combobox para archivos
        nombres_archivos = [Path(g).name for g in self.archivos_disponibles]
        self.combo_archivos = ttk.Combobox(control_frame, values=nombres_archivos, width=80)
        self.combo_archivos.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

        if nombres_archivos:
            self.combo_archivos.set(nombres_archivos[0])

        ttk.Button(control_frame, text="Actualizar Lista", command=self.actualizar_lista).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Buscar Archivo", command=self.buscar_archivo_manual).pack(side=tk.LEFT)

        # ===== FILTROS DE TIPO DE ARCHIVO =====
        filtros_frame = ttk.Frame(main_frame)
        filtros_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(filtros_frame, text="Mostrar tipos:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(filtros_frame, text="Imágenes", variable=self.show_imagenes,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(filtros_frame, text="Textos", variable=self.show_textos,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(filtros_frame, text="Datos", variable=self.show_datos,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)

        # ===== FUENTES A INCLUIR =====
        fuentes_frame = ttk.Frame(main_frame)
        fuentes_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(fuentes_frame, text="Incluir fuentes:").pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Checkbutton(fuentes_frame, text="Cache - Gráficos", variable=self.show_cache_graficos,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Cache - Reportes", variable=self.show_cache_reportes,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Cache - Métricas", variable=self.show_cache_metricas,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Análisis", variable=self.show_analisis,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Imágenes intermedias", variable=self.show_imagenes_intermedias,
                       command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        
        # Subcarpetas de imágenes intermedias
        subimgs_frame = ttk.Frame(fuentes_frame)
        subimgs_frame.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(subimgs_frame, text="[subcarpetas]").pack(side=tk.TOP, anchor='w')
        
        self.chk_original = ttk.Checkbutton(subimgs_frame, text="original", variable=self.show_img_original)
        self.chk_original.pack(side=tk.LEFT, padx=2)
        self.chk_enhanced = ttk.Checkbutton(subimgs_frame, text="enhanced", variable=self.show_img_enhanced)
        self.chk_enhanced.pack(side=tk.LEFT, padx=2)
        self.chk_blurred = ttk.Checkbutton(subimgs_frame, text="blurred", variable=self.show_img_blurred)
        self.chk_blurred.pack(side=tk.LEFT, padx=2)
        self.chk_mask = ttk.Checkbutton(subimgs_frame, text="mask", variable=self.show_img_mask)
        self.chk_mask.pack(side=tk.LEFT, padx=2)
        self.chk_roi = ttk.Checkbutton(subimgs_frame, text="roi", variable=self.show_img_roi)
        self.chk_roi.pack(side=tk.LEFT, padx=2)

        # Información
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        rutas_info = self.obtener_rutas_busqueda()
        info_text = f"Encontrados {len(self.archivos_disponibles)} archivos | Buscando en: {len(rutas_info)} ubicaciones"
        self.info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 9))
        self.info_label.pack(side=tk.LEFT)

        # Botón para mostrar rutas
        if rutas_info:
            ttk.Button(info_frame, text=" Ver Rutas",
                      command=lambda: self.mostrar_ventana_rutas(rutas_info),
                      width=10).pack(side=tk.RIGHT)

        # ===== ÁREA DE VISUALIZACIÓN PRINCIPAL =====
        # Frame para contenido (gráficos o texto)
        self.frame_contenido = ttk.Frame(main_frame)
        self.frame_contenido.pack(fill=tk.BOTH, expand=True)

        # Bind del combo
        self.combo_archivos.bind('<<ComboboxSelected>>', lambda e: self.on_combo_selection())

        # Configurar eventos para actualización automática
        try:
            self.show_imagenes_intermedias.trace_add('write', lambda *args: (self._toggle_subimgs_state(), self.actualizar_lista()))
            for var in [self.show_img_original, self.show_img_enhanced, self.show_img_blurred, 
                       self.show_img_mask, self.show_img_roi]:
                var.trace_add('write', lambda *args: self.actualizar_lista())
        except Exception:
            # Fallback para versiones antiguas de tkinter
            for chk in [self.chk_original, self.chk_enhanced, self.chk_blurred, self.chk_mask, self.chk_roi]:
                chk.config(command=self.actualizar_lista)

        self._toggle_subimgs_state()

        # Cargar primer archivo si existe
        if self.archivos_disponibles:
            self.mostrar_archivo(self.archivos_disponibles[0])
        else:
            self.mostrar_mensaje_no_archivos()

    def _toggle_subimgs_state(self):
        """Habilita o deshabilita los checkbuttons de subcarpetas según el estado del padre"""
        state = 'normal' if self.show_imagenes_intermedias.get() else 'disabled'
        try:
            for chk in (self.chk_original, self.chk_enhanced, self.chk_blurred, self.chk_mask, self.chk_roi):
                chk.config(state=state)
        except Exception:
            pass
    
    def mostrar_ventana_rutas(self, rutas):
        """Muestra una ventana con las rutas de búsqueda"""
        ventana_rutas = tk.Toplevel(self.root)
        ventana_rutas.title("Rutas de Búsqueda")
        ventana_rutas.geometry("600x400")
        
        ttk.Label(ventana_rutas, text="Rutas donde se buscan archivos:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        texto_rutas = scrolledtext.ScrolledText(ventana_rutas, wrap=tk.WORD, width=70, height=20)
        texto_rutas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, ruta in enumerate(rutas, 1):
            texto_rutas.insert(tk.END, f"{i}. {ruta}\n")
        
        # Añadir información del sistema de rutas
        if sistema_rutas:
            rutas_sistema = sistema_rutas.obtener_rutas_importantes()
            texto_rutas.insert(tk.END, f"\n--- SISTEMA DE RUTAS ---\n")
            for clave, valor in rutas_sistema.items():
                texto_rutas.insert(tk.END, f"{clave}: {valor}\n")
        
        texto_rutas.config(state=tk.DISABLED)
    
    def mostrar_mensaje_no_archivos(self):
        """Muestra mensaje cuando no hay archivos"""
        self.limpiar_area_contenido()
        mensaje_frame = ttk.Frame(self.frame_contenido)
        mensaje_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(mensaje_frame, text="No se encontraron archivos", 
                 font=('Arial', 14, 'bold')).pack(pady=20)
        
        info_text = """Para generar archivos:

1. Ejecute el pipeline ANFIS completo desde la ventana principal
2. Los archivos se guardarán automáticamente en las ubicaciones persistentes

Rutas de búsqueda actuales:"""
        
        text_widget = tk.Text(mensaje_frame, height=12, wrap=tk.WORD, font=('Arial', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        text_widget.insert(tk.END, info_text + "\n\n")
        
        for i, ruta in enumerate(self.obtener_rutas_busqueda(), 1):
            text_widget.insert(tk.END, f"{i}. {ruta}\n")
        
        # Añadir información del sistema
        if sistema_rutas:
            text_widget.insert(tk.END, f"\n--- INFORMACIÓN DEL SISTEMA ---\n")
            rutas_info = sistema_rutas.obtener_rutas_importantes()
            for clave, valor in rutas_info.items():
                text_widget.insert(tk.END, f"{clave}: {valor}\n")
        
        text_widget.config(state=tk.DISABLED)
        
        # Botón para actualizar
        ttk.Button(mensaje_frame, text="Actualizar Búsqueda", 
                  command=self.actualizar_lista).pack(pady=10)
    
    def actualizar_lista(self):
        """Actualiza la lista de archivos disponibles"""
        self.archivos_disponibles = self.buscar_archivos()
        
        # Actualizar combo con nombres de archivo
        nombres_archivos = [Path(g).name for g in self.archivos_disponibles]
        self.combo_archivos['values'] = nombres_archivos
        
        rutas_info = self.obtener_rutas_busqueda()
        info_text = f"Encontrados {len(self.archivos_disponibles)} archivos | Buscando en: {len(rutas_info)} ubicaciones"
        self.info_label.config(text=info_text)
        
        if self.archivos_disponibles:
            self.combo_archivos.set(nombres_archivos[0])
            self.mostrar_archivo(self.archivos_disponibles[0])
        else:
            self.mostrar_mensaje_no_archivos()
    
    def buscar_archivo_manual(self):
        """Abre diálogo para buscar archivo manualmente"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[
                ("Todos los archivos soportados", "*.png *.jpg *.jpeg *.txt *.log *.csv *.json"),
                ("Imágenes", "*.png *.jpg *.jpeg"),
                ("Archivos de texto", "*.txt *.log"),
                ("Datos", "*.csv *.json"),
                ("Todos los archivos", "*.*")
            ]
        )
        if archivo:
            # Agregar a la lista si no está
            if archivo not in self.archivos_disponibles:
                self.archivos_disponibles.insert(0, archivo)
                nombres_archivos = [Path(g).name for g in self.archivos_disponibles]
                self.combo_archivos['values'] = nombres_archivos
                self.combo_archivos.set(Path(archivo).name)
            
            self.mostrar_archivo(archivo)
    
    def mostrar_archivo(self, ruta_archivo):
        """Muestra el archivo seleccionado según su tipo"""
        self.limpiar_area_contenido()
        
        archivo_path = Path(ruta_archivo)
        if not archivo_path.exists():
            self.mostrar_error(f"Archivo no encontrado: {ruta_archivo}")
            return
        
        extension = archivo_path.suffix.lower()
        
        try:
            if extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                self.mostrar_imagen(archivo_path)
            elif extension in ['.txt', '.log', '.md', '.rst']:
                self.mostrar_texto(archivo_path)
            elif extension in ['.csv']:
                self.mostrar_csv(archivo_path)
            elif extension in ['.json']:
                self.mostrar_json(archivo_path)
            else:
                self.mostrar_texto(archivo_path)  # Intentar mostrar como texto por defecto
                
        except Exception as e:
            self.mostrar_error(f"Error al cargar archivo: {str(e)}")
            print(f" Error cargando {ruta_archivo}: {e}")
    
    def mostrar_imagen(self, archivo_path):
        """Muestra una imagen"""
        try:
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            img = plt.imread(archivo_path)

            # Detectar si es escala de grises
            is_gray = False
            try:
                if img.ndim == 2:
                    is_gray = True
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = img.reshape(img.shape[0], img.shape[1])
                    is_gray = True
            except Exception:
                is_gray = False

            if is_gray:
                if getattr(img, 'dtype', None) is not None and img.dtype == np.uint8:
                    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                else:
                    ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            ax.axis('off')
            nombre_archivo = archivo_path.name
            ax.set_title(f"Imagen: {nombre_archivo}", pad=20, fontsize=12)
            fig.tight_layout()
            
            # Integrar en tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.frame_contenido)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Barra de herramientas
            toolbar_frame = ttk.Frame(self.frame_contenido)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
        except Exception as e:
            raise e
    
    def mostrar_texto(self, archivo_path):
        """Muestra un archivo de texto"""
        try:
            # Detectar encoding
            encodings = ['utf-8', 'latin-1', 'cp1252']
            contenido = None
            
            for encoding in encodings:
                try:
                    with open(archivo_path, 'r', encoding=encoding) as f:
                        contenido = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if contenido is None:
                # Último intento con errores ignorados
                with open(archivo_path, 'r', encoding='utf-8', errors='ignore') as f:
                    contenido = f.read()
            
            # Crear widget de texto
            text_frame = ttk.Frame(self.frame_contenido)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(text_frame, text=f"Archivo: {archivo_path.name}", 
                     font=('Arial', 11, 'bold')).pack(pady=5)
            
            text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                   font=('Consolas', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget.insert(tk.END, contenido)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            raise e
    
    def mostrar_csv(self, archivo_path):
        """Muestra un archivo CSV en formato de tabla"""
        try:
            # Leer CSV
            df = pd.read_csv(archivo_path)
            
            # Crear frame para la tabla
            table_frame = ttk.Frame(self.frame_contenido)
            table_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(table_frame, text=f"CSV: {archivo_path.name} ({len(df)} filas, {len(df.columns)} columnas)", 
                     font=('Arial', 11, 'bold')).pack(pady=5)
            
            # Crear treeview para mostrar la tabla
            tree = ttk.Treeview(table_frame)
            
            # Configurar columnas
            tree["columns"] = list(df.columns)
            tree["show"] = "headings"
            
            # Agregar columnas
            for column in df.columns:
                tree.heading(column, text=column)
                tree.column(column, width=100)
            
            # Agregar datos
            for index, row in df.iterrows():
                tree.insert("", tk.END, values=list(row))
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Empaquetar
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
        except Exception as e:
            # Si falla la tabla, mostrar como texto
            self.mostrar_texto(archivo_path)
    
    def mostrar_json(self, archivo_path):
        """Muestra un archivo JSON formateado"""
        try:
            with open(archivo_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            # Formatear JSON
            contenido = json.dumps(datos, indent=2, ensure_ascii=False)
            
            # Mostrar como texto formateado
            text_frame = ttk.Frame(self.frame_contenido)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(text_frame, text=f"JSON: {archivo_path.name}", 
                     font=('Arial', 11, 'bold')).pack(pady=5)
            
            text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                   font=('Consolas', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget.insert(tk.END, contenido)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            # Si falla JSON, mostrar como texto plano
            self.mostrar_texto(archivo_path)
    
    def mostrar_error(self, mensaje):
        """Muestra un mensaje de error"""
        error_frame = ttk.Frame(self.frame_contenido)
        error_frame.pack(expand=True, fill=tk.BOTH)
        ttk.Label(error_frame, text=mensaje, 
                 foreground='red', font=('Arial', 10)).pack(expand=True)
    
    def on_combo_selection(self):
        """Maneja la selección del combobox"""
        selected_name = self.combo_archivos.get()
        for archivo in self.archivos_disponibles:
            if Path(archivo).name == selected_name:
                self.mostrar_archivo(archivo)
                break
    
    def limpiar_area_contenido(self):
        """Limpia el área de contenido"""
        for widget in self.frame_contenido.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = VentanaGraficos(root)
    root.mainloop()

if __name__ == "__main__":
    main()