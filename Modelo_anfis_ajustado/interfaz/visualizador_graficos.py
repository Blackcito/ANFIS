# interfaz/visualizador_graficos.py - VERSIÓN PORTABLE

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
        self.root.title("Visualizador de Gráficos - ANFIS")
        self.root.geometry("1000x700")
        # Flags para controlar las fuentes que se buscarán
        self.show_cache_graficos = tk.BooleanVar(value=True)         # cache/resultados/graficos
        self.show_analisis = tk.BooleanVar(value=True)               # directorio de analisis
        # Imagenes intermedias: parent + subfolders
        # Dejar el parent marcado tal como pidió el usuario
        self.show_imagenes_intermedias = tk.BooleanVar(value=True)

        # Sub-checks para carpetas internas dentro de directorio_imagenes_intermedias
        self.show_img_original = tk.BooleanVar(value=True)
        self.show_img_enhanced = tk.BooleanVar(value=True)
        self.show_img_blurred = tk.BooleanVar(value=True)
        self.show_img_mask = tk.BooleanVar(value=True)
        self.show_img_roi = tk.BooleanVar(value=True)

        self.graficos_disponibles = []
        self.crear_interfaz()
        # llenar lista inicial
        self.actualizar_lista()
    
    def obtener_rutas_busqueda(self):
        """Obtiene las rutas donde buscar gráficos usando sistema centralizado"""
        rutas_busqueda = []
        
        if not sistema_rutas:
            print(" Sistema de rutas no disponible")
            return [os.getcwd()]
        
        #  USAR SISTEMA CENTRALIZADO DE RUTAS
        

        # 2. Directorio de cache persistente para gráficos (opcional)
        try:
            cache_graficos = sistema_rutas.cache_dir / "resultados" / "graficos"
            if self.show_cache_graficos.get() and cache_graficos.exists():
                rutas_busqueda.append(str(cache_graficos))
        except Exception:
            pass

        # 3. Directorio de análisis persistente (opcional)
        try:
            if self.show_analisis.get() and config and hasattr(config.analisis, 'directorio_analisis'):
                analisis_dir = Path(config.analisis.directorio_analisis)
                if analisis_dir.exists():
                    rutas_busqueda.append(str(analisis_dir))
        except Exception:
            pass

        # 4. Directorio de imágenes intermedias (opcional) -> incluir subcarpetas seleccionadas
        try:
            if self.show_imagenes_intermedias.get() and config and hasattr(config.procesamiento, 'directorio_imagenes_intermedias'):
                imagenes_dir = Path(config.procesamiento.directorio_imagenes_intermedias)
                # Lista de pares (variable, nombre_subcarpeta)
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

                        # First try exact subfolder name (legacy)
                        ruta_sub = imagenes_dir / sub
                        if ruta_sub.exists():
                            rutas_busqueda.append(str(ruta_sub))
                            continue

                        # Fallback: some pipelines save with numeric prefixes like '01_original', '02_enhanced',
                        # so scan children and match by substring to be tolerant.
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
        
        # 5. Directorios de desarrollo/backup (solo en modo desarrollo)
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
            print(f" Creado directorio de análisis: {analisis_dir}")
        
        #print(f" Rutas de búsqueda configuradas: {len(rutas_validas)} ubicaciones")
        return rutas_validas
    
    def buscar_graficos(self):
        """Buscar archivos de gráficos en múltiples ubicaciones - ACTUALIZADO"""
        carpetas_busqueda = self.obtener_rutas_busqueda()
        extensiones = ('*.png', '*.jpg', '*.jpeg')
        graficos = []
        
        #print(f" Buscando gráficos en {len(carpetas_busqueda)} ubicaciones...")
        
        for carpeta in carpetas_busqueda:
            try:
                carpeta_path = Path(carpeta)
                if carpeta_path.exists():
                    #  USAR pathlib PARA MÁS PORTABILIDAD
                    for extension in extensiones:
                        patron = carpeta_path / "**" / extension
                        graficos.extend([str(p) for p in carpeta_path.glob(f"**/{extension}")])
                    
                    # Búsqueda recursiva adicional
                    for archivo in carpeta_path.rglob("*"):
                        if archivo.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            ruta_str = str(archivo)
                            if ruta_str not in graficos:
                                graficos.append(ruta_str)
            except Exception as e:
                print(f" Error buscando en {carpeta}: {e}")
        
        # Ordenar por fecha de modificación (más recientes primero)
        graficos_ordenados = []
        for grafico in graficos:
            try:
                grafico_path = Path(grafico)
                if grafico_path.exists():
                    graficos_ordenados.append(grafico)
            except Exception as e:
                print(f" Error accediendo a {grafico}: {e}")
        
        # Ordenar por fecha de modificación
        graficos_ordenados.sort(key=lambda x: Path(x).stat().st_mtime if Path(x).exists() else 0, reverse=True)
        
        #print(f" Encontrados {len(graficos_ordenados)} gráficos")
        return graficos_ordenados
    
    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="Seleccionar gráfico:").pack(side=tk.LEFT, padx=(0, 10))

        # Mostrar nombres de archivo en el combo, no rutas completas
        nombres_graficos = [Path(g).name for g in self.graficos_disponibles]
        self.combo_graficos = ttk.Combobox(control_frame, values=nombres_graficos, width=80)
        self.combo_graficos.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

        if nombres_graficos:
            self.combo_graficos.set(nombres_graficos[0])

        ttk.Button(control_frame, text="Actualizar Lista", command=self.actualizar_lista).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Buscar Archivo", command=self.buscar_archivo).pack(side=tk.LEFT)

        # ===== Fuentes a incluir (checkboxes) =====
        fuentes_frame = ttk.Frame(main_frame)
        fuentes_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(fuentes_frame, text="Incluir fuentes:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(fuentes_frame, text="Cache - Gráficos", variable=self.show_cache_graficos,
                        command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Análisis", variable=self.show_analisis,
                        command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(fuentes_frame, text="Imágenes intermedias", variable=self.show_imagenes_intermedias,
                        command=self.actualizar_lista).pack(side=tk.LEFT, padx=5)
        # Mini-caja con subcarpetas dentro de 'imagenes_intermedias'
        subimgs_frame = ttk.Frame(fuentes_frame)
        subimgs_frame.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(subimgs_frame, text="[subcarpetas]").pack(side=tk.TOP, anchor='w')
        # Create and keep references so we can enable/disable them when parent toggles
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
        info_text = f"Encontrados {len(self.graficos_disponibles)} gráficos | Buscando en: {len(rutas_info)} ubicaciones"
        self.info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 9))
        self.info_label.pack(side=tk.LEFT)

        # Botón para mostrar rutas
        if rutas_info:
            ttk.Button(info_frame, text=" Ver Rutas",
                       command=lambda: self.mostrar_ventana_rutas(rutas_info),
                       width=10).pack(side=tk.RIGHT)

        # Área del gráfico
        self.frame_grafico = ttk.Frame(main_frame)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True)

        # Bind del combo (una sola vez)
        self.combo_graficos.bind('<<ComboboxSelected>>', lambda e: self.on_combo_selection())

        # Bind variable traces for subfolder toggles and parent toggle so changes always refresh
        # Use trace_add (available in tkinter) to reliably call actualizar_lista on changes
        try:
            # Parent toggle: enable/disable subchecks when changed
            self.show_imagenes_intermedias.trace_add('write', lambda *args: (self._toggle_subimgs_state(), self.actualizar_lista()))
            # Subfolder toggles: refresh list when any subfolder changes
            self.show_img_original.trace_add('write', lambda *args: self.actualizar_lista())
            self.show_img_enhanced.trace_add('write', lambda *args: self.actualizar_lista())
            self.show_img_blurred.trace_add('write', lambda *args: self.actualizar_lista())
            self.show_img_mask.trace_add('write', lambda *args: self.actualizar_lista())
            self.show_img_roi.trace_add('write', lambda *args: self.actualizar_lista())
        except Exception:
            # Fallback: rely on Checkbutton command callbacks (older tkinter)
            self.chk_original.config(command=self.actualizar_lista)
            self.chk_enhanced.config(command=self.actualizar_lista)
            self.chk_blurred.config(command=self.actualizar_lista)
            self.chk_mask.config(command=self.actualizar_lista)
            self.chk_roi.config(command=self.actualizar_lista)

        # Ensure initial enabled/disabled state of subchecks matches parent
        self._toggle_subimgs_state()

        # Cargar primer gráfico si existe
        if self.graficos_disponibles:
            self.mostrar_grafico(self.graficos_disponibles[0])
        else:
            self.mostrar_mensaje_no_graficos()

    def _toggle_subimgs_state(self):
        """Enable or disable the subfolder checkbuttons depending on parent toggle."""
        state = 'normal' if self.show_imagenes_intermedias.get() else 'disabled'
        try:
            for chk in (self.chk_original, self.chk_enhanced, self.chk_blurred, self.chk_mask, self.chk_roi):
                chk.config(state=state)
        except Exception:
            # If any widget is not yet created, ignore
            pass
    
    def mostrar_ventana_rutas(self, rutas):
        """Muestra una ventana con las rutas de búsqueda"""
        ventana_rutas = tk.Toplevel(self.root)
        ventana_rutas.title("Rutas de Búsqueda de Gráficos")
        ventana_rutas.geometry("600x400")
        
        ttk.Label(ventana_rutas, text="Rutas donde se buscan gráficos:", 
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
    
    def mostrar_mensaje_no_graficos(self):
        """Muestra mensaje cuando no hay gráficos"""
        self.limpiar_area_grafico()
        mensaje_frame = ttk.Frame(self.frame_grafico)
        mensaje_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(mensaje_frame, text="No se encontraron gráficos", 
                 font=('Arial', 14, 'bold')).pack(pady=20)
        
        info_text = """Para generar gráficos:

1. Ejecute el pipeline ANFIS completo desde la ventana principal
2. Los gráficos se guardarán automáticamente en las ubicaciones persistentes

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
        """Actualiza la lista de gráficos disponibles"""
        self.graficos_disponibles = self.buscar_graficos()
        
        # Actualizar combo con nombres de archivo
        nombres_graficos = [Path(g).name for g in self.graficos_disponibles]
        self.combo_graficos['values'] = nombres_graficos
        
        rutas_info = self.obtener_rutas_busqueda()
        info_text = f"Encontrados {len(self.graficos_disponibles)} gráficos | Buscando en: {len(rutas_info)} ubicaciones"
        self.info_label.config(text=info_text)
        
        if self.graficos_disponibles:
            self.combo_graficos.set(nombres_graficos[0])
            self.mostrar_grafico(self.graficos_disponibles[0])
        else:
            self.mostrar_mensaje_no_graficos()
    
    def buscar_archivo(self):
        """Abre diálogo para buscar archivo manualmente"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de gráfico",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            # Agregar a la lista si no está
            if archivo not in self.graficos_disponibles:
                self.graficos_disponibles.insert(0, archivo)
                nombres_graficos = [Path(g).name for g in self.graficos_disponibles]
                self.combo_graficos['values'] = nombres_graficos
                self.combo_graficos.set(Path(archivo).name)
            
            self.mostrar_grafico(archivo)
    
    def mostrar_grafico(self, ruta_archivo):
        """Muestra el gráfico seleccionado"""
        self.limpiar_area_grafico()
        
        archivo_path = Path(ruta_archivo)
        if not archivo_path.exists():
            error_frame = ttk.Frame(self.frame_grafico)
            error_frame.pack(expand=True, fill=tk.BOTH)
            ttk.Label(error_frame, text=f"Archivo no encontrado: {ruta_archivo}", 
                     foreground='red', font=('Arial', 10)).pack(expand=True)
            return
        
        try:
            # Crear figura usando Figure directamente (evita plt.figure)
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            img = plt.imread(archivo_path)

            # Mostrar en escala de grises si la imagen es monocroma.
            # plt.imread puede devolver arrays 2D (grayscale) o 3D (RGB/RGBA).
            is_gray = False
            try:
                if img.ndim == 2:
                    is_gray = True
                elif img.ndim == 3 and img.shape[2] == 1:
                    # Some images may have a singleton channel dimension
                    img = img.reshape(img.shape[0], img.shape[1])
                    is_gray = True
            except Exception:
                is_gray = False

            if is_gray:
                # If uint8 use 0-255 scale, otherwise let matplotlib auto-scale
                if getattr(img, 'dtype', None) is not None and img.dtype == np.uint8:
                    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                else:
                    ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')
            nombre_archivo = archivo_path.name
            ax.set_title(f"Gráfico: {nombre_archivo}", pad=20, fontsize=12)
            fig.tight_layout()
            
            # Integrar en tkinter usando el backend TkAgg
            canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Barra de herramientas
            toolbar_frame = ttk.Frame(self.frame_grafico)
            toolbar_frame.pack(fill=tk.X)

            # Usar NavigationToolbar2Tk importado al inicio
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
        except Exception as e:
            error_frame = ttk.Frame(self.frame_grafico)
            error_frame.pack(expand=True, fill=tk.BOTH)
            ttk.Label(error_frame, text=f"Error al cargar imagen: {str(e)}", 
                     foreground='red', font=('Arial', 10)).pack(expand=True)
            print(f" Error cargando {ruta_archivo}: {e}")
    
    def on_combo_selection(self):
        """Maneja la selección del combobox"""
        selected_name = self.combo_graficos.get()
        # Encontrar la ruta completa correspondiente al nombre seleccionado
        for grafico in self.graficos_disponibles:
            if Path(grafico).name == selected_name:
                self.mostrar_grafico(grafico)
                break
    
    def limpiar_area_grafico(self):
        """Limpia el área del gráfico"""
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = VentanaGraficos(root)
    root.mainloop()

if __name__ == "__main__":
    main()