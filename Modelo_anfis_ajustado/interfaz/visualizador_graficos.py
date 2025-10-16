# interfaz/visualizador_graficos.py - ACTUALIZADO

import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import glob

class VentanaGraficos:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Graficos - ANFIS")
        self.root.geometry("1000x700")
        
        self.graficos_disponibles = self.buscar_graficos()
        self.crear_interfaz()
    
    def buscar_graficos(self):
        """Buscar archivos de graficos generados solo en carpetas de resultados"""
        # Definir carpetas donde buscar graficos
        carpetas_busqueda = [
            "./Modelo_anfis_ajustado/utils/cache/resultados/graficos",  # Graficos en cache      # Graficos de analisis

        ]
        
        extensiones = ('*.png', '*.jpg', '*.jpeg')
        graficos = []
        
        for carpeta in carpetas_busqueda:
            if os.path.exists(carpeta):
                # Buscar con patrones
                for extension in extensiones:
                    patron = os.path.join(carpeta, extension)
                    graficos.extend(glob.glob(patron))
                
                # Buscar recursivamente en subdirectorios
                for root_dir, dirs, files in os.walk(carpeta):
                    for archivo in files:
                        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                            ruta_completa = os.path.join(root_dir, archivo)
                            graficos.append(ruta_completa)
        
        # Eliminar duplicados y ordenar
        graficos = list(set(graficos))
        graficos.sort(key=os.path.getmtime, reverse=True)  # Ordenar por fecha modificacion
        
        return graficos
    
    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Seleccionar grafico:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.combo_graficos = ttk.Combobox(control_frame, values=self.graficos_disponibles, width=50)
        self.combo_graficos.pack(side=tk.LEFT, padx=(0, 10))
        if self.graficos_disponibles:
            self.combo_graficos.set(self.graficos_disponibles[0])
        
        ttk.Button(control_frame, text="Actualizar Lista", 
                  command=self.actualizar_lista).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Buscar Archivo", 
                  command=self.buscar_archivo).pack(side=tk.LEFT)
        
        # Informacion
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text=f"Encontrados {len(self.graficos_disponibles)} graficos")
        self.info_label.pack(side=tk.LEFT)
        
        # Area del grafico
        self.frame_grafico = ttk.Frame(main_frame)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True)
        
        # Cargar primer grafico si existe
        if self.graficos_disponibles:
            self.mostrar_grafico(self.graficos_disponibles[0])
    
    def actualizar_lista(self):
        self.graficos_disponibles = self.buscar_graficos()
        self.combo_graficos['values'] = self.graficos_disponibles
        self.info_label.config(text=f"Encontrados {len(self.graficos_disponibles)} graficos")
        
        if self.graficos_disponibles:
            self.combo_graficos.set(self.graficos_disponibles[0])
            self.mostrar_grafico(self.graficos_disponibles[0])
        else:
            self.limpiar_area_grafico()
            ttk.Label(self.frame_grafico, text="No se encontraron graficos").pack(expand=True)
    
    def buscar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de grafico",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            self.combo_graficos.set(archivo)
            self.mostrar_grafico(archivo)
    
    def mostrar_grafico(self, ruta_archivo):
        self.limpiar_area_grafico()
        
        if not os.path.exists(ruta_archivo):
            ttk.Label(self.frame_grafico, text="Archivo no encontrado").pack(expand=True)
            return
        
        try:
            # Crear figura de matplotlib
            fig = plt.figure(figsize=(10, 6))
            img = plt.imread(ruta_archivo)
            plt.imshow(img)
            plt.axis('off')
            nombre_archivo = os.path.basename(ruta_archivo)
            plt.title(f"Grafico: {nombre_archivo}", pad=20, fontsize=12)
            plt.tight_layout()
            
            # Integrar en tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Barra de herramientas
            toolbar = NavigationToolbar2Tk(canvas, self.frame_grafico)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Configurar evento de seleccion del combo
            self.combo_graficos.bind('<<ComboboxSelected>>', 
                                   lambda e: self.mostrar_grafico(self.combo_graficos.get()))
            
        except Exception as e:
            ttk.Label(self.frame_grafico, text=f"Error al cargar imagen: {str(e)}").pack(expand=True)
    
    def limpiar_area_grafico(self):
        """Limpia el area del grafico"""
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = VentanaGraficos(root)
    root.mainloop()

if __name__ == "__main__":
    main()