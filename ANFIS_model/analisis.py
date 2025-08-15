import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from anfis_sugeno import compute_weights, n_vars, reglas
import warnings
warnings.filterwarnings('ignore')

# Configuración para español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class AnalizadorReglasANFIS:
    def __init__(self, mf_params, theta, X_data, y_data):
        """
        Inicializa el analizador de reglas ANFIS
        
        Args:
            mf_params: Parámetros de funciones de membresía optimizados
            theta: Parámetros del consecuente Sugeno
            X_data: Datos de entrada (características GLCM)
            y_data: Etiquetas verdaderas
        """
        self.mf_params = mf_params
        self.theta = theta
        self.X_data = X_data
        self.y_data = y_data
        self.nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                                       'Energía', 'Media', 'Entropía', 'Varianza']
        self.reglas = list(product(['bajo', 'alto'], repeat=n_vars))
        self.activaciones_reglas = []
        self.importancia_reglas = []
        
    def calcular_activaciones_globales(self):
        """
        Calcula la activación de cada regla para todo el dataset
        """
        activaciones_totales = np.zeros(len(self.reglas))
        
        for x in self.X_data:
            w = compute_weights(x, self.mf_params)
            activaciones_totales += w
            
        # Normalizar por número de muestras
        self.activaciones_reglas = activaciones_totales / len(self.X_data)
        return self.activaciones_reglas
    
    def calcular_importancia_reglas(self):
        """
        Calcula la importancia de cada regla basada en:
        1. Frecuencia de activación
        2. Magnitud de los parámetros del consecuente
        3. Contribución a la salida final
        """
        if len(self.activaciones_reglas) == 0:
            self.calcular_activaciones_globales()
            
        importancia = []
        
        for j, regla in enumerate(self.reglas):
            # Parámetros del consecuente para esta regla
            p_j = self.theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
            r_j = self.theta[j*(n_vars+1) + n_vars]
            
            # Importancia = activación * magnitud de parámetros
            magnitud_params = np.sqrt(np.sum(p_j**2) + r_j**2)
            importancia_regla = self.activaciones_reglas[j] * magnitud_params
            
            importancia.append({
                'regla_idx': j,
                'regla_texto': self._regla_a_texto(regla),
                'regla_condicion': regla,
                'activacion_media': self.activaciones_reglas[j],
                'magnitud_parametros': magnitud_params,
                'importancia_total': importancia_regla,
                'parametros_consecuente': p_j,
                'bias': r_j
            })
            
        self.importancia_reglas = sorted(importancia, 
                                       key=lambda x: x['importancia_total'], 
                                       reverse=True)
        return self.importancia_reglas
    
    def _regla_a_texto(self, regla):
        """Convierte una regla de tupla a texto legible"""
        condiciones = []
        for i, etiqueta in enumerate(regla):
            condiciones.append(f"{self.nombres_caracteristicas[i]} es {etiqueta}")
        return "SI " + " Y ".join(condiciones)
    
    def obtener_top_reglas(self, n_top=10):
        """
        Obtiene las n reglas más importantes
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
            
        return self.importancia_reglas[:n_top]
    
    def generar_reporte_textual(self, n_top=15, archivo_salida=None):
        """
        Genera un reporte textual con las reglas más importantes
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
            
        reporte = []
        reporte.append("="*80)
        reporte.append("ANÁLISIS DE REGLAS ANFIS - DETECCIÓN DE TUMORES CEREBRALES")
        reporte.append("="*80)
        reporte.append(f"\nTotal de reglas generadas: {len(self.reglas)}")
        reporte.append(f"Reglas con activación > 0.01: {sum(1 for r in self.importancia_reglas if r['activacion_media'] > 0.01)}")
        reporte.append(f"Top {n_top} reglas más importantes:\n")
        
        for i, regla in enumerate(self.importancia_reglas[:n_top]):
            reporte.append(f"\n--- REGLA #{i+1} (Índice: {regla['regla_idx']}) ---")
            reporte.append(f"Condición: {regla['regla_texto']}")
            reporte.append(f"Activación media: {regla['activacion_media']:.4f}")
            reporte.append(f"Magnitud parámetros: {regla['magnitud_parametros']:.4f}")
            reporte.append(f"Importancia total: {regla['importancia_total']:.4f}")
            reporte.append(f"Bias: {regla['bias']:.4f}")
            
            # Mostrar parámetros más influyentes
            params = np.abs(regla['parametros_consecuente'])
            top_params = np.argsort(params)[-3:][::-1]
            reporte.append("Características más influyentes:")
            for idx in top_params:
                reporte.append(f"  - {self.nombres_caracteristicas[idx]}: {regla['parametros_consecuente'][idx]:.4f}")
        
        reporte_texto = "\n".join(reporte)
        
        if archivo_salida:
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                f.write(reporte_texto)
            print(f"Reporte guardado en: {archivo_salida}")
        
        return reporte_texto
    
    def graficar_importancia_reglas(self, n_top=15, save_path=None):
        """
        Crea gráfico de barras con las reglas más importantes
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Gráfico 1: Importancia total
        top_reglas = self.importancia_reglas[:n_top]
        indices = [f"R{r['regla_idx']}" for r in top_reglas]
        importancias = [r['importancia_total'] for r in top_reglas]
        
        bars1 = ax1.bar(indices, importancias, color='skyblue', alpha=0.7)
        ax1.set_title(f'Top {n_top} Reglas ANFIS por Importancia Total', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Índice de Regla')
        ax1.set_ylabel('Importancia Total')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, val in zip(bars1, importancias):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Gráfico 2: Activación media
        activaciones = [r['activacion_media'] for r in top_reglas]
        bars2 = ax2.bar(indices, activaciones, color='lightcoral', alpha=0.7)
        ax2.set_title(f'Activación Media de Top {n_top} Reglas', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Índice de Regla')
        ax2.set_ylabel('Activación Media')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, val in zip(bars2, activaciones):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        return fig
    
    def crear_mapa_calor_reglas(self, n_top=10, save_path=None):
        """
        Crea un mapa de calor mostrando las condiciones de las reglas más importantes
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
            
        # Preparar matriz para el mapa de calor
        top_reglas = self.importancia_reglas[:n_top]
        matriz_reglas = []
        etiquetas_reglas = []
        
        for regla in top_reglas:
            fila = []
            for condicion in regla['regla_condicion']:
                fila.append(1 if condicion == 'alto' else 0)
            matriz_reglas.append(fila)
            etiquetas_reglas.append(f"R{regla['regla_idx']}")
        
        matriz_reglas = np.array(matriz_reglas)
        
        # Crear el mapa de calor
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(matriz_reglas, cmap='RdYlBu_r', aspect='auto')
        
        # Configurar etiquetas
        ax.set_xticks(range(len(self.nombres_caracteristicas)))
        ax.set_xticklabels(self.nombres_caracteristicas, rotation=45, ha='right')
        ax.set_yticks(range(len(etiquetas_reglas)))
        ax.set_yticklabels(etiquetas_reglas)
        
        # Añadir texto en cada celda
        for i in range(len(etiquetas_reglas)):
            for j in range(len(self.nombres_caracteristicas)):
                text = 'ALTO' if matriz_reglas[i, j] == 1 else 'BAJO'
                color = 'white' if matriz_reglas[i, j] == 1 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=8)
        
        ax.set_title(f'Condiciones de las Top {n_top} Reglas ANFIS\n(Azul=BAJO, Rojo=ALTO)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Características GLCM')
        ax.set_ylabel('Reglas (ordenadas por importancia)')
        
        # Añadir barra de color
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nivel de la característica')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['BAJO', 'ALTO'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mapa de calor guardado en: {save_path}")
        
        plt.show()
        return fig
    
    def analizar_contribucion_caracteristicas(self, save_path=None):
        """
        Analiza qué características son más importantes globalmente
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
        
        # Calcular importancia por característica
        importancia_por_caracteristica = np.zeros(n_vars)
        
        for regla in self.importancia_reglas:
            params_abs = np.abs(regla['parametros_consecuente'])
            importancia_por_caracteristica += params_abs * regla['activacion_media']
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(self.nombres_caracteristicas, importancia_por_caracteristica, 
                     color='green', alpha=0.7)
        ax.set_title('Importancia Global por Característica GLCM', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Características GLCM')
        ax.set_ylabel('Importancia Acumulada')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, importancia_por_caracteristica):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de características guardado en: {save_path}")
        
        plt.show()
        return fig, importancia_por_caracteristica
    
    def generar_analisis_completo(self, carpeta_salida="analisis_anfis"):
        """
        Genera análisis completo con todos los gráficos y reportes
        """
        import os
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
        
        print("Generando análisis completo de reglas ANFIS...")
        
        # 1. Reporte textual
        reporte_path = os.path.join(carpeta_salida, "reporte_reglas.txt")
        reporte = self.generar_reporte_textual(n_top=20, archivo_salida=reporte_path)
        
        # 2. Gráfico de importancia
        importancia_path = os.path.join(carpeta_salida, "importancia_reglas.png")
        self.graficar_importancia_reglas(n_top=15, save_path=importancia_path)
        
        # 3. Mapa de calor
        mapa_calor_path = os.path.join(carpeta_salida, "mapa_calor_reglas.png")
        self.crear_mapa_calor_reglas(n_top=12, save_path=mapa_calor_path)
        
        # 4. Análisis de características
        caracteristicas_path = os.path.join(carpeta_salida, "importancia_caracteristicas.png")
        fig_caract, imp_caract = self.analizar_contribucion_caracteristicas(save_path=caracteristicas_path)
        
        # 5. Crear CSV con datos de reglas
        self.exportar_datos_reglas(os.path.join(carpeta_salida, "datos_reglas.csv"))
        
        print(f"\nAnálisis completo guardado en la carpeta: {carpeta_salida}")
        
        return {
            'reporte_texto': reporte,
            'top_reglas': self.obtener_top_reglas(20),
            'importancia_caracteristicas': imp_caract
        }
    
    def exportar_datos_reglas(self, archivo_csv):
        """
        Exporta datos de todas las reglas a CSV para análisis posterior
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()
        
        datos_export = []
        for regla in self.importancia_reglas:
            fila = {
                'regla_idx': regla['regla_idx'],
                'activacion_media': regla['activacion_media'],
                'magnitud_parametros': regla['magnitud_parametros'],
                'importancia_total': regla['importancia_total'],
                'bias': regla['bias']
            }
            
            # Añadir condiciones
            for i, condicion in enumerate(regla['regla_condicion']):
                fila[f'{self.nombres_caracteristicas[i]}_condicion'] = condicion
            
            # Añadir parámetros del consecuente
            for i, param in enumerate(regla['parametros_consecuente']):
                fila[f'{self.nombres_caracteristicas[i]}_parametro'] = param
            
            datos_export.append(fila)
        
        df = pd.DataFrame(datos_export)
        df.to_csv(archivo_csv, index=False)
        print(f"Datos de reglas exportados a: {archivo_csv}")


# Función de uso fácil
def analizar_reglas_anfis(mf_params, theta, X_data, y_data, carpeta_salida="analisis_anfis"):
    """
    Función de conveniencia para análisis rápido de reglas ANFIS
    """
    analizador = AnalizadorReglasANFIS(mf_params, theta, X_data, y_data)
    return analizador.generar_analisis_completo(carpeta_salida)


if __name__ == "__main__":
    # Ejemplo de uso (necesitarías tener los datos cargados)
    print("Para usar este módulo, importa y ejecuta:")
    print("from analisis_reglas import analizar_reglas_anfis")
    print("resultados = analizar_reglas_anfis(mf_opt, theta_opt, X_train, y_train)")