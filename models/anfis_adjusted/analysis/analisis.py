import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from itertools import product

from core.anfis_sugeno import compute_weights, n_vars, reglas
import warnings
warnings.filterwarnings('ignore')

# Importar configuración y caché
from config.configuracion import config
from utils.cache import sistema_cache

# Configuración para español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class AnalizadorReglasANFIS:
    def __init__(self, mf_params, theta, X_data, y_data):
        """
        Inicializa el analizador de reglas ANFIS
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
        self.top_reglas = config.analisis.top_reglas_mostrar
        self.ultimas_metricas_clasificacion = {}
        
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
            #print(f"Reporte guardado en: {archivo_salida}")
        
        return reporte_texto
    
    def graficar_importancia_reglas(self, n_top=15, save_path=None, visualizar_graficos=True):
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
            #print(f"Gráfico guardado en: {save_path}")
        if visualizar_graficos:
            try:
                plt.show()
            except Exception:
                pass
        return fig
    
    def crear_mapa_calor_reglas(self, n_top=10, save_path=None, visualizar_graficos=True):
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
            #print(f"Mapa de calor guardado en: {save_path}")
        if visualizar_graficos:
            try:
                plt.show()
            except Exception:
                pass
        return fig
    def analizar_contribucion_caracteristicas(self, save_path=None, visualizar_graficos=True):
        """
        Analiza qué características son más importantes globalmente
        """
        # Delegar el cálculo a un método dedicado y luego plotear si corresponde
        importancia_por_caracteristica = self.calcular_importancia_por_caracteristica()

        # Crear gráfico usando la información ya calculada
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
            #print(f"Gráfico de características guardado en: {save_path}")
        if visualizar_graficos:
            try:
                plt.show()
            except Exception:
                pass
        return fig, importancia_por_caracteristica

    def calcular_importancia_por_caracteristica(self):
        """
        Calcula y retorna la importancia acumulada por característica a partir de
        `self.importancia_reglas`. No realiza plotting; puede usarse independientemente
        del flag de guardar gráficos.
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()

        importancia_por_caracteristica = np.zeros(n_vars)

        for regla in self.importancia_reglas:
            params_abs = np.abs(regla['parametros_consecuente'])
            importancia_por_caracteristica += params_abs * regla['activacion_media']

        return importancia_por_caracteristica
    
    def generar_analisis_completo(self, carpeta_salida=None, guardar_graficos_analisis=None, guardar_graficos_cache=None, visualizar_graficos=True):
        """
        Genera análisis completo con todos los gráficos y reportes - ACTUALIZADO
        """
        carpeta_salida = config.analisis.directorio_analisis
        datos_csv = None
        datos_txt = None
        
        import os
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
        
        print("Generando análisis completo de reglas ANFIS...")
        
    
        # 2. Gráficos de análisis (carpeta análisis)
        fig_importancia = None
        fig_mapa = None
        fig_caract = None
        if guardar_graficos_analisis:
            importancia_path = os.path.join(carpeta_salida, "importancia_reglas.png")
            fig_importancia = self.graficar_importancia_reglas(
                n_top=self.top_reglas, 
                save_path=importancia_path,
                visualizar_graficos=visualizar_graficos
            )
            mapa_calor_path = os.path.join(carpeta_salida, "mapa_calor_reglas.png")
            fig_mapa = self.crear_mapa_calor_reglas(
                n_top=min(12, self.top_reglas), 
                save_path=mapa_calor_path,
                visualizar_graficos=visualizar_graficos
            )
            caracteristicas_path = os.path.join(carpeta_salida, "importancia_caracteristicas.png")
            fig_caract, imp_caract = self.analizar_contribucion_caracteristicas(
                save_path=caracteristicas_path,
                visualizar_graficos=visualizar_graficos
            )
        else:
            # Siempre calcular la métrica aunque no se guarde el gráfico
            imp_caract = self.calcular_importancia_por_caracteristica()
            # Mostrar los gráficos aunque no se guarden
            fig_importancia = self.graficar_importancia_reglas(n_top=self.top_reglas, visualizar_graficos=visualizar_graficos)
            fig_mapa = self.crear_mapa_calor_reglas(n_top=min(12, self.top_reglas), visualizar_graficos=visualizar_graficos)
            fig_caract, _ = self.analizar_contribucion_caracteristicas(visualizar_graficos=visualizar_graficos)

        # 3. Guardar en cache si corresponde
        if guardar_graficos_cache:
            if fig_importancia is not None:
                sistema_cache.guardar_grafico("importancia_reglas", fig_importancia)
            if fig_mapa is not None:
                sistema_cache.guardar_grafico("mapa_calor_reglas", fig_mapa)
            if fig_caract is not None:
                sistema_cache.guardar_grafico("importancia_caracteristicas", fig_caract)

        ####### Guardado de reportes, métricas y datos ####

        # 1. Generar contenidos
        reporte = self.generar_reporte_textual(n_top=self.top_reglas)
        datos_csv, datos_txt = self.exportar_datos_reglas()
        valores_metricas = imp_caract.tolist() if hasattr(imp_caract, 'tolist') else list(imp_caract)
        
        # Preparar métricas unificadas
        metricas = {
            'analisis_reglas': {
                'importancia_caracteristicas': valores_metricas,
                'nombres_caracteristicas': self.nombres_caracteristicas
            }
        }
        
        # Agregar métricas de clasificación si están disponibles
        if self.ultimas_metricas_clasificacion:
            metricas.update(self.ultimas_metricas_clasificacion)

        # Guardar métricas unificadas en memoria para el pipeline
        self.metricas_unificadas = metricas

        # 2. Guardar en carpeta de análisis si está configurado
        if config.analisis.guardar_reportes:
            reporte_path = os.path.join(carpeta_salida, "reporte_reglas.txt")
            with open(reporte_path, 'w', encoding='utf-8') as f:
                f.write(reporte)
                
        if config.analisis.guardar_metricas:
            metricas_path = os.path.join(carpeta_salida, "metricas.json")
            with open(metricas_path, 'w', encoding='utf-8') as f:
                json.dump(metricas, f, indent=2)

        if config.analisis.guardar_datos_reglas:
            datos_base = os.path.join(carpeta_salida, "datos_reglas")
            with open(datos_base + ".csv", 'w', encoding='utf-8') as f:
                f.write(datos_csv)
            with open(datos_base + ".txt", 'w', encoding='utf-8') as f:
                f.write(datos_txt)

        # 3. Guardar en caché si está configurado
        if config.cache.guardar_cache_reportes:
            sistema_cache.guardar_reporte("reporte_reglas", reporte)
            
        if config.cache.guardar_cache_metricas:
            # Guardar métricas unificadas en caché usando el nombre del modelo
            # para evitar crear dos ficheros distintos.
            try:
                sistema_cache.guardar_metricas(config.entrenamiento.nombre_modelo, metricas)
                #print("guardando metricas en cache... (unificadas)")
            except Exception as e:
                print(f"Error guardando metricas en cache: {e}")
            
        if config.cache.guardar_cache_datos_reglas:
            sistema_cache.guardar_datos_reglas("analisis", datos_csv, datos_txt)
        
        #print(f" Análisis completo guardado en: {carpeta_salida}")

        return {
            'reporte_texto': reporte if config.analisis.guardar_reportes else "No generado",
            'top_reglas': self.obtener_top_reglas(self.top_reglas),
            'importancia_caracteristicas': imp_caract,
            'graficos_generados': guardar_graficos_analisis
        }
    
    def exportar_datos_reglas(self, archivo_salida=None):
        """
        Genera y opcionalmente exporta los datos de las reglas en formatos CSV y TXT.
        Retorna una tupla con el contenido CSV y TXT generado.
        """
        if len(self.importancia_reglas) == 0:
            self.calcular_importancia_reglas()

        # Generar DataFrame
        datos_export = []
        for regla in self.importancia_reglas:
            fila = {
                'regla_idx': regla['regla_idx'],
                'activacion_media': regla['activacion_media'],
                'magnitud_parametros': regla['magnitud_parametros'],
                'importancia_total': regla['importancia_total'],
                'bias': regla['bias']
            }
            for i, condicion in enumerate(regla['regla_condicion']):
                fila[f'{self.nombres_caracteristicas[i]}_condicion'] = condicion
            for i, param in enumerate(regla['parametros_consecuente']):
                fila[f'{self.nombres_caracteristicas[i]}_parametro'] = param
            datos_export.append(fila)
        
        df = pd.DataFrame(datos_export)
        
        # Generar contenido CSV
        contenido_csv = df.to_csv(index=False)
        
        # Generar contenido TXT formateado
        lineas_txt = []
        lineas_txt.append("="*80)
        lineas_txt.append("DATOS DETALLADOS DE REGLAS ANFIS")
        lineas_txt.append("="*80 + "\n")
        
        for _, fila in df.iterrows():
            lineas_txt.append(f"Regla {fila['regla_idx']}:")
            lineas_txt.append(f"  Activación media: {fila['activacion_media']:.4f}")
            lineas_txt.append(f"  Magnitud parámetros: {fila['magnitud_parametros']:.4f}")
            lineas_txt.append(f"  Importancia total: {fila['importancia_total']:.4f}")
            lineas_txt.append(f"  Bias: {fila['bias']:.4f}")
            lineas_txt.append("  Condiciones:")
            for nombre in self.nombres_caracteristicas:
                cond_col = f"{nombre}_condicion"
                if cond_col in fila:
                    lineas_txt.append(f"    - {nombre}: {fila[cond_col]}")
            lineas_txt.append("  Parámetros:")
            for nombre in self.nombres_caracteristicas:
                param_col = f"{nombre}_parametro"
                if param_col in fila:
                    lineas_txt.append(f"    - {nombre}: {fila[param_col]:.4f}")
            lineas_txt.append("\n" + "-"*50 + "\n")
            
        contenido_txt = "\n".join(lineas_txt)

        # Si se proporciona ruta, guardar archivos
        if archivo_salida:
            ruta_csv = archivo_salida if archivo_salida.endswith('.csv') else archivo_salida + '.csv'
            ruta_txt = archivo_salida if archivo_salida.endswith('.txt') else archivo_salida + '.txt'
            
            with open(ruta_csv, 'w', encoding='utf-8') as f:
                f.write(contenido_csv)
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write(contenido_txt)
                
            print(f"Datos exportados en:\n  CSV: {ruta_csv}\n  TXT: {ruta_txt}")

        # Retornar contenidos generados
        return contenido_csv, contenido_txt



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