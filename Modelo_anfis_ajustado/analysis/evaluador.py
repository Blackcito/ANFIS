# analysis/evaluador.py - CORREGIR

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from core.prediction import predict_sugeno
import os

# Importar configuración y caché
from config.configuracion import config
from utils.cache import sistema_cache

class EvaluadorANFIS:
    def __init__(self, modelo, datos):
        self.mf_params = modelo['mf_params']
        self.theta = modelo['theta']
        self.X = datos['X']
        self.y = datos['y']
        self.nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                                       'Energia', 'Media', 'Entropia', 'Varianza']
    
    def evaluar_modelo(self, save_plots=None):
        """Evaluacion completa del modelo con graficos ROC"""
        # Usar configuracion si no se especifica
        save_plots = save_plots if save_plots is not None else config.analisis.guardar_graficos

        print("\n" + "="*50)
        print("EVALUACION DEL MODELO ANFIS")
        print("="*50)
        
        # Predicciones
        y_cont, y_pred = predict_sugeno(self.X, self.mf_params, self.theta)
        
        # Metricas basicas
        print("\nREPORTE DE CLASIFICACION:")
        print("-" * 40)
        reporte = classification_report(self.y, y_pred, target_names=['No Tumor', 'Tumor'], digits=4)
        print(reporte)
        
        print("\nMATRIZ DE CONFUSION:")
        print("-" * 30)
        cm = confusion_matrix(self.y, y_pred)
        print(cm)
        
        # Metricas detalladas
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\nMETRICAS DETALLADAS:")
        print(f"  Sensibilidad (Recall): {sensitivity:.4f}")
        print(f"  Especificidad: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Curva ROC
        if len(np.unique(self.y)) > 1:
            fpr, tpr, _ = roc_curve(self.y, y_cont)
            roc_auc = auc(fpr, tpr)
            print(f"  AUC-ROC: {roc_auc:.4f}")
        else:
            roc_auc = 0
            print("  No se puede calcular ROC (solo una clase presente)")
        
        # Preparar resultados
        resultados_eval = {
            'metricas': {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1,
                'auc': roc_auc
            },
            'predicciones': {'y_cont': y_cont, 'y_pred': y_pred},
            'matriz_confusion': cm,
            'reporte_clasificacion': reporte
        }
        
        # Graficos - CORREGIDO: Usar sistema de archivos del proyecto
        if save_plots:
            fig = self._generar_graficos_evaluacion(cm, fpr, tpr, roc_auc, y_cont, y_pred)
            
        # Guardar metricas en cache si esta configurado
        if config.cache.usar_cache_resultados:
            sistema_cache.guardar_metricas(
                config.entrenamiento.nombre_modelo, 
                resultados_eval['metricas']
            )
        
        return resultados_eval
    
    def _generar_graficos_evaluacion(self, cm, fpr, tpr, roc_auc, y_cont, y_pred):
        """Genera graficos de evaluacion completos - CORREGIDO"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matriz de Confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        ax1.set_title('Matriz de Confusion', fontweight='bold')
        ax1.set_ylabel('Valor Real')
        ax1.set_xlabel('Prediccion')
        
        # 2. Curva ROC (si hay mas de una clase)
        if roc_auc > 0:
            ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Tasa de Falsos Positivos')
            ax2.set_ylabel('Tasa de Verdaderos Positivos')
            ax2.set_title('Curva ROC', fontweight='bold')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No disponible\n(solo una clase)', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Curva ROC (No disponible)', fontweight='bold')
        
        # 3. Distribucion de predicciones
        if len(np.unique(self.y)) > 1:
            ax3.hist(y_cont[self.y == 0], alpha=0.7, label='No Tumor', bins=20, color='blue')
            ax3.hist(y_cont[self.y == 1], alpha=0.7, label='Tumor', bins=20, color='red')
        else:
            ax3.hist(y_cont, alpha=0.7, label='Todas las muestras', bins=20, color='gray')
        ax3.axvline(x=0.5, color='black', linestyle='--', label='Umbral')
        ax3.set_xlabel('Valor de Prediccion')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribucion de Predicciones', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metricas de rendimiento
        if cm.size >= 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        else:
            sensitivity = specificity = precision = f1 = 0
        
        metricas = ['Sensibilidad', 'Especificidad', 'Precision', 'F1-Score']
        valores = [sensitivity, specificity, precision, f1]
        
        bars = ax4.bar(metricas, valores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Valor')
        ax4.set_title('Metricas de Rendimiento', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Anadir valores en las barras
        for bar, valor in zip(bars, valores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{valor:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # CORRECCIÓN: Guardar en el sistema de caché en lugar de directorio actual
        if config.analisis.guardar_graficos:
            # Guardar en directorio de análisis
            os.makedirs(config.analisis.directorio_analisis, exist_ok=True)
            ruta_analisis = os.path.join(config.analisis.directorio_analisis, "evaluacion_completa_modelo.png")
            plt.savefig(ruta_analisis, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico de evaluación guardado en: {ruta_analisis}")
            
            # También guardar en caché si está configurado
            if config.cache.usar_cache_resultados:
                sistema_cache.guardar_grafico("evaluacion_completa", fig)
        
        plt.show()
        
        return fig