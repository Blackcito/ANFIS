import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from core.prediction import predict_sugeno

class EvaluadorANFIS:
    def __init__(self, modelo, datos):
        self.mf_params = modelo['mf_params']
        self.theta = modelo['theta']
        self.X = datos['X']
        self.y = datos['y']
        self.nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                                       'Energía', 'Media', 'Entropía', 'Varianza']
    
    def evaluar_modelo(self, save_plots=True):
        """Evaluación completa del modelo con gráficos ROC"""
        print("\n" + "="*50)
        print("EVALUACIÓN DEL MODELO ANFIS")
        print("="*50)
        
        # Predicciones
        y_cont, y_pred = predict_sugeno(self.X, self.mf_params, self.theta)
        
        # Métricas básicas
        print("\n📊 REPORTE DE CLASIFICACIÓN:")
        print("-" * 40)
        reporte = classification_report(self.y, y_pred, target_names=['No Tumor', 'Tumor'], digits=4)
        print(reporte)
        
        print("\n🎯 MATRIZ DE CONFUSIÓN:")
        print("-" * 30)
        cm = confusion_matrix(self.y, y_pred)
        print(cm)
        
        # Métricas detalladas
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\n📈 MÉTRICAS DETALLADAS:")
        print(f"  Sensibilidad (Recall): {sensitivity:.4f}")
        print(f"  Especificidad: {specificity:.4f}")
        print(f"  Precisión: {precision:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Curva ROC
        if len(np.unique(self.y)) > 1:
            fpr, tpr, _ = roc_curve(self.y, y_cont)
            roc_auc = auc(fpr, tpr)
            print(f"  AUC-ROC: {roc_auc:.4f}")
        else:
            roc_auc = 0
            print("  ⚠️  No se puede calcular ROC (solo una clase presente)")
        
        # Gráficos
        if save_plots:
            self._generar_graficos_evaluacion(cm, fpr, tpr, roc_auc, y_cont, y_pred)
        
        return {
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
    
    def _generar_graficos_evaluacion(self, cm, fpr, tpr, roc_auc, y_cont, y_pred):
        """Genera gráficos de evaluación completos"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matriz de Confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        ax1.set_title('Matriz de Confusión', fontweight='bold')
        ax1.set_ylabel('Valor Real')
        ax1.set_xlabel('Predicción')
        
        # 2. Curva ROC (si hay más de una clase)
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
        
        # 3. Distribución de predicciones
        # Verificar que hay datos para ambas clases
        if len(np.unique(self.y)) > 1:
            ax3.hist(y_cont[self.y == 0], alpha=0.7, label='No Tumor', bins=20, color='blue')
            ax3.hist(y_cont[self.y == 1], alpha=0.7, label='Tumor', bins=20, color='red')
        else:
            ax3.hist(y_cont, alpha=0.7, label='Todas las muestras', bins=20, color='gray')
        ax3.axvline(x=0.5, color='black', linestyle='--', label='Umbral')
        ax3.set_xlabel('Valor de Predicción')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Predicciones', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Métricas de rendimiento - CALCULAR DESDE MATRIZ DE CONFUSIÓN
        if cm.size >= 4:  # Matriz 2x2
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        else:
            # Caso donde solo hay una clase
            sensitivity = specificity = precision = f1 = 0
        
        metricas = ['Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score']
        valores = [sensitivity, specificity, precision, f1]
        
        bars = ax4.bar(metricas, valores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Valor')
        ax4.set_title('Métricas de Rendimiento', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, valores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{valor:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('evaluacion_completa_modelo.png', dpi=300, bbox_inches='tight')
        plt.show()