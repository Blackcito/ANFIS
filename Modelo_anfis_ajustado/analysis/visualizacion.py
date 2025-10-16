import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_evaluacion_modelo(y_true, y_pred, y_cont, save_path=None):
    """Genera gráficos de evaluación del modelo"""
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_cont)
    roc_auc = auc(fpr, tpr)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Curva ROC
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Tasa de Falsos Positivos')
    ax1.set_ylabel('Tasa de Verdaderos Positivos')
    ax1.set_title('Curva ROC - ANFIS')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Tumor', 'Tumor'],
               yticklabels=['No Tumor', 'Tumor'], ax=ax2)
    ax2.set_title('Matriz de Confusión')
    ax2.set_ylabel('Valores Reales')
    ax2.set_xlabel('Predicciones')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_reglas_discriminativas(comparacion_clases, save_path=None):
    """Visualización de reglas discriminativas"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Diferencia de activaciones
    top_discriminativas = np.argsort(np.abs(comparacion_clases['diferencia_activaciones']))[-15:]
    diferencias = comparacion_clases['diferencia_activaciones'][top_discriminativas]
    labels = [f'R{i}' for i in top_discriminativas]
    
    colors = ['red' if d > 0 else 'green' for d in diferencias]
    bars1 = ax1.barh(labels, diferencias, color=colors, alpha=0.7)
    ax1.set_title('Top 15 Reglas Más Discriminativas\n(Rojo=Tumor, Verde=No Tumor)')
    ax1.set_xlabel('Diferencia de Activación (Tumor - No Tumor)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    for bar, val in zip(bars1, diferencias):
        ax1.text(val + (0.001 if val > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=8)
    
    # Gráfico 2: Activaciones por clase
    top_5_tumor = comparacion_clases['top_reglas_tumor'][:5]
    top_5_no_tumor = comparacion_clases['top_reglas_no_tumor'][:5]
    
    reglas_mostrar = np.concatenate([top_5_tumor, top_5_no_tumor])
    labels_reglas = [f'R{i}' for i in reglas_mostrar]
    
    act_tumor = comparacion_clases['activaciones_tumor'][reglas_mostrar]
    act_no_tumor = comparacion_clases['activaciones_no_tumor'][reglas_mostrar]
    
    x = np.arange(len(labels_reglas))
    width = 0.35
    
    ax2.bar(x - width/2, act_tumor, width, label='Tumor', color='red', alpha=0.7)
    ax2.bar(x + width/2, act_no_tumor, width, label='No Tumor', color='green', alpha=0.7)
    
    ax2.set_title('Activación de Reglas por Clase\n(Top 5 de cada clase)')
    ax2.set_xlabel('Reglas')
    ax2.set_ylabel('Activación Promedio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_reglas, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig