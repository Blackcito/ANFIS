# predict_sugeno_con_analisis.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from procesamiento_image import process_all_images
from Training_ANFIS import train_anfis
from anfis_sugeno import compute_weights, n_vars, reglas
from analisis import AnalizadorReglasANFIS
import os
from datetime import datetime

def crear_carpeta_resultados():
    """Crea una carpeta única para guardar todos los resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = f"Resultados_ANFIS_{timestamp}"
    os.makedirs(carpeta, exist_ok=True)
    print(f"\n Todos los resultados se guardarán en: {carpeta}")
    return carpeta



def predict_sugeno(X, mf_params, theta, threshold=0.5):
    """
    Devuelve las salidas continuas y binarias (umbral > threshold)
    según el ANFIS-Sugeno entrenado.
    """
    y_cont = []
    for x in X:
        w_bar = compute_weights(x, mf_params)
        total = 0.0
        for j in range(len(reglas)):
            p_j = theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
            r_j = theta[j*(n_vars+1) + n_vars]
            total += w_bar[j] * (p_j.dot(x) + r_j)
        y_cont.append(total)
    y_cont = np.array(y_cont)
    y_bin = (y_cont > threshold).astype(int)
    return y_cont, y_bin

def predict_con_explicacion(x_sample, mf_params, theta, n_top_reglas=5):
    """
    Predice una muestra individual y explica qué reglas fueron más activas
    """
    w_bar = compute_weights(x_sample, mf_params)
    
    # Calcular contribuciones individuales
    contribuciones = []
    total = 0.0
    
    for j in range(len(reglas)):
        p_j = theta[j*(n_vars+1) : j*(n_vars+1)+n_vars]
        r_j = theta[j*(n_vars+1) + n_vars]
        contribucion = w_bar[j] * (p_j.dot(x_sample) + r_j)
        total += contribucion
        
        contribuciones.append({
            'regla_idx': j,
            'regla': reglas[j],
            'activacion': w_bar[j],
            'contribucion': contribucion,
            'parametros': p_j,
            'bias': r_j
        })
    
    # Ordenar por contribución absoluta
    contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
    
    prediction = (total > 0.5).astype(int)
    
    return {
        'prediccion_continua': total,
        'prediccion_binaria': prediction,
        'top_reglas_activas': contribuciones[:n_top_reglas],
        'todas_contribuciones': contribuciones
    }

def generar_reporte_completo(X_train, y_train, mf_opt, theta_opt, save_plots=True):
    """
    Genera reporte completo con evaluación del modelo y análisis de reglas
    """
    print("="*60)
    print("REPORTE COMPLETO - ANFIS PARA DETECCIÓN DE TUMORES CEREBRALES")
    print("="*60)
    
    # 1. Predicciones del modelo
    y_cont, y_pred = predict_sugeno(X_train, mf_opt, theta_opt)
    
    # 2. Métricas de evaluación
    print("\n MÉTRICAS DE EVALUACIÓN:")
    print("-" * 30)
    print("\nReporte de Clasificación:")
    print(classification_report(y_train, y_pred, 
                              target_names=['No Tumor', 'Tumor'],
                              digits=4))
    
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    
    # Métricas adicionales
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # Sensibilidad
    specificity = tn / (tn + fp)  # Especificidad
    precision = tp / (tp + fp)    # Precisión
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    print(f"\nMétricas Detalladas:")
    print(f"Sensibilidad (Recall): {sensitivity:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # 3. Curva ROC
    if save_plots:
        fpr, tpr, _ = roc_curve(y_train, y_cont)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Curva ROC
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC - ANFIS')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Matriz de confusión
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Valores Reales')
        plt.xlabel('Predicciones')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('evaluacion_modelo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Análisis de reglas ANFIS
    print("\n ANÁLISIS DE REGLAS ANFIS:")
    print("-" * 30)
    
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_train, y_train)
    
    # Generar análisis completo
    if save_plots:
        resultados_analisis = analizador.generar_analisis_completo("analisis_reglas_anfis")
    else:
        analizador.calcular_importancia_reglas()
        resultados_analisis = {'top_reglas': analizador.obtener_top_reglas(10)}
    
    # Mostrar top 5 reglas más importantes
    print("\nTop 5 Reglas Más Importantes:")
    nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                              'Energía', 'Media', 'Entropía', 'Varianza']
    
    for i, regla in enumerate(resultados_analisis['top_reglas'][:5]):
        print(f"\n🔸 REGLA #{i+1} (Índice: {regla['regla_idx']})")
        condiciones = []
        for j, etiqueta in enumerate(regla['regla_condicion']):
            condiciones.append(f"{nombres_caracteristicas[j]} es {etiqueta}")
        print(f"   Condición: SI " + " Y ".join(condiciones))
        print(f"   Activación media: {regla['activacion_media']:.4f}")
        print(f"   Importancia: {regla['importancia_total']:.4f}")
    
    # 5. Ejemplo de predicción explicada
    print("\n EJEMPLO DE PREDICCIÓN EXPLICADA:")
    print("-" * 35)
    
    # Tomar una muestra aleatoria
    idx_ejemplo = np.random.randint(0, len(X_train))
    muestra_ejemplo = X_train[idx_ejemplo]
    etiqueta_real = y_train[idx_ejemplo]
    
    explicacion = predict_con_explicacion(muestra_ejemplo, mf_opt, theta_opt, n_top_reglas=3)
    
    print(f"Muestra #{idx_ejemplo}:")
    print(f"Etiqueta real: {'Tumor' if etiqueta_real == 1 else 'No Tumor'}")
    print(f"Predicción: {'Tumor' if explicacion['prediccion_binaria'] == 1 else 'No Tumor'}")
    print(f"Confianza: {explicacion['prediccion_continua']:.4f}")
    
    print("\nTop 3 reglas más activas para esta predicción:")
    for i, regla_activa in enumerate(explicacion['top_reglas_activas']):
        condiciones = []
        for j, etiqueta in enumerate(reglas[regla_activa['regla_idx']]):
            condiciones.append(f"{nombres_caracteristicas[j]}={etiqueta}")
        print(f"  {i+1}. Regla {regla_activa['regla_idx']}: {' & '.join(condiciones)}")
        print(f"     Activación: {regla_activa['activacion']:.4f}, "
              f"Contribución: {regla_activa['contribucion']:.4f}")
    
    return {
        'metricas': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc': roc_auc if save_plots else None
        },
        'predicciones': {'y_cont': y_cont, 'y_pred': y_pred},
        'analisis_reglas': resultados_analisis,
        'ejemplo_explicacion': explicacion
    }

def analizar_casos_especificos(X_data, y_data, mf_opt, theta_opt, 
                              casos_indices=None, n_casos=5):
    """
    Analiza casos específicos mostrando qué reglas los clasificaron
    """
    if casos_indices is None:
        # Seleccionar algunos casos aleatorios
        casos_indices = np.random.choice(len(X_data), n_casos, replace=False)
    
    print("\n🔬 ANÁLISIS DE CASOS ESPECÍFICOS:")
    print("-" * 40)
    
    nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                              'Energía', 'Media', 'Entropía', 'Varianza']
    
    casos_analizados = []
    
    for i, idx in enumerate(casos_indices):
        muestra = X_data[idx]
        etiqueta_real = y_data[idx]
        
        explicacion = predict_con_explicacion(muestra, mf_opt, theta_opt, n_top_reglas=3)
        
        print(f"\n CASO #{i+1} (Muestra #{idx}):")
        print(f"Etiqueta real: {' Tumor' if etiqueta_real == 1 else ' No Tumor'}")
        print(f"Predicción: {' Tumor' if explicacion['prediccion_binaria'] == 1 else ' No Tumor'}")
        print(f"Confianza: {explicacion['prediccion_continua']:.4f}")
        
        # Mostrar valores de características
        print("Características GLCM:")
        for j, (nombre, valor) in enumerate(zip(nombres_caracteristicas, muestra)):
            print(f"  {nombre}: {valor:.4f}")
        
        # Reglas más activas
        print("Reglas más influyentes:")
        for j, regla_activa in enumerate(explicacion['top_reglas_activas']):
            condiciones = []
            for k, etiqueta in enumerate(reglas[regla_activa['regla_idx']]):
                condiciones.append(f"{nombres_caracteristicas[k]}={etiqueta}")
            print(f"  {j+1}. R{regla_activa['regla_idx']}: {' & '.join(condiciones)}")
            print(f"     Act: {regla_activa['activacion']:.3f}, Contrib: {regla_activa['contribucion']:.3f}")
        
        casos_analizados.append({
            'indice': idx,
            'etiqueta_real': etiqueta_real,
            'prediccion': explicacion['prediccion_binaria'],
            'confianza': explicacion['prediccion_continua'],
            'caracteristicas': muestra,
            'top_reglas': explicacion['top_reglas_activas']
        })
    
    return casos_analizados

def comparar_predicciones_por_clase(X_data, y_data, mf_opt, theta_opt):
    """
    Compara qué reglas son más activas para cada clase (tumor vs no tumor)
    """
    print("\n COMPARACIÓN DE REGLAS POR CLASE:")
    print("-" * 40)
    
    # Separar datos por clase
    indices_no_tumor = np.where(y_data == 0)[0]
    indices_tumor = np.where(y_data == 1)[0]
    
    # Calcular activaciones promedio por clase
    activaciones_no_tumor = np.zeros(len(reglas))
    activaciones_tumor = np.zeros(len(reglas))
    
    # Para casos sin tumor
    for idx in indices_no_tumor:
        w = compute_weights(X_data[idx], mf_opt)
        activaciones_no_tumor += w
    activaciones_no_tumor /= len(indices_no_tumor)
    
    # Para casos con tumor
    for idx in indices_tumor:
        w = compute_weights(X_data[idx], mf_opt)
        activaciones_tumor += w
    activaciones_tumor /= len(indices_tumor)
    
    # Encontrar reglas más discriminativas
    diferencia_activaciones = activaciones_tumor - activaciones_no_tumor
    
    # Top reglas para cada clase
    top_tumor_indices = np.argsort(diferencia_activaciones)[-10:][::-1]
    top_no_tumor_indices = np.argsort(diferencia_activaciones)[:10]
    
    nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                              'Energía', 'Media', 'Entropía', 'Varianza']
    
    print("\n Top 5 reglas más activas para TUMOR:")
    for i, regla_idx in enumerate(top_tumor_indices[:5]):
        condiciones = []
        for j, etiqueta in enumerate(reglas[regla_idx]):
            condiciones.append(f"{nombres_caracteristicas[j]}={etiqueta}")
        print(f"  {i+1}. R{regla_idx}: {' & '.join(condiciones)}")
        print(f"     Activación tumor: {activaciones_tumor[regla_idx]:.4f}")
        print(f"     Activación no-tumor: {activaciones_no_tumor[regla_idx]:.4f}")
        print(f"     Diferencia: +{diferencia_activaciones[regla_idx]:.4f}")
    
    print("\n Top 5 reglas más activas para NO TUMOR:")
    for i, regla_idx in enumerate(top_no_tumor_indices[:5]):
        condiciones = []
        for j, etiqueta in enumerate(reglas[regla_idx]):
            condiciones.append(f"{nombres_caracteristicas[j]}={etiqueta}")
        print(f"  {i+1}. R{regla_idx}: {' & '.join(condiciones)}")
        print(f"     Activación no-tumor: {activaciones_no_tumor[regla_idx]:.4f}")
        print(f"     Activación tumor: {activaciones_tumor[regla_idx]:.4f}")
        print(f"     Diferencia: {diferencia_activaciones[regla_idx]:.4f}")
    
    return {
        'activaciones_no_tumor': activaciones_no_tumor,
        'activaciones_tumor': activaciones_tumor,
        'diferencia_activaciones': diferencia_activaciones,
        'top_reglas_tumor': top_tumor_indices[:10],
        'top_reglas_no_tumor': top_no_tumor_indices[:10]
    }

def crear_visualizacion_reglas_discriminativas(comparacion_clases, save_path=None):
    """
    Crea visualización de las reglas más discriminativas entre clases
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Diferencia de activaciones
    top_discriminativas = np.argsort(np.abs(comparacion_clases['diferencia_activaciones']))[-15:]
    diferencias = comparacion_clases['diferencia_activaciones'][top_discriminativas]
    labels = [f'R{i}' for i in top_discriminativas]
    
    colors = ['red' if d > 0 else 'green' for d in diferencias]
    bars1 = ax1.barh(labels, diferencias, color=colors, alpha=0.7)
    ax1.set_title('Top 15 Reglas Más Discriminativas\n(Rojo=Tumor, Verde=No Tumor)', 
                  fontweight='bold')
    ax1.set_xlabel('Diferencia de Activación (Tumor - No Tumor)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Añadir valores
    for bar, val in zip(bars1, diferencias):
        ax1.text(val + (0.001 if val > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=8)
    
    # Gráfico 2: Activaciones por clase para top reglas
    top_5_tumor = comparacion_clases['top_reglas_tumor'][:5]
    top_5_no_tumor = comparacion_clases['top_reglas_no_tumor'][:5]
    
    reglas_mostrar = np.concatenate([top_5_tumor, top_5_no_tumor])
    labels_reglas = [f'R{i}' for i in reglas_mostrar]
    
    act_tumor = comparacion_clases['activaciones_tumor'][reglas_mostrar]
    act_no_tumor = comparacion_clases['activaciones_no_tumor'][reglas_mostrar]
    
    x = np.arange(len(labels_reglas))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, act_tumor, width, label='Tumor', color='red', alpha=0.7)
    bars3 = ax2.bar(x + width/2, act_no_tumor, width, label='No Tumor', color='green', alpha=0.7)
    
    ax2.set_title('Activación de Reglas por Clase\n(Top 5 de cada clase)', fontweight='bold')
    ax2.set_xlabel('Reglas')
    ax2.set_ylabel('Activación Promedio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_reglas, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()
    return fig

def ejecutar_analisis_completo_mejorado():
    """
    Función principal que ejecuta todo el análisis completo mejorado
    """
    print(" INICIANDO ANÁLISIS COMPLETO DE ANFIS")
    print("=" * 50)
    
    # 1. Preprocesar imágenes y extraer características
    print(" Extrayendo características GLCM...")
    X_train, y_train = process_all_images()
    
    # 2. Entrenar el modelo ANFIS-Sugeno
    print(" Entrenando modelo ANFIS...")
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=15)
    
    # 3. Generar reporte completo con análisis de reglas
    print(" Generando reporte completo...")
    resultados = generar_reporte_completo(X_train, y_train, mf_opt, theta_opt, save_plots=True)
    
    # 4. Analizar casos específicos
    print(" Analizando casos específicos...")
    casos_especificos = analizar_casos_especificos(X_train, y_train, mf_opt, theta_opt, n_casos=5)
    
    # 5. Comparar reglas por clase
    print(" Comparando reglas por clase...")
    comparacion = comparar_predicciones_por_clase(X_train, y_train, mf_opt, theta_opt)
    
  
    
    # 7. Resumen final
    print("\n" + "="*60)
    print(" ANÁLISIS COMPLETO FINALIZADO")
    print("="*60)
    
    print(f" Precisión del modelo: {resultados['metricas']['precision']:.4f}")
    print(f" Sensibilidad: {resultados['metricas']['sensitivity']:.4f}")
    print(f" Especificidad: {resultados['metricas']['specificity']:.4f}")
    print(f" F1-Score: {resultados['metricas']['f1_score']:.4f}")
    if resultados['metricas']['auc']:
        print(f" AUC-ROC: {resultados['metricas']['auc']:.4f}")
    
    print(f"\n Archivos generados:")
    print("  - analisis_reglas_anfis/ (carpeta con análisis detallado)")
    print("  - evaluacion_modelo.png (métricas del modelo)")
    print("  - reglas_discriminativas.png (reglas más importantes)")
    
    print(f"\n Top 3 reglas más importantes globalmente:")
    for i, regla in enumerate(resultados['analisis_reglas']['top_reglas'][:3]):
        print(f"  {i+1}. Regla {regla['regla_idx']} - Importancia: {regla['importancia_total']:.4f}")
    
    return {
        'modelo': {'mf_params': mf_opt, 'theta': theta_opt},
        'datos': {'X': X_train, 'y': y_train},
        'resultados': resultados,
        'casos_especificos': casos_especificos,
        'comparacion_clases': comparacion
    }

if __name__ == "__main__":
    # Ejecutar análisis completo
    resultados_completos = ejecutar_analisis_completo_mejorado()
    
    print("\n ¡Análisis terminado! Revisa los archivos generados para más detalles.")
    print("\nPara análisis adicionales, puedes usar:")
    print("- resultados_completos['modelo'] para acceder al modelo entrenado")
    print("- resultados_completos['resultados']['analisis_reglas'] para datos de reglas")
    print("- resultados_completos['comparacion_clases'] para análisis por clase")