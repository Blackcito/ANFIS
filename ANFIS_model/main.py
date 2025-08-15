# ejemplo_uso.py - Integración simple con tu código existente

"""
Este script muestra cómo integrar fácilmente el análisis de reglas 
con tu código ANFIS existente.
"""

import numpy as np
from procesamiento_image import process_all_images
from Training_ANFIS import train_anfis
from analisis import AnalizadorReglasANFIS
from prediccion2 import generar_reporte_completo, ejecutar_analisis_completo_mejorado

def ejemplo_basico():
    """
    Ejemplo básico: Solo análisis de reglas más importantes
    """
    print("🔍 EJEMPLO BÁSICO - ANÁLISIS DE REGLAS")
    print("-" * 45)
    
    # 1. Obtener datos y entrenar modelo (tu código existente)
    X_train, y_train = process_all_images()
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    # 2. Crear analizador de reglas
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_train, y_train)
    
    # 3. Obtener top 10 reglas más importantes
    top_reglas = analizador.obtener_top_reglas(n_top=10)
    
    # 4. Mostrar resultados
    nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                              'Energía', 'Media', 'Entropía', 'Varianza']
    
    print(f"\n📊 TOP 10 REGLAS MÁS IMPORTANTES:")
    for i, regla in enumerate(top_reglas):
        condiciones = []
        for j, etiqueta in enumerate(regla['regla_condicion']):
            condiciones.append(f"{nombres_caracteristicas[j]}={etiqueta}")
        
        print(f"\n🔸 #{i+1} - Regla {regla['regla_idx']}")
        print(f"   Condición: {' & '.join(condiciones)}")
        print(f"   Importancia: {regla['importancia_total']:.4f}")
        print(f"   Activación media: {regla['activacion_media']:.4f}")

def ejemplo_con_graficos():
    """
    Ejemplo con análisis completo y gráficos
    """
    print("📈 EJEMPLO CON GRÁFICOS - ANÁLISIS COMPLETO")
    print("-" * 50)
    
    # 1. Obtener datos y entrenar
    X_train, y_train = process_all_images()
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    # 2. Análisis completo con gráficos
    analizador = AnalizadorReglasANFIS(mf_opt, theta_opt, X_train, y_train)
    
    # Generar todos los análisis y gráficos
    resultados = analizador.generar_analisis_completo("mi_analisis_anfis")
    
    print(f"✅ Análisis completo guardado en: mi_analisis_anfis/")
    print(f"📊 Archivos generados:")
    print(f"   - reporte_reglas.txt")
    print(f"   - importancia_reglas.png")
    print(f"   - mapa_calor_reglas.png") 
    print(f"   - importancia_caracteristicas.png")
    print(f"   - datos_reglas.csv")
    
    return resultados

def ejemplo_prediccion_explicada():
    """
    Ejemplo de cómo explicar predicciones individuales
    """
    print("🔎 EJEMPLO - PREDICCIÓN EXPLICADA")
    print("-" * 35)
    
    from prediccion2 import predict_con_explicacion
    
    # 1. Entrenar modelo
    X_train, y_train = process_all_images()
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    # 2. Tomar una muestra para explicar
    idx_muestra = 42  # Puedes cambiar este índice
    muestra = X_train[idx_muestra]
    etiqueta_real = y_train[idx_muestra]
    
    # 3. Generar explicación
    explicacion = predict_con_explicacion(muestra, mf_opt, theta_opt, n_top_reglas=5)
    
    # 4. Mostrar explicación
    nombres_caracteristicas = ['Contraste', 'ASM', 'Homogeneidad', 
                              'Energía', 'Media', 'Entropía', 'Varianza']
    
    print(f"\n📋 EXPLICACIÓN PARA MUESTRA #{idx_muestra}:")
    print(f"Etiqueta real: {'🔴 Tumor' if etiqueta_real == 1 else '🟢 No Tumor'}")
    print(f"Predicción: {'🔴 Tumor' if explicacion['prediccion_binaria'] == 1 else '🟢 No Tumor'}")
    print(f"Confianza: {explicacion['prediccion_continua']:.4f}")
    
    print(f"\n📊 Valores de características GLCM:")
    for nombre, valor in zip(nombres_caracteristicas, muestra):
        print(f"   {nombre}: {valor:.4f}")
    
    print(f"\n🔍 Top 5 reglas que influyeron en esta decisión:")
    for i, regla_activa in enumerate(explicacion['top_reglas_activas']):
        condiciones = []
        regla_idx = regla_activa['regla_idx']
        from anfis_sugeno import reglas
        
        for j, etiqueta in enumerate(reglas[regla_idx]):
            condiciones.append(f"{nombres_caracteristicas[j]}={etiqueta}")
        
        print(f"   {i+1}. Regla {regla_idx}: {' & '.join(condiciones)}")
        print(f"      Activación: {regla_activa['activacion']:.4f}")
        print(f"      Contribución a salida: {regla_activa['contribucion']:.4f}")

def ejemplo_comparacion_clases():
    """
    Ejemplo de análisis comparativo entre clases (tumor vs no tumor)
    """
    print("⚖️ EJEMPLO - COMPARACIÓN ENTRE CLASES")
    print("-" * 40)
    
    from prediccion2 import comparar_predicciones_por_clase
    
    # 1. Entrenar modelo
    X_train, y_train = process_all_images()
    mf_opt, theta_opt = train_anfis(X_train, y_train, swarmsize=30, maxiter=10)
    
    # 2. Comparar reglas por clase
    comparacion = comparar_predicciones_por_clase(X_train, y_train, mf_opt, theta_opt)
    
    print("✅ Análisis comparativo completado!")
    print("Este análisis muestra qué reglas son más activas para cada clase.")

def menu_interactivo():
    """
    Menú interactivo para elegir qué análisis ejecutar
    """
    print("\n" + "="*60)
    print("🧠 SISTEMA DE ANÁLISIS ANFIS - DETECCIÓN DE TUMORES")
    print("="*60)
    print("\nSelecciona el tipo de análisis que deseas ejecutar:")
    print("\n1. 🔍 Análisis básico (solo top reglas importantes)")
    print("2. 📈 Análisis completo con gráficos")
    print("3. 🔎 Explicación de predicción individual")
    print("4. ⚖️ Comparación de reglas por clase")
    print("5. 🚀 Análisis completo mejorado (todo incluido)")
    print("6. ❌ Salir")
    
    try:
        opcion = input("\nIngresa tu opción (1-6): ").strip()
        
        if opcion == "1":
            ejemplo_basico()
        elif opcion == "2":
            ejemplo_con_graficos()
        elif opcion == "3":
            ejemplo_prediccion_explicada()
        elif opcion == "4":
            ejemplo_comparacion_clases()
        elif opcion == "5":
            ejecutar_analisis_completo_mejorado()
        elif opcion == "6":
            print("👋 ¡Hasta luego!")
            return
        else:
            print("❌ Opción no válida. Intenta de nuevo.")
            menu_interactivo()
            
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Intenta de nuevo.")
        menu_interactivo()

# Script principal
if __name__ == "__main__":
    # Puedes ejecutar ejemplos individuales o el menú interactivo
    
    # Opción 1: Ejecutar un ejemplo específico
    # ejemplo_basico()
    # ejemplo_con_graficos()
    # ejemplo_prediccion_explicada()
    
    # Opción 2: Menú interactivo (recomendado)
    menu_interactivo()
    
    # Opción 3: Análisis completo directo
    # ejecutar_analisis_completo_mejorado()