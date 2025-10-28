# core/pipeline_anfis.py - MEJORADO

import numpy as np
from core.training import train_anfis
from analysis.evaluador import EvaluadorANFIS
from analysis.analisis import AnalizadorReglasANFIS
from config.configuracion import config
from utils.cache import sistema_cache
from core.gestor_datos import gestor_datos

class PipelineANFIS:
    """Pipeline principal con gestión inteligente de opciones"""
    
    def __init__(self):
        self.modelo_actual = None
        self.opciones_actuales = {}
    
    def ejecutar(self, **opciones_personalizadas):
        """
        Ejecuta el pipeline con opciones personalizables
        """
        self.opciones_actuales = self._combinar_opciones(opciones_personalizadas)
        
        print(" Iniciando pipeline ANFIS...")
        self._mostrar_opciones()
        
        # 1. Cargar datos según necesidades
        datos = self._cargar_datos_inteligente()
        if datos is None:
            return None
        
        # 2. Gestionar modelo
        modelo = self._gestionar_modelo(datos)
        if modelo is None:
            return None
        
        # 3. Evaluar y analizar
        resultados = self._evaluar_y_analizar(modelo, datos)
        
        # 4. Resumen
        self._mostrar_resumen(resultados)
        return resultados
    
    def _combinar_opciones(self, opciones_personalizadas):
        """Combina opciones personalizadas con configuración global"""
        opciones = {
            'entrenar_nuevo': False,
            'nombre_modelo': None,
            'forzar_reprocesamiento': False,
            # Por defecto NO usar cache a menos que se especifique
            'usar_cache_entrenamiento': False,
            'usar_cache_prueba': False,
        }
        
        # Actualizar con opciones personalizadas
        opciones.update(opciones_personalizadas)
        return opciones
    
    def _mostrar_opciones(self):
        """Muestra las opciones actuales del pipeline"""
        print(" Opciones del pipeline:")
        print(f"   - Entrenar nuevo: {self.opciones_actuales['entrenar_nuevo']}")
        print(f"   - Cache entrenamiento: {self.opciones_actuales['usar_cache_entrenamiento']}")
        print(f"   - Cache prueba: {self.opciones_actuales['usar_cache_prueba']}")
        print(f"   - Forzar reprocesamiento: {self.opciones_actuales['forzar_reprocesamiento']}")
    
    def _cargar_datos_inteligente(self):
        """Carga solo los datos necesarios según las opciones - ACTUALIZADO"""
        if self.opciones_actuales['entrenar_nuevo']:
            # Necesitamos ambos conjuntos
            print(" Cargando datos de entrenamiento y prueba...")
            X_train, y_train = gestor_datos.cargar_datos_entrenamiento(
                tumor_dir=self.opciones_actuales.get('train_tumor_dir'),
                notumor_dir=self.opciones_actuales.get('train_notumor_dir'),
                usar_cache=self.opciones_actuales['usar_cache_entrenamiento'],
                cache_especifico=self.opciones_actuales.get('cache_entrenamiento_especifico'),
                forzar_reprocesamiento=self.opciones_actuales['forzar_reprocesamiento']
            )
            X_test, y_test = gestor_datos.cargar_datos_prueba(
                tumor_dir=self.opciones_actuales.get('test_tumor_dir'),
                notumor_dir=self.opciones_actuales.get('test_notumor_dir'),
                usar_cache=self.opciones_actuales['usar_cache_prueba'],
                cache_especifico=self.opciones_actuales.get('cache_prueba_especifico'),
                forzar_reprocesamiento=self.opciones_actuales['forzar_reprocesamiento']
            )
            return {'train': (X_train, y_train), 'test': (X_test, y_test)}
        else:
            # Solo necesitamos datos de prueba
            print(" Cargando solo datos de prueba...")
            X_test, y_test = gestor_datos.cargar_datos_prueba(
                tumor_dir=self.opciones_actuales.get('test_tumor_dir'),
                notumor_dir=self.opciones_actuales.get('test_notumor_dir'),
                usar_cache=self.opciones_actuales['usar_cache_prueba'],
                cache_especifico=self.opciones_actuales.get('cache_prueba_especifico'),
                forzar_reprocesamiento=self.opciones_actuales['forzar_reprocesamiento']
            )
            return {'train': (np.array([]), np.array([])), 'test': (X_test, y_test)}
    
    def _gestionar_modelo(self, datos):
        """Gestiona la obtención del modelo"""
        X_train, y_train = datos['train']
        
        if self.opciones_actuales['entrenar_nuevo']:
            if len(X_train) == 0:
                print(" No hay datos de entrenamiento")
                return None
            
            print(" Entrenando nuevo modelo...")
            return self._entrenar_modelo(X_train, y_train)
        else:
            print(" Cargando modelo existente...")
            modelo = self._cargar_modelo_existente()
            
            if modelo is None and len(X_train) > 0:
                print(" Modelo no encontrado, entrenando nuevo...")
                return self._entrenar_modelo(X_train, y_train)
            
            return modelo
    
    def _entrenar_modelo(self, X_train, y_train):
        """Entrena un nuevo modelo"""
        mf_opt, theta_opt = train_anfis(
            X_train, y_train,
            swarmsize=config.entrenamiento.tamano_enjambre,
            maxiter=config.entrenamiento.max_iteraciones,
            guardar_modelo=config.entrenamiento.guardar_modelo,
            nombre_modelo=config.entrenamiento.nombre_modelo
        )
        return {'mf_params': mf_opt, 'theta': theta_opt}
    
    def _cargar_modelo_existente(self):
        """Carga un modelo existente"""
        mf_opt, theta_opt, metadatos = sistema_cache.cargar_modelo(
            nombre_modelo=self.opciones_actuales['nombre_modelo']
        )
        
        if mf_opt is not None:
            return {'mf_params': mf_opt, 'theta': theta_opt}
        return None
    
    def _evaluar_y_analizar(self, modelo, datos):
        """Evalúa el modelo y genera análisis"""
        X_test, y_test = datos['test']
        resultados = {}
        visualizar_graficos = self.opciones_actuales.get('visualizar_graficos', True)
        if len(X_test) > 0:
            print(" Evaluando modelo...")
            evaluador = EvaluadorANFIS(modelo, {'X': X_test, 'y': y_test})
            # Controlar si se deben guardar/mostrar los gráficos de evaluación
            resultados_eval = evaluador.evaluar_modelo(
                guardar_graficos=self.opciones_actuales.get('guardar_graficos', True),
                visualizar_graficos=visualizar_graficos
            )
            resultados['evaluacion'] = resultados_eval
            
            print(" Analizando reglas...")
            analizador = AnalizadorReglasANFIS(
                modelo['mf_params'], modelo['theta'], X_test, y_test
            )
            
            # Compartir métricas de clasificación con el analizador
            analizador.ultimas_metricas_clasificacion = evaluador.ultimas_metricas
            
            # Pasar flags separados para análisis y cache
            resultados_analisis = analizador.generar_analisis_completo(
                guardar_graficos_analisis=self.opciones_actuales.get('guardar_graficos', True),
                guardar_graficos_cache=config.cache.guardar_cache_graficos,
                visualizar_graficos=visualizar_graficos
            )
            
            # Guardar métricas unificadas en cache
            """ if config.cache.guardar_cache_metricas and hasattr(analizador, 'metricas_unificadas'):
                sistema_cache.guardar_metricas(
                    config.entrenamiento.nombre_modelo, 
                    analizador.metricas_unificadas
                ) """
        return resultados
    
    @staticmethod
    def guardar_grafico_persistente(fig, nombre_archivo):
        """Guarda gráficos en el directorio persistente de resultados"""
        try:
            # Asegurar que el directorio existe
            directorio = config.analisis.directorio_analisis
            if directorio:
                import os
                os.makedirs(directorio, exist_ok=True)
                
                # Guardar gráfico
                ruta = os.path.join(directorio, nombre_archivo)
                fig.savefig(ruta, dpi=300, bbox_inches='tight')
                print(f" Gráfico guardado: {ruta}")
                return ruta
        except Exception as e:
            print(f" Error guardando gráfico: {e}")
            return None
    
    def _mostrar_resumen(self, resultados):
        """Muestra resumen de resultados"""
        print("\n" + "="*60)
        print(" RESUMEN FINAL")
        print("="*60)
        
        if 'evaluacion' in resultados:
            metricas = resultados['evaluacion']['metricas'].get('clasificacion', {})
            if metricas:
                print(f"Precisión: {metricas.get('precision', 0):.4f}")
                print(f"Sensibilidad: {metricas.get('sensitivity', 0):.4f}")
                print(f"Especificidad: {metricas.get('specificity', 0):.4f}")
                print(f"F1-Score: {metricas.get('f1_score', 0):.4f}")
                if metricas.get('auc', 0) > 0:
                    print(f"AUC-ROC: {metricas.get('auc', 0):.4f}")

# Instancia global para uso fácil
pipeline_anfis = PipelineANFIS()


