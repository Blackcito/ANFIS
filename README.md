[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Blackcito/ANFIS)

#  ANFIS — Detección de Tumores Cerebrales

Sistema de clasificación binaria para la detección de tumores cerebrales en imágenes MRI, basado en un **Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS)** con inferencia tipo Sugeno.

---

## ¿Qué hace este proyecto?

Dado un conjunto de imágenes MRI en escala de grises, el sistema:

1. **Preprocesa** cada imagen (CLAHE → filtro bilateral → segmentación adaptativa)
2. **Extrae características** de textura usando matrices GLCM (7 features)
3. **Entrena** un modelo ANFIS con funciones de membresía gaussianas optimizadas por PSO
4. **Clasifica** cada imagen como `Tumor` (1) o `No Tumor` (0)
5. **Explica** la predicción mostrando las reglas difusas más activas




---

## Estructura del proyecto

```
Modelo_anfis_ajustado/
│
├── core/
│   ├── anfis_sugeno.py        # Motor ANFIS: reglas, pesos, parámetros Sugeno
│   ├── training.py            # Entrenamiento PSO + LSE
│   ├── prediction.py          # Inferencia binaria y explicable
│   ├── pipeline_anfis.py      # Pipeline principal orquestador
│   └── gestor_datos.py        # Carga y normalización de datos
│
├── features/
│   └── procesamiento_image.py # Preprocesamiento + extracción GLCM
│
├── analysis/
│   ├── evaluador.py           # Métricas, ROC, matriz de confusión
│   └── analisis.py            # Importancia de reglas y visualizaciones
│
├── interfaz/
│   ├── configurador.py
│   ├── visualizador_graficos.py  
│   └── ventana_principal.py   # GUI (tkinter)
│
├── config/
│   ├── ruta.py
│   ├── menu_configuracion.py  
│   └── configuracion.py       # Parámetros globales del sistema
│
├── utils/
│   └── cache.py               # Sistema de caché para features y modelos
│
└── main.py                    # Punto de entrada (menú interactivo / legacy)
```

---

## Arquitectura del modelo

El modelo ANFIS implementa una arquitectura neuro-difusa de 6 capas:

```
Imagen MRI
    │
    ▼
Preprocesamiento (CLAHE + Bilateral + Segmentación)
    │
    ▼
Extracción GLCM → [Contraste, ASM, Homogeneidad, Energía, Media, Entropía, Varianza]
    │                         (vector de 7 features)
    ▼
Funciones de Membresía Gaussianas
    2 por feature (bajo / alto) → 14 parámetros (μ, σ)
    │
    ▼
128 Reglas Difusas  (2^7 combinaciones de bajo/alto)
    │
    ▼
Pesos Normalizados  w̄_j = w_j / Σw
    │
    ▼
Consecuentes Sugeno  ŷ = Σ w̄_j · (p_j · x + r_j)
    │
    ▼
Umbral 0.5 → Tumor / No Tumor
```
<img width="4023" height="903" alt="image" src="https://github.com/user-attachments/assets/fec342c9-7eeb-4bd3-8d92-4e72f5cde5c4" />

### Entrenamiento híbrido

| Componente | Algoritmo | Qué optimiza |
|---|---|---|
| Funciones de membresía | PSO (Particle Swarm Optimization) | Parámetros no lineales (μ, σ) |
| Consecuentes | LSE (Mínimos Cuadrados) | Parámetros lineales (p_j, r_j) |

<img width="743" height="1022" alt="image" src="https://github.com/user-attachments/assets/c7e3330f-596e-4e5b-8eda-de446f7f6c00" />


---

## Features GLCM extraídas

| # | Feature | Descripción |
|---|---|---|
| 1 | Contraste | Variaciones locales de intensidad |
| 2 | ASM | Angular Second Moment (uniformidad) |
| 3 | Homogeneidad | Similitud entre píxeles adyacentes |
| 4 | Energía | Suma de elementos al cuadrado |
| 5 | Media | Nivel de gris promedio |
| 6 | Entropía | Aleatoriedad de la textura |
| 7 | Varianza | Dispersión de la distribución de intensidad |

**Configuración GLCM:** distancias `[1, 2, 3]`, ángulos `[0°, 45°, 90°, 135°]`, 256 niveles, simétrico y normalizado.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Blackcito/ANFIS.git
cd ANFIS/Modelo_anfis_ajustado

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

```
numpy
opencv-python
scikit-image
scikit-learn
scikit-fuzzy
pyswarm
matplotlib
seaborn
```

---

## Uso

### GUI (recomendado)

```bash
python interfaz/ventana_principal.py
```

### Línea de comandos

```bash
python main.py
```

Menú disponible:

```
1. Pipeline completo (entrenar nuevo modelo)
2. Solo evaluación (usar modelo existente)
3. Evaluar modelo específico por nombre
4. Configurar sistema
5. Gestión de caché
6. Salir
```



---

## Sistema de caché

El sistema cachea automáticamente las features extraídas para acelerar experimentos repetidos:

| Operación | Sin caché | Con caché |
|---|---|---|
| Extracción GLCM (1000 imgs) | ~1–5 seg/imagen | ~0.1 seg total |
| Carga de modelo | N/A | <1 seg |

El caché se guarda en `./features_cache/` como archivos `.npz`. Para forzar reprocesamiento:

```python
pipeline_anfis.ejecutar(forzar_reprocesamiento=True, ...)
```

---

## Salidas del sistema

### Métricas de evaluación
- Matriz de confusión
- Curva ROC + AUC
- Precisión, Sensibilidad, Especificidad, F1-Score

### Análisis de reglas difusas
- Importancia por regla (activación media × magnitud del consecuente)
- Mapa de calor de condiciones (bajo/alto por feature)
- Contribución global de cada feature GLCM
- Discriminación por clase (reglas más activas para Tumor vs No Tumor)



---

## Parámetros clave

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `n_vars` | 7 | Número de features de entrada |
| `swarmsize` | 50 | Partículas en el enjambre PSO |
| `maxiter` | 20 | Iteraciones máximas PSO |
| `threshold` | 0.5 | Umbral de clasificación binaria |
| `phip` | 1.5 | Atracción al mejor personal (PSO) |
| `phig` | 2.0 | Atracción al mejor global (PSO) |
| `omega` | 0.5 | Inercia de velocidad (PSO) |

---

## Variantes del proyecto

| Carpeta | Descripción |
|---|---|
| `models/anfis_adjusted/` | ✅ Versión principal — ANFIS binario con GUI y caché |
| `models/anfis_3cat/` | ANFIS multiclase (Meningioma / No Tumor / Pituitaria) |
| `models/mamdani/` | Variante con inferencia tipo Mamdani |
| `models/sandbox/` | Scripts experimentales de aprendizaje |
| `training_data/` | Datos de entrenamiento y prueba |

---

## Generar ejecutable (.exe)

```bash
pyinstaller ANFIS_Tumor_Cerebral.spec
# El ejecutable se genera en dist/
```

---

## Licencia

Ver [LICENSE](./LICENSE).