
############# Entrenamiento ANFIS #############

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variables de Entrada (5 características GLCM)
contraste = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'contraste')
homogeneidad = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'homogeneidad')
entropia = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'entropia')
asm = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'asm')
energia = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'energia')
varianza = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'varianza')
media = ctrl.Antecedent(np.arange(-3, 3, 0.1), 'media')




# Variable de salida:
diagnostico = ctrl.Consequent(np.arange(0, 1, 0.01), 'diagnostico', defuzzify_method='centroid')

# Usar funciones Sugeno de orden cero:
""" diagnostico['tumor'] = fuzz.trapmf(diagnostico.universe, [0.5, 1.0, 1.0, 1.0])
diagnostico['no_tumor'] = fuzz.trapmf(diagnostico.universe, [0.0, 0.0, 0.0, 0.5])
 """

# Usar función trapezoidal para mejor interpretación
diagnostico['tumor'] = fuzz.trapmf(diagnostico.universe, [0, 0, 0.4, 0.6])
diagnostico['no_tumor'] = fuzz.trapmf(diagnostico.universe, [0.4, 0.6, 1, 1])

# Funciones de membresía para cada entrada (ej: contraste)
contraste['bajo'] = fuzz.gaussmf(contraste.universe, -1.5, 0.5)
contraste['alto'] = fuzz.gaussmf(contraste.universe, 1.5, 0.5)

homogeneidad['bajo'] = fuzz.gaussmf(homogeneidad.universe, -1, 0.5)
homogeneidad['alto'] = fuzz.gaussmf(homogeneidad.universe, 1.1, 0.5)

entropia['bajo'] = fuzz.gaussmf(entropia.universe, -1.2, 0.5)
entropia['alto'] = fuzz.gaussmf(entropia.universe, 1.1, 0.5)

asm['bajo'] = fuzz.gaussmf(asm.universe, -1.2, 0.5)
asm['alto'] = fuzz.gaussmf(asm.universe, 1.2, 0.5)

energia['bajo'] = fuzz.gaussmf(energia.universe, -1.2, 0.5)
energia['alto'] = fuzz.gaussmf(energia.universe, 1.2, 0.5)

varianza['bajo'] = fuzz.gaussmf(varianza.universe, -1.2, 0.5)
varianza['alto'] = fuzz.gaussmf(varianza.universe, 1.2, 0.5)

media['bajo'] = fuzz.gaussmf(media.universe, -1.2, 0.5)
media['alto'] = fuzz.gaussmf(media.universe, 1.2, 0.5)


""" # Regla 1: Si contraste es bajo Y homogeneidad es alta Y entropía es baja -> tumor
regla1 = ctrl.Rule(contraste['bajo'] & homogeneidad['alto'] & entropia['bajo'],diagnostico['tumor'])

# Regla 2: Si contraste es alto Y homogeneidad es baja Y entropía es alta -> no tumor
regla2 = ctrl.Rule(contraste['alto'] & homogeneidad['bajo'] & entropia['alto'],diagnostico['no_tumor'])

regla3 = ctrl.Rule(energia['bajo'] & asm['bajo'] & varianza['alto'],diagnostico['tumor'])

# Regla 4: Si contraste es bajo Y ASM es alto Y homogeneidad es alta -> tumor
regla4 = ctrl.Rule(contraste['bajo'] & asm['alto'] & homogeneidad['alto'], diagnostico['tumor'])

# Regla 5: Si energía es alta Y entropía es baja Y varianza es baja -> tumor
regla5 = ctrl.Rule(energia['alto'] & entropia['bajo'] & varianza['bajo'], diagnostico['tumor'])

# Regla 6: Si contraste es alto Y ASM es bajo Y homogeneidad es baja -> no tumor
regla6 = ctrl.Rule(contraste['alto'] & asm['bajo'] & homogeneidad['bajo'], diagnostico['no_tumor'])

# Regla 7: Si energía es baja Y entropía es alta Y varianza es alta -> no tumor
regla7 = ctrl.Rule(energia['bajo'] & entropia['alto'] & varianza['alto'], diagnostico['no_tumor'])

# Regla 8: Si contraste es bajo Y energía es alta Y entropía es baja -> tumor
regla8 = ctrl.Rule(contraste['bajo'] & energia['alto'] & entropia['bajo'], diagnostico['tumor'])

# Regla 9: Si contraste es alto Y energía es baja Y entropía es alta -> no tumor
regla9 = ctrl.Rule(contraste['alto'] & energia['bajo'] & entropia['alto'], diagnostico['no_tumor'])

# Sistema actualizado con las nuevas reglas
sistema = ctrl.ControlSystem([
    regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, regla9
])
simulador = ctrl.ControlSystemSimulation(sistema)
 """



from itertools import product

# Generar todas las combinaciones posibles (2^7 = 128 reglas):
inputs = [contraste, homogeneidad, entropia, asm, energia, varianza, media]
mf_names = ['bajo', 'alto']

# Generar reglas automáticamente (simplificado):
rules = []
for combo in product(['bajo', 'alto'], repeat=7):
    condition = (
        contraste[combo[0]] & homogeneidad[combo[1]] & 
        entropia[combo[2]] & asm[combo[3]] & 
        energia[combo[4]] & varianza[combo[5]] & media[combo[6]]
    )
    output = 'tumor' if sum(1 for c in combo if c == 'alto') > 3 else 'no_tumor'
    rules.append(ctrl.Rule(condition, diagnostico[output]))
sistema = ctrl.ControlSystem(rules)
simulador = ctrl.ControlSystemSimulation(sistema)

# Se exportan las variables necesarias para entrenamiento y predicción
__all__ = ['contraste', 'homogeneidad', 'entropia', 'asm', 'energia', 'varianza', "media", 'diagnostico', 'simulador']