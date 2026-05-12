# Práctica 4

#### Integrantes:
- Román Ismael Contreras Morales.

## La perplejidad

La perplejidad es una medida de incertidumbre de una distribución de probabilidad discreta.
Se puede usar como una métrica para evaluar el desempeño de un modelo de lenguaje.
Intuitivamente representa el grado de incertidumbre o "sorpresa" de un modelo al predecir la siguiente palabra, dado un contexto.

## Desempeño de los modelos

Bajo la métrica de la perplejidad, el mejor modelo fue el de palabras completas.
Como comentamos más ampliamente en la práctica, es probable que esto se deba a los OOV que se convierten artificialmente en UNK.

Dentro de los modelos de subpalabras el desempeño fue mejor para los de 5-gramas.

Sería necesario entrenarlos por más épocas, y comparar los tokenizadores entrenados con los preentrenados.
