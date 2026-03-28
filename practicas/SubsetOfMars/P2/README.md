# Práctica 2 - Lingüística Computacional

## Descripción

En esta práctica se trabajó con dos temas principales:

1. **Verificación empírica de la Ley de Zipf**
   - Generación de un lenguaje artificial a partir de un alfabeto definido.
   - Obtención de frecuencias de palabras en un corpus de una lengua de bajos recursos digitales (maya yucateco).
   - Estimación del parámetro $\alpha$.
   - Elaboración de gráficas de rango vs. frecuencia en escala normal y logarítmica.
   - Comparación entre el comportamiento del lenguaje artificial y el corpus natural.

2. **Visualización de la diversidad lingüística**
   - Filtrado de datos de Glottolog con base en coordenadas geográficas.
   - Visualización de las lenguas de México en un mapa, coloreadas por familia lingüística.
   - Repetición del procedimiento para otro país (India).
   - Análisis comparativo de la diversidad lingüística observada.

## Archivos

- `P2.ipynb`: notebook principal de la práctica.
- `P2.py`: versión en script del notebook, vinculada mediante Jupytext.
- `README.md`: descripción general de la práctica.

## Requisitos

Para ejecutar esta práctica se utilizaron librerías de Python como:

- `collections`
- `random`
- `re`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `plotly`
- `os`

Además, para la segunda parte se utilizaron los archivos de Glottolog:

- `languages_and_dialects_geo.csv`
- `languoid.csv`

Estos archivos deben colocarse dentro de una carpeta llamada `data`.

## Notas

El corpus de lengua de bajos recursos digitales se construyó a partir de textos en **maya yucateco** obtenidos de **Wikimedia Incubator**.

Para la visualización de diversidad lingüística se emplearon datos de **Glottolog**.

## Uso de IA

Se utilizó inteligencia artificial como apoyo para:

- corregir ortografía,
- mejorar el estilo de redacción,
- agregar etiquetas y comentarios para una mejor legibilidad del código,
- y apoyar en la organización general del notebook.

