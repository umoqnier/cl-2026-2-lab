# Práctica 1 — Lingüística Computacional

## Descripción

Este proyecto contiene el desarrollo de la Práctica 1 de la materia **Lingüística Computacional**.  
El objetivo de la práctica es explorar dos niveles del análisis lingüístico utilizando datos reales:

- **Fonética**, mediante el uso de transcripciones en **IPA** y la aplicación de **distancia de Levenshtein** para encontrar aproximaciones fonéticas entre palabras.
- **Morfología**, utilizando el corpus **SIGMORPHON** para analizar la estructura morfológica de diferentes lenguas.

El análisis se realizó utilizando **Python** y se presenta en un **notebook de Jupyter**.

---

# Parte 1 — Fonética

En esta sección se utiliza un diccionario de transcripciones fonéticas en **IPA** para el español.  
El notebook implementa:

- Descarga del corpus fonético
- Conversión del corpus a una estructura de datos en Python
- Consulta de transcripciones fonéticas
- Implementación de la **distancia de Levenshtein**
- Búsqueda de palabras fonéticamente similares cuando una palabra no se encuentra en el diccionario

Esto permite obtener una transcripción aproximada incluso cuando la palabra no está registrada en el corpus.

---

# Parte 2 — Morfología

En esta sección se utiliza el corpus **SIGMORPHON 2022** para analizar la estructura morfológica de distintas lenguas.

Se procesan datos de:

- Español (`spa`)
- Ruso (`rus`)
- Húngaro (`hun`)

El análisis incluye:

- Descarga y procesamiento del corpus
- Conversión de los datos a un **DataFrame de pandas**
- Cálculo del número de morfemas por palabra
- Comparación del comportamiento morfológico entre lenguas
- Cálculo de proporciones de categorías morfológicas
- Visualización de resultados mediante **gráficas**

---

# Resultados

Los resultados muestran diferencias claras entre las lenguas analizadas.  
El **húngaro** presenta un mayor número promedio de morfemas por palabra y una mayor presencia de estructuras morfológicas complejas, lo cual es consistente con su carácter **aglutinante**.

En contraste, **español** y **ruso** presentan estructuras menos segmentadas dentro del corpus analizado.

---

# Tecnologías utilizadas

- Python 3
- Jupyter Notebook
- pandas
- matplotlib
- requests

---

# Nota sobre el uso de IA

Durante la elaboración de este proyecto se utilizó **ChatGPT** como herramienta de apoyo para:

- corrección ortográfica del texto
- mejora del formato del notebook
- asistencia en la redacción de documentación

