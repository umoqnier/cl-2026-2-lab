# Práctica 2: Propiedades Estadísticas del Lenguaje y Diversidad

Contiene el desarrollo de la Práctica 2 para la asignatura de Lingüística Computacional. El objetivo es verificar empíricamente las leyes de distribución de frecuencias en el lenguaje y analizar la riqueza taxonómica de las lenguas del mundo mediante herramientas de visualización geográfica.

## Contenido

- **Verificación de la Ley de Zipf**: Análisis comparativo de la distribución de frecuencias ($f$) frente al rango ($r$) en dos tipos de corpus:
  - **Lenguaje Artificial**: Generación de texto aleatorio mediante un script para observar si emergen leyes de potencia a partir de caracteres al azar.
  - **Lengua de Bajos Recursos**: Procesamiento de un corpus real (nahuatl) para contrastar el comportamiento del lenguaje natural y estimar el parámetro $\alpha$.
- **Diversidad Lingüística**: Mapeo interactivo de la distribución geográfica de lenguas en México y comparación con regiones de alta diversidad global (como Papúa Nueva Guinea) utilizando la base de datos de **Glottolog**.

## Estructura de Archivos

- `Practica02LC.ipynb`: Notebook con el desarrollo experimental, gráficas de regresión lineal para $\alpha$ y mapas interactivos de Plotly.
- `Practica02LC.py`: Script de Python con el código fuente de la práctica.

## Requisitos y Dependencias

Para ejecutar los scripts y el notebook de esta práctica, es necesario contar con Python 3.10+ y las siguientes dependencias:

```bash
pip install pandas matplotlib plotly numpy elotl
