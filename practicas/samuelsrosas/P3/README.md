# Práctica 3: Representaciones Vectoriales

Contiene el desarrollo de la Práctica 3 para la asignatura de Lingüística Computacional. El objetivo es explorar la representación matemática de documentos y palabras mediante matrices dispersas y modelos semánticos distribucionales, así como analizar los sesgos inherentes en los modelos de lenguaje pre-entrenados.

## Contenido

- **Matrices Dispersas y Búsqueda de Documentos**: Implementación de un motor de búsqueda simple para comparar el rendimiento de **Bolsa de Palabras (BoW)** y **TF-IDF**. Se evalúa cómo la penalización *Inverse Document Frequency* (IDF) ayuda a discriminar una query "tramposa" (basada en repetición de palabras genéricas de desarrollo backend) frente al conteo puro de frecuencias para identificar correctamente la temática real (adopción de mascotas) utilizando la similitud coseno.

- **Búsqueda de Sesgos (Word Embeddings)**: Uso del modelo pre-entrenado `glove-wiki-gigaword-100` mediante la biblioteca `gensim` para realizar álgebra de palabras y evaluar analogías. Se identifican y explican sesgos de género y segregación ocupacional en áreas tecnológicas, proponiendo estrategias de mitigación a nivel de balanceo de datos y ajustes matemáticos de los vectores.

>**Nota:** Para el desarrollo de la sección de "Matrices Dispersas", se utilizó asistencia de Inteligencia Artificial como herramienta de redacción para generar el corpus de documentos

## Estructura de Archivos

- `Practica03LC.ipynb`: Notebook con el desarrollo, generación de tablas comparativas de similitud y evaluación de analogías semánticas en consola.
- `Practica03LC.py`: Script de Python con el código fuente de la práctica.

## Requisitos y Dependencias

Para ejecutar los scripts y el notebook de esta práctica, es necesario contar con Python 3.10+ y las siguientes dependencias:

```bash
pip install pandas scikit-learn nltk gensim
