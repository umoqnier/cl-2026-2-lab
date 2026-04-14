# Práctica 3: Representaciones vectoriales

## Descripción

En esta práctica se trabajó con dos enfoques de representación vectorial aplicados al análisis de texto.

En la primera parte se construyó un pequeño corpus de cinco documentos distribuidos en dos temas contrastantes: la búsqueda de vida en Marte y el estridentismo en México. A partir de este corpus se generaron representaciones mediante **Bag of Words (BoW)** y **TF-IDF**, y posteriormente se calculó la similitud coseno entre los documentos y una *query* tramposa diseñada para mezclar vocabulario de ambos temas.

En la segunda parte se utilizaron **vectores de palabras preentrenados** con el modelo `glove-wiki-gigaword-100` a través de `gensim`, con el fin de explorar asociaciones semánticas y observar ejemplos de sesgo de género en analogías construidas con la función `most_similar()`.

## Contenido

Este directorio contiene los siguientes archivos:

- `P3.ipynb`: notebook principal de la práctica.
- `P3.py`: versión en script del notebook, generada con Jupytext.
- `README.md`: archivo descriptivo de la práctica.

## Herramientas utilizadas

- Python
- pandas
- nltk
- scikit-learn
- gensim
- Jupyter Notebook
- Jupytext

## Nota sobre el uso de IA

Se utilizó **ChatGPT** como apoyo para:

- corrección ortográfica;
- revisión de la adecuación, coherencia y cohesión de las conclusiones;
- revisión de la adecuación, coherencia y cohesión de este archivo `README.md`.


## Observaciones

La práctica se centró en comparar el comportamiento de BoW y TF-IDF en una tarea sencilla de recuperación de información, así como en explorar cómo los embeddings preentrenados pueden reflejar sesgos presentes en los datos con los que fueron entrenados.