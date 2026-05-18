# Práctica 3: Representaciones_Vectoriales.ipynb
## Requisitos

- Python >= 3.12

- Jupyter Notebook o JupyterLab (o algún otro entorno para ejecutar notebooks).

## Dependencias

Son necesarias las siguientes dependencias.
- `nltk`  
  - Tokenización (`word_tokenize`) y descarga de recurso (`nltk.download("punkt_tab")`).

- `numpy`  
  - Importada como `np` (aunque en este notebook no se usa explícitamente en operaciones visibles).

- `pandas`  
  - Construcción y visualización de `DataFrame` (matrices BoW/TF-IDF y tabla comparativa).

- `scikit-learn`  
  - `CountVectorizer` (BoW), `TfidfVectorizer` (TF-IDF), `cosine_similarity` (similitud entre query y documentos).

- `gensim`  
  - Carga de embeddings preentrenados (`gensim.downloader`, modelo `glove-wiki-gigaword-100`).
## Ejecución

Abre el archivo `Practica3_Representaciones_Vectoriales.ipynb` en algun entorno capaz de ejecutar notebooks y ejecuta las celdas en orden.

> **Nota:** El notebook descarga datos desde internet, por lo que se requiere conexión a la red al ejecutarlo.

> Es necesario tener los documentos en el directorio de Documentos, en caso de no estar disponibles en la entrega puede encontrarlos en https://drive.google.com/drive/folders/1k3qNq9-SrMtDV5a9jdqIZ9w6VeIObBwQ?usp=sharing

### Uso de herramientas generativas

- Utilicé LLM para redactar los documentos que hablen de temas de manera más o menos extensa.
