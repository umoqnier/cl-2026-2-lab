# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 3: Representaciones vectoriales

# %% [markdown]
# ## 1. Matrices dispersas y búsqueda de documentos

# %% [markdown]
# ### 1.1 Preparación

# %%
# %pip install nltk pandas scikit-learn

# %%
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Descarga recursos necesarios de NLTK
nltk.download("punkt")
nltk.download("punkt_tab")

# Preprocesamiento simple para tokenizar y limpiar texto
def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language="spanish")
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# %% [markdown]
# ### 1.2 Corpus de 5 documentos

# %%
# Tema 1: búsqueda de vida en Marte
doc_1 = "La exploración de Marte busca evidencias de vida pasada mediante el análisis de suelo, rocas y posibles rastros de agua en la superficie del planeta."
doc_2 = "Las misiones espaciales han enviado robots a Marte para estudiar su atmósfera y detectar compuestos químicos que puedan indicar actividad biológica."
doc_3 = "El estudio de microorganismos en condiciones extremas en la Tierra ayuda a comprender cómo podría existir vida en ambientes hostiles como los de Marte."

# Tema 2: estridentismo en México
doc_4 = "El estridentismo fue un movimiento artístico y literario en México que surgió en la década de 1920, caracterizado por su ruptura con las tradiciones estéticas."
doc_5 = "Los estridentistas promovían una estética moderna influenciada por la tecnología, la velocidad y la vida urbana, reflejando los cambios sociales de su época."

documents = [doc_1, doc_2, doc_3, doc_4, doc_5]

# %% [markdown]
# ### 1.3 Vectorización con Bag of Words

# %%
# Construye la representación BoW del corpus
vectorizer = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
bag_of_words_corpus = vectorizer.fit_transform(documents)

# Visualiza la matriz BoW como arreglo
bag_of_words_corpus.toarray()


# %% [markdown]
# ### 1.4 Tabla de frecuencias con BoW

# %%
# Convierte una matriz de texto en un DataFrame con nombres de columnas
def create_bow_dataframe(docs_raw: list, titles: list[str], vectorizer) -> pd.DataFrame:
    matrix = vectorizer.fit_transform(docs_raw)
    df = pd.DataFrame(
        matrix.toarray(),
        index=titles,
        columns=vectorizer.get_feature_names_out()
    )
    return df


# %%
# Etiquetas para identificar los documentos en la tabla
titles = ["Marte 1", "Marte 2", "Marte 3", "Estridentismo 1", "Estridentismo 2"]

# Tabla BoW del corpus
docs_matrix_bow = create_bow_dataframe(
    documents,
    titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)

docs_matrix_bow

# %% [markdown]
# ### 1.5 Vectorización con TF-IDF

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
# Tabla TF-IDF del corpus
docs_matrix_tfidf = create_bow_dataframe(
    documents,
    titles,
    TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)

docs_matrix_tfidf

# %% [markdown]
# ### 1.6 Query tramposa

# %%
# Query dirigida al tema de Marte, pero con palabras asociadas al estridentismo
query = """
La búsqueda de vida en Marte analiza suelo, atmósfera y rastros de agua en el planeta,
pero también puede pensarse desde una visión moderna, urbana, tecnológica y de vanguardia.
"""

# %% [markdown]
# ### 1.7 Similitud coseno con BoW

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Convierte la query al mismo espacio vectorial de BoW
query_bow = vectorizer.transform([query])

# Calcula similitud coseno entre la query y cada documento del corpus
similaridades_bow = cosine_similarity(query_bow, bag_of_words_corpus)[0]

similaridades_bow

# %%
# Tabla de similitudes con BoW
resultados_bow = pd.DataFrame({
    "Documento": titles,
    "Similitud_BoW": similaridades_bow
})

resultados_bow.sort_values(by="Similitud_BoW", ascending=False)

# %% [markdown]
# ### 1.8 Similitud coseno con TF-IDF

# %%
# Crea un vectorizador TF-IDF y ajusta el corpus
tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
tfidf_corpus = tfidf_vectorizer.fit_transform(documents)

# Convierte la query al espacio TF-IDF
query_tfidf = tfidf_vectorizer.transform([query])

# Calcula similitud coseno entre la query y cada documento
similaridades_tfidf = cosine_similarity(query_tfidf, tfidf_corpus)[0]

similaridades_tfidf

# %%
# Tabla de similitudes con TF-IDF
resultados_tfidf = pd.DataFrame({
    "Documento": titles,
    "Similitud_TF_IDF": similaridades_tfidf
})

resultados_tfidf.sort_values(by="Similitud_TF_IDF", ascending=False)

# %% [markdown]
# ### 1.9 Tabla comparativa de resultados

# %%
# Tabla comparativa final
resultados_comparativos = resultados_bow.merge(resultados_tfidf, on="Documento")

resultados_comparativos.sort_values(by="Similitud_TF_IDF", ascending=False)

# %% [markdown]
# ### 1.10 Análisis de resultados

# %% [markdown]
# <div style="text-align: justify;">
#
# Al comparar los resultados, se observa que el documento más similar a la query varía entre los modelos Bag of Words (BoW) y TF-IDF. Esta diferencia se debe a la forma en que cada método pondera las palabras dentro del corpus.
#
# En el caso de BoW, todas las palabras contribuyen según su frecuencia absoluta, por lo que los términos más comunes tienden a dominar la representación. Como resultado, la query puede verse influida por palabras frecuentes o compartidas entre distintos temas, incluso si no aportan un valor semántico relevante.
#
# Por el contrario, TF-IDF introduce un mecanismo de ponderación que penaliza las palabras que aparecen en múltiples documentos y resalta aquellas que son más específicas. Esto permite que la similitud entre la query y los documentos esté determinada principalmente por términos más representativos del contenido.
#
# Además, la query construida incluye palabras de ambos temas, lo que introduce ambigüedad. En este contexto, BoW resulta más sensible a esta mezcla de vocabulario, mientras que TF-IDF logra atenuar dicho efecto al priorizar las palabras más informativas.
#
# En conclusión, TF-IDF ofrece una representación más adecuada para tareas de recuperación de información, ya que reduce el impacto de términos frecuentes y destaca aquellos que mejor capturan el contenido semántico de los documentos.
#
# </div>

# %% [markdown]
# ## 2. Búsqueda de sesgos
