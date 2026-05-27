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
# Al comparar los resultados, se observa que el documento más similar a la query es el mismo tanto en Bag of Words (BoW) como en TF-IDF: Marte 1. Sin embargo, los valores de similitud no son idénticos, lo que indica que ambos modelos representan de forma distinta la relación entre la query y los documentos.
#
# En el caso de BoW, la similitud depende directamente de la frecuencia de las palabras compartidas, por lo que los términos comunes pueden tener un peso considerable dentro de la representación. Esto hace que la query conserve una similitud relativamente alta con documentos que comparten vocabulario, incluso si algunas de esas palabras no son las más informativas.
#
# Por su parte, TF-IDF ajusta los pesos de las palabras según su importancia dentro del corpus. Las palabras que aparecen en varios documentos reciben menos peso, mientras que los términos más específicos adquieren mayor relevancia. Por ello, aunque el documento más similar sigue siendo el mismo, los puntajes cambian y la comparación se vuelve más sensible a las palabras realmente representativas del tema.
#
# La query construida mezcla vocabulario del tema de Marte con términos asociados al estridentismo, lo que introduce ambigüedad. En este caso, ambos modelos siguen identificando correctamente un documento del tema de Marte como el más cercano, pero TF-IDF atenúa el impacto de las palabras compartidas o frecuentes y produce una ponderación más refinada.
#
# En conclusión, en este corpus TF-IDF no cambió el documento más relevante, pero sí ofreció una representación más precisa de la similitud, al reducir la influencia de términos menos informativos y resaltar los más específicos.
#
# </div>

# %% [markdown]
# ## 2. Búsqueda de sesgos

# %% [markdown]
# ### 2.1 Carga del modelo

# %%
import gensim.downloader as gensim_api

word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %% [markdown]
# ### 2.2 Comparación hombre vs mujer

# %%
print("Hombre + profesión - mujer:")
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))

print("\nMujer + profesión - hombre:")
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))

# %% [markdown]
# ### 2.3 Análisis de diferencias

# %% [markdown]
# <div style="text-align: justify;">
#
# Al comparar los resultados obtenidos, se observan diferencias en las palabras asociadas a hombres y mujeres en relación con la noción de profesión. En el caso de "man + profession - woman", aparecen términos como practice, knowledge, skill, reputation, philosophy y discipline, que remiten a habilidades, prestigio y formación intelectual.
#
# En cambio, para "woman + profession - man" aparecen palabras como nursing, teaching, childbirth, teacher y educator, que se relacionan con el cuidado, la docencia y la maternidad. Aunque también aparecen términos generales como professions, practitioner y academic, la lista muestra una asociación más marcada con roles históricamente feminizados.
#
# Esto puede interpretarse como un reflejo de sesgo de género en los datos de entrenamiento del modelo. Los vectores no “entienden” el mundo social, sino que aprenden patrones estadísticos presentes en grandes corpus de texto. Por ello, si en esos textos ciertas profesiones o actividades aparecen con mayor frecuencia vinculadas a hombres o a mujeres, el modelo reproduce dichas asociaciones.
#
# </div>

# %% [markdown]
# ### 2.4 Analogías para identificar sesgo

# %%
# Analogías para explorar sesgo de género
print("woman + doctor - man:")
print(word_vectors.most_similar(positive=['woman', 'doctor'], negative=['man']))

print("\nman + nurse - woman:")
print(word_vectors.most_similar(positive=['man', 'nurse'], negative=['woman']))

# %% [markdown]
# ### 2.5 Explicación del sesgo

# %% [markdown]
# <div style="text-align: justify;">
#
# Las analogías realizadas muestran un patrón claro de sesgo de género en los embeddings. En el caso de "woman + doctor - man", el modelo devuelve palabras como nurse, physician, doctors, patient, dentist, pregnant, nursing y mother. Aunque varias pertenecen al campo médico, también aparecen términos asociados al cuidado y a la maternidad, como nurse, pregnant y mother.
#
# Por otro lado, en la analogía "man + nurse - woman", el modelo produce palabras como doctor, physician, surgeon, psychiatrist, technician, officer y sergeant. Aquí se observa una asociación más fuerte con profesiones de prestigio, autoridad o especialización técnica.
#
# En conjunto, estos resultados sugieren que el modelo reproduce asociaciones sociales presentes en los datos de entrenamiento: lo femenino aparece más vinculado al cuidado y la maternidad, mientras que lo masculino se relaciona con profesiones de mayor autoridad o estatus. Esto no significa que el modelo comprenda estas diferencias, sino que aprende patrones estadísticos del lenguaje y, al hacerlo, también hereda sesgos culturales e históricos.
#
# </div>

# %% [markdown]
# ### 2.6 Mitigación de sesgos 

# %% [markdown]
# <div style="text-align: justify;">
#
# Una estrategia consiste en revisar y balancear mejor los datos de entrenamiento, de modo que profesiones, actividades y atributos no queden desproporcionadamente asociados a un solo género.
#
# También pueden utilizarse técnicas de debiasing sobre los embeddings, orientadas a reducir asociaciones problemáticas entre palabras. A esto se suma la necesidad de evaluar periódicamente el modelo con pruebas específicas de sesgo, especialmente si se pretende utilizar en contextos sensibles.
#
# En términos generales, la mitigación no depende de una sola solución, sino de una combinación de mejores datos, evaluación constante y ajustes posteriores al entrenamiento.
#
# </div>
