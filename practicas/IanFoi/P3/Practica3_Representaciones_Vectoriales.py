# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cl-2026-2-lab
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Práctica 3: Representaciones Vectoriales
#
# **Fecha de entrega: 31 de Marzo de 2026 @ 11:59pm**
#
# ### Matrices dispersas y búsqueda de documentos
#
# Este apartado requiere que construyas un motor de búsqueda entre documentos comparando el rendimiento de una Bolsa de Palabras (BoW) y TF-IDF para procesar un documento "tramposo" (documento con muchas palabras que aportan poco significado o valor temático):
#
#
#

# %% [markdown]
# Comenzamos el notebook importando las bibliotecas necesarias para el trabajo.

# %%
from pathlib import Path
import nltk
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as gensim_api


# %% [markdown]
# 1. Define una lista de 5 documentos cortos divididos en dos temas contrastantes.
#     - Ej: 3 de Revolución Rusa y 2 de comida vegana.
#     
# Comenzamos escribiendo los documentos, para ello utilicé la IA que redactara tres documentos que hablen de álgebra lineal y otros dos que hablen de dinosaurios con una longitud de al menos 300 palabras. 
# Posteriormente, los obtenemos en nuestro código.
#

# %%
ruta_corpus = Path('./Documentos')
documentos = []
nombres_documentos = []
archivos = list(ruta_corpus.glob('algebra*.txt')) + list(ruta_corpus.glob('dinos*.txt'))

for archivo in archivos:
    with open(archivo, 'r', encoding='utf-8') as f:
        documentos.append(f.read())
        nombres_documentos.append(archivo.name)
print(documentos)
print(nombres_documentos)

# %% [markdown]
#
# 2. Crea una query "tramposa", esto es, crea un documento dirigido a alguna temática pero repitiendo intencionalmente palabras comunes o verbos genéricos que aparezcan en los documentos de la otra temática.
#

# %% [markdown]
# Creamos una query con la temática de un cuento de dinosaurios explicando el concepto de espacio vectorial, de esta manera, tenemos un documento que habla de los espacios vectoriales pero incluye multiples veces terminos de dinosaurios. Para ello, pedi a la IA que redactara un cuento de dinosaurios que explica los espacios vectoriales como si fuera una historia con dinosaurios.

# %%
with open('./Documentos/trampa.txt', 'r', encoding='utf-8') as f:
    texto_query = f.read()

query = [texto_query]

print(query)


# %% [markdown]
#
# 3. Vectoriza para crear una BoW y calcula la similitud coseno entre la query y los 5 documentos

# %% [markdown]
# Reciclamos el codigo de la clase para vectorizar y crear la BoW correspondiente a los documentos que creamos.
#

# %%
def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language="spanish")
    # Ignoramos signos de puntuación y palabras de longitud 1
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# %%
nltk.download("punkt_tab")

# %%
vectorizer = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
bag_of_words_corpus = vectorizer.fit_transform(documentos)
diccionario = vectorizer.vocabulary_
sorted(diccionario.items(), key=lambda x: x[1])
for column_idx, word in enumerate(vectorizer.get_feature_names_out()):
    print(column_idx, word)


# %% [markdown]
# Obtenemos una lista de palabras ordenada por índice de columna, que es el mismo orden que las columnas de la matriz de bag of words. Esto nos permite interpretar cada columna de la matriz como la frecuencia de una palabra específica en cada documento. De esta manera podemos visualizar la BoW como una matriz cuyas filas corresponden a la frecuencia de las palabras en cada uno de los documentos.

    # %%
    # Visualizando la matriz
bag_of_words_corpus.toarray()


# %% [markdown]
# Finalmente utilizando la función de la clase creamos un dataframe de pandas para visualizar de manera más comoda la información.

# %%
def create_bow_dataframe(docs_raw: list, titles: list[str], vectorizer) -> pd.DataFrame:
    # fit_transform ajusta el vocabulario y crea la matriz en un solo paso
    matrix = vectorizer.fit_transform(docs_raw)

    # Podemos crear el DataFrame directamente pasando la matriz a un array tradicional
    # vectorizer.get_feature_names_out() nos da la lista de palabras en el orden exacto de las columnas
    df = pd.DataFrame(
        matrix.toarray(), index=titles, columns=vectorizer.get_feature_names_out()
    )
    return df


# %%
titles = ["ESPACIOS", "SISTEMAS", "VALORES_PROPIOS", "CARNIVOROS", "HERBIVOROS", "TRAMPAS" ]
docs_matrix = create_bow_dataframe(
    documentos + query,
    titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)#, binary=True),
)

# %%
docs_matrix

# %% [markdown]
# Ahora, calculamos la similitud coseno entre la query y los cinco documentos.

# %%
query_vector = docs_matrix.loc["TRAMPAS"].values.reshape(1, -1)
cos_similarity_bow = []
for doc_title in titles[:-1]:
    current_doc_values = docs_matrix.loc[doc_title].values.reshape(1, -1)
    similarity = cosine_similarity(query_vector, current_doc_values)[0][0]
    cos_similarity_bow.append(similarity)
    print(f"Similitud entre 'TRAMPAS' y '{doc_title}': {similarity:.4f}")


# %% [markdown]
#
# 4. Repite el proceso usando TF-IDF
#
# Obtenemos de manera similar al ejercicio anterior el dataframe de los vectores de la BoW usando TF-IDF cambiando el vectorizador por uno del tipo Tfidf.

# %%
docs_matrix_tfidf = create_bow_dataframe(
    documentos + query,
    titles,
    vectorizer=TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)
docs_matrix_tfidf

# %% [markdown]
# Y luego calculamos la diferencia coseno entre la query y los cinco documentos: 

# %%
query_vector = docs_matrix_tfidf.loc["TRAMPAS"].values.reshape(1, -1)
cos_similarity_tfidf = []
for doc_title in titles[:-1]:  
    current_doc_values = docs_matrix_tfidf.loc[doc_title].values.reshape(1, -1)
    similarity = cosine_similarity(query_vector, current_doc_values)[0][0]
    cos_similarity_tfidf.append(similarity)
    print(f"Similitud entre 'TRAMPAS' y '{doc_title}': {similarity:.4f}")

# %% [markdown]
# 5. Imprime un DataFrame o tabla comparativa que muestre los scores de similitud de BoW y TF-IDF del query contra los 5 documentos.
#     

# %% [markdown]
# Obtenemos el dataframe.

# %%
df_comparativo = pd.DataFrame({
    'Documento': nombres_documentos,
    'Score BoW': cos_similarity_bow,
    'Score TF-IDF': cos_similarity_tfidf
})
df_comparativo

# %% [markdown]
# - ¿Cambió el documento clasificado como "más similar/relevante" al pasar de BoW a TF-IDF? Identifica el cambio si lo hubo.
#
#     Podemos ver que en la BoW sin tf-idf el documento más cercano a la query tramposa de dinosaurios hablando de espacios vectoriales es el documento que habla de espacios vectoriales, sin embargo, tambien tiene muchisima relación el texto que habla de dinosaurios herbivoros y carnivoros dado que en la query tramposamente agregamos palabras que se relacionan a los dinosaurios. En un modelo en el que buscamos caracterizar los documentos la tematica de nuestra trampa no deberia de ser acerca de los dinosaurios herbívoros pues es claro que el tema central son las características de un espacio vectorial.
#
#     Ahora bien, con respecto a la matriz con TF-IDF tenemos que los documentos más cercanos a la query tramposa en realidad son los que hablan de algebra lineal y los documentos de dinosaurios pierden muchisima relación con la query. Este resultado es ideal, pues es claro que en nuestra query el tema que caracteriza el documento trampa es el algebra lineal por lo que debería perder relevancia el tema de los dinosaurios.
#
# - Explica brevemente, basándote en la penalización idf (Inverse Document Frequency), cómo y por qué TF-IDF procesó de manera distinta las palabras de tu "trampa léxica" en comparación con BoW.
#
#     Dado que la BoW se basa en contar frecuencias, como la query trampa metia terminos relacionados a los dinosaurios herbivoros y carnívoros aumentó la frecuencia de dichos términos lo que infló su similitud con los documentos de dinosaurios poniendolo como tema central del texto aunque no lo fuera.
#     Por otro lado, cuando utilizamos TF-IDF la penalización idf le agregó mas peso a las palabras frecuentes en el documento pero no tan frecuentes en los demás. Por ello, palabras como vector, escalar o neutro que aparecen en la query y en especifico en el documento que define los espacios vectoriales tienen mayor peso comparado con palabras como dinosaurio o algunos verbos genéricos que aparecen tanto en el documento de dinosaurios herbívoros y carnívoros o en general en los cinco documentos.

# %% [markdown]
# ### Búsqueda de sesgos
#
# 1. Descarga el modelo `glove-wiki-gigaword-100` con la api de `gensim` y ejecuta el siguiente código:
#
# ```python
# print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
# print()
# print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))
# ```
#

# %%
word_vectors = gensim_api.load("glove-wiki-gigaword-100")
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))


# %% [markdown]
#
# 2. Identifica las diferencias en la lista de palabras asociadas a hombres/mujeres y profesiones, explica como esto reflejaría un sesgo de genero.
#
# A pesar de que en ambos casos aparecen términos vinculados al ámbito profesional (como professions, practitioner o practice), es evidente un sesgo de género marcado. Mientras que a los hombres se les vincula con el trabajo, las habilidades técnicas y el conocimiento aplicado, a las mujeres se las asocia predominantemente con la enfermería o la enseñanza. Me causa mucho ruido que el término "embarazo" aparezca en esta relación.
#
# Este fenómeno refleja los roles de género presentes en los corpus de entrenamiento. Históricamente, muchos textos describen al hombre bajo el rol de proveedor, orientándolo al desarrollo de capacidades profesionales, mientras que a menudo se subestima la competencia femenina o se la posiciona como una opción menos favorable. Esto explica por qué las palabras referentes a habilidades técnicas no muestran una similitud vectorial con "mujer" y "profesión". Además, persiste la idea errónea de que las profesiones "naturales" para las mujeres son aquellas relacionadas con el cuidado y la educación, lo que mete a la lista de resultados con términos como enfermera, educadora o maestra.
#
# Asimismo, es alarmante encontrar la palabra "parto" asociada a la mujer en un contexto profesional. Esto es, probablemente, un reflejo de la discriminación estructural: en puestos de alta jerarquía y remuneración, se suele penalizar la posibilidad de la maternidad. Los empleadores —en su mayoría hombres— suelen percibir el embarazo y sus secuelas como un factor de "poca confiabilidad" o una potencial pérdida de productividad. En consecuencia, el embarazo y el parto adquieren una relevancia desproporcionada en el desarrollo profesional de las mujeres, convirtiéndose en un factor de exclusión que, lógicamente, no afecta la trayectoria de los hombres.

# %% [markdown]
#
# 3. Utiliza la función `.most_similar()` para identificar analogías que exhiba algún tipo de sesgo de los vectores pre-entrenados.
#     - Explica brevemente que sesgo identificar
#     Podemos ver que la relación entre personas mexicanas, o en general de cualquier país formado, se relaciona con progreso económico basado en capital

# %%
print(word_vectors.most_similar(positive=['american', 'migration'], negative=['mexican']))
print(word_vectors.most_similar(positive=['american', 'immigration'], negative=['mexican']))
print()
print(word_vectors.most_similar(positive=['mexican', 'migration'], negative=['american']))
print(word_vectors.most_similar(positive=['mexican', 'immigration'], negative=['american']))

# %% [markdown]
# Podemos observar que, dado que muchos de estos modelos se entrenan principalmente con datos generados en Estados Unidos, tienden a incorporar los sesgos presentes en el discurso social y mediático de ese contexto. En particular, esto se refleja en la forma en que se representa el fenómeno de la migración. Por un lado, la inmigración hacia ese país por parte de paises como mexico suele asociarse con connotaciones negativas, vinculándose frecuentemente con la ilegalidad, la criminalidad o actividades como el tráfico de drogas. Por otro lado, la migración desde ese pais se relaciona con ideas positivas como el desarrollo, la expansión de oportunidades o el acceso a la educación.
#
# Esta diferencia en las asociaciones no es casual, sino que responde a narrativas predominantes que posicionan la inmigración como un problema interno que debe ser gestionado o controlado, mientras que la movilidad humana de grupos privilegiados se interpretan como procesos legítimos o incluso deseables. Como consecuencia, los modelos no solo reproducen estas diferencias, sino que pueden reforzarlas al presentarlas como relaciones naturales.
#

# %% [markdown]
#
# 4. Si fuera tu trabajo crear un modelo ¿Como mitigarías estos sesgos al crear vectores de palabras?
#
# Considero que el problema del sesgo en estos sistemas no radica únicamente en su funcionamiento técnico, sino que es, una consecuencia de los sesgos sociales, culturales y de género presentes en nuestras sociedades. Dado que estos modelos se entrenan con grandes volúmenes de texto, y que históricamente la producción escrita ha estado dominada por hombres en posiciones de privilegio, existe una sobrerrepresentación de sus perspectivas frente a las de poblaciones vulnerables. Como resultado, el sistema tiende a aprender y reproducir estas visiones como si fueran neutrales o universales, reflejándolas en sus respuestas.
#
# Para solucionarlo creo que se debe fomentar y facilitar el acceso a la expresión y producción de conocimiento por parte de estas poblaciones. Sin embargo, sabemos que reducir las desigualdades estructurales que limitan dicha participación es una tarea compleja y de largo plazo, por ello, una medida inmediata sería incorporar de manera deliberada literatura y contenidos que representen las experiencias, visiones del mundo y problemáticas de estos grupos, así como textos críticos que evidencien las desigualdades generadas por los sesgos existentes. Asimismo, durante el proceso de entrenamiento, podría aplicarse un peso mayor a estos materiales, con el fin de compensar su menor presencia en comparación con los textos canónicos.
