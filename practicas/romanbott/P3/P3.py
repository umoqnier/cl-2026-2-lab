# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 3 - Representaciones vectoriales

# %% [markdown]
# ## Matrices dispersas y búsqueda de documentos

# %%
import nltk
import re
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import ChainMap
import gensim.downloader as gensim_api

# %%
nltk.download("punkt")


# %% [markdown]
# Usaremos cinco artículos de wikipedia.
# Dos sobre animales:
# - Gato
# - Demonio de Tasmania
# y tres sobre temas de física:
# - El demonio de Laplace
# - El núcleo del demonio
# - El gato de Schrödinger
#
# Para obtener el texto de los artículos usamos la función `get_wikipedia_text`

# %%
def get_wikipedia_text(title: str) -> str:
    """
    Obtiene el contenido de texto plano de un artículo de Wikipedia en español.

    Args:
        title (str): El título del artículo de Wikipedia.

    Returns:
        str: El contenido del artículo o un mensaje de error si no se encuentra.
    """
    
    URL = "https://es.wikipedia.org/w/api.php"
    
    headers = {
        'User-Agent': 'Simple python script.'
    }
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    
    response = requests.get(URL, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        pages = data["query"]["pages"]
        page_id = next(iter(pages))
        return pages[page_id].get("extract", "Article not found.")
    else:
        return f"Error: {response.status_code}"



# %%
NOMBRES_ARTICULOS = [
    "Demonio_de_Laplace",
    "Sarcophilus_harrisii",
    "Núcleo_del_demonio",
    "Felis_catus",
    "Gato_de_Schrödinger"
]

# %%
ARTICULOS = {nombre: get_wikipedia_text(nombre) for nombre in NOMBRES_ARTICULOS}


# %%
def create_bow_dataframe(corpus: dict[str, str], vectorizer) -> pd.DataFrame:
    """
    Crea un DataFrame de pandas representando una matriz ocurrencias de términos por documento.

    Args:
        corpus (dict): Diccionario con títulos como llaves y contenido como valores.
        vectorizer: Instancia de CountVectorizer o TfidfVectorizer de sklearn.

    Returns:
        pd.DataFrame: Matriz donde las filas son documentos y las columnas son palabras.
    """
    matrix = vectorizer.fit_transform(corpus.values())

    df = pd.DataFrame(
        matrix.toarray(), index=corpus.keys(), columns=vectorizer.get_feature_names_out()
    )
    return df


# %%
def simple_preprocess(text: str):
    """
    Tokeniza y limpia un texto eliminando puntuación, números y palabras cortas.

    Args:
        text (str): Texto original a procesar.

    Returns:
        list: Lista de tokens (palabras) normalizados en minúsculas.
    """
    
    tokens = word_tokenize(text.lower(), language="spanish")
    # Ignoramos signos de puntuación y palabras de longitud 1
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# %%
def compute_similarities(df: pd.DataFrame, row: str) -> pd.Series:
    """
    Calcula la similitud de coseno entre una fila específica y el resto del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con vectores de palabras.
        row (str): Nombre del índice de la fila que se usará como base de comparación.

    Returns:
        pd.Series: Valores de similitud ordenados por el índice original.
    """
    
    row = df.loc[[row]]
    similarities = cosine_similarity(row, df)
    return pd.Series(similarities[0], index=df.T.columns)


# %%
def compute_most_similar(corpus: dict[str, str], query: str) -> pd.DataFrame:
    """
    Compara una consulta (query) contra un corpus usando modelos BoW y TF-IDF.

    Args:
        corpus (dict): Diccionario de documentos existentes.
        query (str): Texto de búsqueda o consulta.

    Returns:
        pd.DataFrame: Similitudes de la consulta contra cada documento en ambos modelos.
    """
    
    query_map = {"Query": query}

    updated_corpus = ChainMap(corpus, query_map)
    
    bow = create_bow_dataframe(
        updated_corpus,
        CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
    )
    
    tfidf = create_bow_dataframe(
        updated_corpus,
        TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
    )

    return pd.DataFrame({
        'BoW': compute_similarities(bow, "Query"),
        'TF-IDF': compute_similarities(tfidf, "Query")}).drop("Query")
    


# %%
compute_most_similar(ARTICULOS, "El demonio de Tasmania es un marsupial carnívoro del tamaño de un perro que posee un desagradable olor.")

# %% [markdown]
# En este ejemplo, con el método BoW, los dos artículos más similares son "Demonio de Laplace" y "Gato de Schrödinger",
# mientras que con el método TF-IDF, correctamente se identifica el artículo "Sarcophilus harrisii" como el más similar.

# %%
compute_most_similar(ARTICULOS, "El gato de Schrodinger es un experimento mental.")

# %% [markdown]
# Aquí ambos métodos correctamente identifican el artículo "Gato de Schrödinger".

# %%
compute_most_similar(ARTICULOS, "El gato doméstico es junto con el perro, el animal doméstico más popular como mascota.")

# %% [markdown]
# Con este ejemplo, el método BoW identifica al artículo "Gato de Schrödinger" como el más similar, y ligeramente por arriba del artículo correcto. En cambio el método TF-IDF sí identifica el artículo "Felis catus" como el más similar.

# %% [markdown]
# En el primer ejemplo, es probable que la palabra "demonio" que aparece muchas veces en los artículos de física, y no tantas en el artículo del animal sea la causa de que BoW identifique los artículos incorrectos. En cambio, osando el método TF-IDF, palabras como "Tasmania" que aparecen únicamente en el artículo "Sarcophilus harrisii" tengan más relevancia que palabras "comunes" como demonio, que aparecen por lo menos en tres de los artículos del corpus.
#
# De forma similar, en el tercer ejemplo, es probable que la palabra "gato" sea determinante para el método BoW, pero que pierda relevancia en el método TF-IDF (es penalizada). En cambio palabras como "mascota" seguramente tienen más peso con este último método.

# %% [markdown]
# ## Búsqueda de sesgos

# %%
word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %%
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))


# %% [markdown]
# Al pasar a la función `most_similar` los parámtros `positive=['man', 'profession']` y `negative=['woman']`
# podemos pensar que estamos tomando un vector dirección:
# $$
# v_{w\rightarrow m} = v_m - v_w
# $$
#
# donde $v_w$ es el vector correspondiente a la palabra "woman" y $v_m$ el vector correspondiente a la palabra "man".
#
# Podríamos interpretar este vector como una representación abstracta del proceso de convertir un término relacionado con "woman" a uno relacionado con "man".
#
# Finalmente al sumar el vector base $v_p$ correspondiente a la palabra "profession", estamos pidiendo a la función `most_similar` que devuelva los vectores más cercanos al vector $v_p + v_{w\rightarrow m}$, es decir los términos más similares a lo que se obtiene si el término "profession" se desplaza en la dirección del vector $v_{w\rightarrow m}$.
#
# En el segundo caso funciona exactamente igual, pero ahora con el vector $v_{m\rightarrow w} = v_w -v_m$.
#
# - Asociaciones obtenidas con el vector $v_{w\rightarrow m}$: Las palabras tienden a ser abstractas y relacionadas con el estatus o la capacidad intelectual: practice, knowledge, skill, reputation, philosophy, mind. Refleja una visión de la profesión como una autoridad intelectual o una búsqueda de prestigio.
#
# - Asociaciones con el vector $v_{m\rightarrow w}$: Las palabras son mucho más específicas y están orientadas al cuidado y la educación: nursing, childbirth, teacher, educator.
#
# El modelo ha "aprendido" que, estadísticamente, la palabra "mujer" aparece más cerca de roles de servicio, cuidado o enseñanza primaria, mientras que "hombre" se vincula a conceptos de liderazgo, conocimiento técnico y reputación.

# %%
def get_analogies(base: str, source: str, target: str) -> list[str]:
    """
    Resuelve una analogía de palabras (Word Embeddings) usando la lógica:
    base es a source como X es a target (X = base - source + target).

    Args:
        base (str): Palabra base.
        source (str): Palabra de origen de la relación.
        target (str): Palabra de destino de la relación.

    Returns:
        list[str]: Lista de términos candidatos que completan la analogía.
    """
    
    return [
        res[0] for res in word_vectors.most_similar(positive=[base, target], negative=source)
    ]


# %%
def get_bias(base: str, source: str, target: str) -> tuple[list[str], list[str]]:
    """
    Detecta sesgos comparando los resultados de una analogía en dos direcciones opuestas.

    Args:
        base (str): Concepto neutro a evaluar (ej. 'job').
        source (str): Atributo A (ej. 'woman').
        target (str): Atributo B (ej. 'man').

    Returns:
        tuple: (Palabras únicas asociadas a target, Palabras únicas asociadas a source).
    """
    
    res = get_analogies(base, source, target)
    inv_res = get_analogies(base, target, source)
    return (
        [word for word in res if word not in inv_res],
        [word for word in inv_res if word not in res]
    )


# %%
def get_bias2(base: str, vector1: str, vector2: str) -> tuple[list[str], list[str]]:
    """
    Proporciona una manera alternativa de buscar sesgos comparando asociaciones directas.

    A diferencia de get_bias, este método no utiliza una resta de vectores (analogía), 
    sino que observa los términos más cercanos a la combinación simple del concepto 
    'base' con dos atributos distintos para identificar disparidades en los resultados.

    Args:
        base (str): Concepto base a analizar.
        vector1 (str): Primer atributo de asociación (ej. 'woman').
        vector2 (str): Segundo atributo de asociación (ej. 'man').

    Returns:
        tuple[list[str], list[str]]: Palabras exclusivas asociadas al primer vector 
        y palabras exclusivas asociadas al segundo.
    """
    
    res = [
        res[0] for res in word_vectors.most_similar(positive=[base, vector1])
    ]
    inv_res = [
        res[0] for res in word_vectors.most_similar(positive=[base, vector2])
    ]
    return (
        [word for word in res if word not in inv_res],
        [word for word in inv_res if word not in res]
    )


# %%
get_bias("writer", "woman", "man")

# %%
get_bias2("hobbies", "woman", "man")

# %%
get_bias2("job", "woman", "man")

# %%
get_bias("feelings", "woman", "man")

# %%
get_bias("job", "woman", "man")

# %%
get_bias("personality", "woman", "man")

# %%
get_bias("talent", "woman", "man")

# %%
get_bias("cuisine", "mexico", "france")

# %%
get_bias("cuisine", "africa", "europe")

# %% [markdown]
# Los resultados de las funciones `get_bias` y `get_bias2` demuestran cómo los vectores de palabras capturan y reproducen prejuicios sistémicos presentes en los datos de entrenamiento.
#
# #### Sesgos de género
#
# Se observa una marcada polarización en la representación de capacidades, roles sociales y estados emocionales:
#
# * **Intelecto vs. apariencia:** En la analogía de `talent`, el vector masculino se desplaza hacia conceptos de éxito y genialidad (**"great"**, **"players"**, **"success"**, **"genius"**), mientras que el femenino se asocia a la estética (**"beauty"**, **"artistic"**).
#
# * **Autoridad vs. cuidado:** Para el concepto `job`, el lado masculino arroja términos de gestión y ejecución (**"manager"**, **"done"**, **"work"**). En contraste, el lado femenino se vincula a roles de servicio y asistencia (**"nursing"**, **"care"**, **"staff"**, **"working"**).
#
#   
# * **Identidad social:** En `hobbies`, el modelo asocia a la mujer casi exclusivamente con roles familiares y reproductivos (**"mother"**, **"wife"**, **"daughter"**, **"children"**). Para el hombre, las asociaciones son más externas o individuales (**"friends"**, **"father"**, **"person"**).
#
#   
# * **Esfera emocional:** En `feelings`, el vector masculino se relaciona con procesos mentales o actitudes de confrontación (**"mind"**, **"attitude"**, **"hatred"**), mientras que el femenino se satura de estados de vulnerabilidad y pasividad (**"affection"**, **"longing"**, **"anxiety"**, **"loneliness"**).
#
#
# #### Sesgos culturales y eurocentrismo (`cuisine`)
#
# Los resultados reflejan una jerarquía donde las culturas occidentales son el estándar de profesionalismo:
#
# * **Francia vs. México:** La cocina francesa es categorizada con términos de prestigio institucional y técnica superior (**"gourmet"**, **"chef"**, **"gastronomy"**, **"menu"**). La cocina mexicana se reduce a ingredientes, sabores o regiones geográficas (**"salsa"**, **"seafood"**, **"oaxaca"**, **"michoacan"**), sugiriendo una percepción de "folclore" frente a "ciencia culinaria".
# * **Europa vs. África:** Se repite el patrón de sofisticación para Europa (**"gourmet"**, **"restaurants"**, **"chefs"**). Para África, el modelo utiliza etiquetas de nicho o descriptores limitados (**"creole"**, **"lowcountry"**, **"culinary"**), omitiendo el léxico de alta cocina profesional asociado al vector europeo.

# %% [markdown]
# ### Cómo mitigar los sesgos en modelos de lenguaje
#
# Para mitigar los sesgos en modelos vectoriales podríamos considerar alguna de las siguientes estrategias:
#
# ### Modificar el "encaje" (_embbeding_)
# Una vez que el modelo ya está entrenado, buscar la "dirección" de posibles sesgos, por ejemplo con el vector $v_{w\rightarrow m} = v_m - v_w$ para posteriormente forzar a que términos que deberían ser neutros se encuentren en un plano ortogonal a esta dirección, y de forma equidistante a  $v_m$ y $v_w$.
# Esta técnica dependería de poder identificar manualmente diversos sesgos, y se corre el riesgo de que al aplicar muchas de estas proyecciones el encaje resultante no se desempeñe bien.
#
# ### Balanceo del corpus
# Se interviene directamente en el corpus antes de que el modelo aprenda de él.
# Se podría buscar que el modelo reciba la misma cantidad de ejemplos asociando roles profesionales o rasgos de personalidad a ambos géneros o distintas etnias.
