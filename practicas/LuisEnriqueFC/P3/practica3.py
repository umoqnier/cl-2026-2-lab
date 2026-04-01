# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### 3. Práctica: Representaciones 
# #### Matrices dispersas y búsqueda de documentos
#
# Este apartado requiere que construyas un motor de búsqueda entre documentos comparando el rendimiento de una Bolsa de Palabras (BoW) y TF-IDF para procesar un documento "tramposo" (documento con muchas palabras que aportan poco significado o valor temático):
#
# 1. Define una lista de 5 documentos cortos divididos en dos temas contrastantes.
#     - Ej: 3 de Revolución Rusa y 2 de comida vegana.
# 2. Crea una query "tramposa", esto es, crea un documento dirigido a alguna temática pero repitiendo intencionalmente palabras comunes o verbos genéricos que aparezcan en los documentos de la otra temática.
# 3. Vectoriza para crear una BoW y calcula la similitud coseno entre la query y los 5 documentos
# 4. Repite el proceso usando TF-IDF
# 5. Imprime un DataFrame o tabla comparativa que muestre los scores de similitud de BoW y TF-IDF del query contra los 5 documentos.
#     - ¿Cambió el documento clasificado como "más similar/relevante" al pasar de BoW a TF-IDF? Identifica el cambio si lo hubo.
#     - Explica brevemente, basándote en la penalización idf (Inverse Document Frequency), cómo y por qué TF-IDF procesó de manera distinta las palabras de tu "trampa léxica" en comparación con BoW.

# %%
# !{sys.executable} -m pip install nltk

# %%
import sys
# !{sys.executable} -m pip install scikit-learn

# %%
import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd

# %%
nltk.download("punkt")

# %%
doc_1="El ramadán es el noveno mes del calendario islámico, respetado por musulmanes como el mes del ayuno, oración, reflexión y comunidad. Cada año el mes en el que se celebra el Ramadán cambia en torno al mes lunar. Es una conmemoración de la primera revelación de Mahoma. El cumplimiento anual del ramadán está considerado como uno de los cinco pilares del islam y su duración es de veintinueve a treinta días, a partir de la luna creciente hasta la próxima luna creciente."
doc_2="El Ramadán es el mes más sagrado para los seguidores del islam. Durante este periodo, quienes están en condiciones de hacerlo deben abstener se de comer, beber, fumar y mantener relaciones sexuales desde el amanecer hasta el atardecer. El propósito, de acuerdo con la tradición, es fortalecer la relación con Dios, practicar la autodisciplina y desarrollar empatía hacia quienes tienen menos recursos. De acuerdo con información de Middle East Eye, el ayuno es uno de los cinco pilares del islam, junto con La declaración de fe La oración La caridad La peregrinación a La Meca. Los musulmanes creen que durante este mes fueron revelados los primeros versículos del Corán al profeta Mahoma, hace más de 1.400 años."
doc_3="La física nuclear es una rama de la física que se ocupa del estudio de los núcleos atómicos, las partículas subatómicas y las interacciones nucleares. Se centra en comprender la estructura y propiedades de los núcleos atómicos, así como las fuerzas y reacciones nucleares que ocurren en ellos. La física nuclear abarca una amplia gama de temas, que incluyen la desintegración radioactiva, la fisión nuclear, la fusión nuclear, la radiactividad, las interacciones de partículas cargadas con la materia, las reacciones nucleares inducidas y la producción de energía a través de procesos nucleares. También se investiga la formación y desintegración de isótopos y la generación de elementos en el Universo, así como la radiación y sus efectos sobre la materia y los seres vivos. Los avances en la física nuclear han llevado al desarrollo de aplicaciones prácticas en diversos campos, como la generación de energía nuclear, la medicina nuclear, la datación por radiocarbono, la investigación en astrofísica y la producción de materiales y radioisótopos para uso industrial y médico."
doc_4="La física nuclear es una rama de la física que estudia las propiedades, comportamiento e interacciones de los núcleos atómicos. En un contexto más amplio, se define la física nuclear y de partículas como la rama de la física que estudia la estructura fundamental de la materia y las interacciones entre las partículas subatómicas. La física nuclear es conocida mayoritariamente por el aprovechamiento de la energía nuclear en centrales nucleares y en el desarrollo de armas nucleares, tanto de fisión nuclear como de fusión nuclear, pero este campo ha dado lugar a aplicaciones en diversos campos, incluyendo medicina nuclear e imágenes por resonancia magnética, ingeniería de implantación de iones en materiales y datación por radiocarbono en geología y arqueología."
doc_5="La física nuclear es una rama fundamental de la física que estudia el núcleo atómico, sus componentes y las interacciones que en él ocurren. A lo largo de más de un siglo, esta disciplina ha evolucionado de simples teorías y descubrimientos experimentales a una ciencia aplicada que influye en múltiples aspectos de la vida moderna. En este artículo exploraremos en detalle qué es la física nuclear, sus fundamentos, su desarrollo histórico y las aplicaciones que han transformado sectores tan diversos como la medicina, la energía, la industria y la investigación científica. La física nuclear se encarga de estudiar las propiedades y la estructura de los núcleos de los átomos. Los núcleos están compuestos por protones y neutrones, partículas subatómicas que interactúan mediante la fuerza nuclear fuerte. Este campo se originó a comienzos del siglo XX, cuando experimentos pioneros revelaron que la mayor parte de la masa de un átomo se concentraba en su núcleo y que las interacciones internas eran mucho más complejas de lo que se pensaba en el modelo atómico clásico. La comprensión de estas interacciones abrió la puerta a aplicaciones revolucionarias que hoy forman parte de nuestro día a día. La importancia de la física nuclear radica en la capacidad de explicar fenómenos tanto a nivel microscópico como macroscópico. Desde la estabilidad de la materia hasta los procesos energéticos que ocurren en el sol, la física nuclear ofrece respuestas a preguntas fundamentales sobre la estructura de la materia y las leyes que rigen el universo."

# %%
documents = [doc_1, doc_2, doc_3, doc_4, doc_5]

# %%
import re
from nltk.tokenize import word_tokenize


def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language="spanish")
    # Ignoramos signos de puntuación y palabras de longitud 1
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
vectorizer = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)

# %%
bag_of_words_corpus = vectorizer.fit_transform(documents)


# %%
diccionario = vectorizer.vocabulary_

# %%
bag_of_words_corpus.toarray()

# %%
print(len(bag_of_words_corpus.toarray()))
len(bag_of_words_corpus.toarray()[1])


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
titles = ["RAMADAN(WIKIPEDIA)", "RAMADAN(EL IMPARCIAL)", "FISICA NUCLEAR(ENERGIA NUCLEAR)","FISICA NUCLEAR(ESTUDYANDO)","FISICA NUCLEAR(WIKIPEDIA)"]
docs_matrix = create_bow_dataframe(
    documents,
    titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)#, binary=True),
)

# %%
docs_matrix

# %%
documents = [doc_1, doc_2, doc_3, doc_4, doc_5]

# %%
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
docs_matrix_tfidf = create_bow_dataframe(
    documents, titles, TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)

# %%
docs_matrix_tfidf

# %%
#Ahora crearemos el documento "tramposo" que contenga algunas palabras similares a las que esten relacionados con ambos documentos tanto del Ramadán
#como de la fisica nuclear
tricky_query = "El Ramadán es una festividad musulmana que se centra en la conmemoración de la primera revelación de mahoma, a través de la interacción entre la comunidad, la disciplina abarca tradiciones como el ayuno y la oración. Esta festividad influye en la manera que los creyentes observan y se sienten respecto a las recompensas espirituales, siendo que es una celebridad donde los creyentes se centran en la fé y la primera revelación de Mahoma. Esta ocurre en el noveno mes del calendario islámico, donde se transforma su vide e influyen en la percepci+on de la misma."

# %%
updated_docs = documents.copy()
updated_docs.append(tricky_query)
updated_titles = titles + ["RAMADAN TRICKY"]

# %%
updated_matrix = create_bow_dataframe(
    updated_docs,
    updated_titles,
    vectorizer=TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None),
)

# %%
from sklearn.metrics.pairwise import cosine_similarity


# %%
for doc_title in updated_titles:
    current_doc_values = updated_matrix.loc[doc_title].values.reshape(1, -1)
    # Seleccionamos [0][0] para extraer el valor numérico (float) de la matriz de resultado
    sim = cosine_similarity(current_doc_values, doc_tricky_values)[0][0]
    print(f"Similitud entre RAMADANTRICKY/{doc_title} = {sim:.2f}")

# %%
docs_matrixbownew = create_bow_dataframe(
    updated_docs,
    updated_titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)#, binary=True),
)

# %%
for doc_title in updated_titles:
    bow_docs_values = docs_matrixbownew.loc[doc_title].values.reshape(1, -1)
    # Seleccionamos [0][0] para extraer el valor numérico (float) de la matriz de resultado
    sim = cosine_similarity(bow_docs_values, doc_tricky_values)[0][0]
    print(f"Similitud entre SAMPLE/{doc_title} = {sim:.2f}")

# %% [markdown]
# ### 2.Busqueda de sesgos
# ##### 1.-Descarga el modelo glove-wiki-gigaword-100 con la api de gensim y ejecuta el siguiente código:
# print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
#
# print()
#
# print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))
# ##### 2.-Identifica las diferencias en la lista de palabras asociadas a hombres/mujeres y profesiones, explica como esto reflejaría un sesgo de genero.
# ##### 3.-Utiliza la función .most_similar() para identificar analogías que exhiba algún tipo de sesgo de los vectores pre-entrenados.
# Explica brevemente que sesgo identificar
# Si fuera tu trabajo crear un modelo ¿Como mitigarías estos sesgos al crear vectores de palabras?

# %%
# !pip install gensim

# %%
import gensim.downloader as gensim_api

# %%
gensim_api.info(name_only=True)

# %%
word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %%
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))

# %% [markdown]
# ### ANÁLISIS
#
# Si analizamos ambos conjuntos de palabras podemos observar una gran diferencia, pues las profesiones que se exponen en ambas si que son bastantes distintas entre sí, como la de enfermera o filosofía en el caso de los hombres, considero que el sesgo se observa más en las palabras relacionadas, pues se le relaciona más a la mujer en profesiones parto, cosa que personalmente me parecen rara además de mencionar vocación y cualificaciones, mientras que para los hombres resaltan la disciplina el conocimiento la habilidad y la relación con la reputación, el hecho de incluir estas características solo en la parte para hombre.
#
# Esto para mi indica que mientras que para los hombres las cualidades indican reconocimiento y disciplina para las mujeres se enfocan más en su lado de cuidados e indicando que estas no sean acordes a dichas características sobretodo me sorprende como no aparece la habilidad en ambos y como en mujeres aparecen como doble palabra respecto a dar orientación/ eduación y para los hombres aparezca habilidad y habilidades en general.

# %%
print(word_vectors.most_similar(positive=['young', 'profession'], negative=['old']))
print()
print(word_vectors.most_similar(positive=['old', 'profession'], negative=['young']))

# %% [markdown]
# ### ANÁLISIS
# En esta ocasión consulte una página para revisitar cuales han sido y siguen siendo los prejuicios mas comúnes en el caso anterior vimos un sesgo sexista, en este decidí optar por si este modelo contiene sesgos relacionados al edadismo, siento que no es tan común notarlo, sin embargo considero que es uno de los más fuertes en la sociedad actual y revisaremos que nos arrojó el modelo.
#
# En este caso seguimos con la relación a las profesiones pues siempre se considera que a cierta edad si eres joven o anciano te corresponden ciertos trabajos y se te limita a otros por lo cual veremos si el modelo tiene dicho sesgo.
#
# De inicio notamos que para la sección de profesiones se da nuevamente enfermera y se quita la característica de creativo así como de agregar a contadores junto a la caracteristica de viejo y se quita la de enseñar en esta misma manteniendola para la de joven además de indicar médicos solo para los jóvenes.
#
# Apesar de ello me agrada como la caracteristica de profesión o profesionistas se mantiene en ambas.
#

# %% [markdown]
# ### COMO EVITAR SESGOS
#
# Yo considero que si tuviera que hacer algun modelo no importa que haga se mantendrán sesgos pues considero que un entrenamiento es mas lento que la actualización en el lenguaje e ideas de la sociedad en general, podríamos intentarlo enviando textos nuevos y certificados respecto a una actualización en la toma de relaciones semánticas, enseñandole nuevos documentos parecidos o cuyas frases ocntengan estas nuevas palabras e ideas intentando corregir el modelo a través del entrenamiento y nutrición con textos de una forma constante y supervisada.
#
# Aunque vuelvo a lo mismo considero que por la cantidad de texto y de revisiones como contenido nuevo que no es comúnmente usado se mantendrá una leve capa o leve tiempo donde los sesgos o prejuicios se mantengan sea largo o corto se mantendrá por el simple hehco de tiempo de entrenamiento y análisis del modelo para generar dicha semántica,
#
# A pesar de ello creo que una buena medida es esa, ir explorando de manera supervisaqda estos textos nuevos que van conteniendo nuevo vocabulario como ideas o conjunciones de frases que uiza en otros anteiores no se vean.

# %% [markdown]
# ### REFERENCIAS
# colaboradores de Wikipedia. (2026, March 29). Física nuclear. Wikipedia, La Enciclopedia Libre. https://es.wikipedia.org/wiki/F%C3%ADsica_nuclear
#
# Cruzito. (2025, March 8). ¿Qué es la Física Nuclear y cuáles son sus Aplicaciones? | Estudyando. Estudyando. https://estudyando.com/que-es-la-fisica-nuclear-y-cuales-son-sus-aplicaciones/
#
# La física nuclear: una introducción simple. (n.d.). [Video]. https://energia-nuclear.net/fisica/fisica-nuclear#google_vignette
#
# colaboradores de Wikipedia. (2026, March 19). Ramadán. Wikipedia, La Enciclopedia Libre. https://es.wikipedia.org/wiki/Ramad%C3%A1n
#
# Zarate, A. (2026, February 18). ¿Qué es el Ramadán, cómo se práctica y cuándo inicia? Todos los detalles sobre el mes más sagrado para los seguidores del islam. ¿Qué Es El Ramadán, Cómo Se Práctica Y Cuándo Inicia? Todos Los Detalles Sobre El Mes Más Sagrado Para Los Seguidores Del Islam. https://www.elimparcial.com/estilos/2026/02/18/que-es-el-ramadan-como-se-practica-y-cuando-inicia-todos-los-detalles-sobre-el-mes-mas-sagrado-para-los-seguidores-del-islam/
#
# PSICOBLOG. (2026, March 19). Prejuicios SOCIALES: Ejemplos y CONSECUENCIAS Impactantes. https://psicoblog.org/prejuicios-sociales-ejemplos/#Tipos_Comunes_de_Prejuicios 
