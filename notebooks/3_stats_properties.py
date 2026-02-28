# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # 3. Propiedades estadísticas del lenguaje

# ## Objetivos

# - Mostrar el uso de CFG y derivados
#     - Ejemplos de parseo de dependencias
# - Ejemplificar etiquetado NER usando bibliotecas existentes
# - Explorar propiedades estadísticas del lenguaje natural y observar los siguientes fenomenos:
#     - La distribución de Zipf
#     - La distribución de Heap

# ## Perspectivas formales

# - Fueron el primer acercamiento al procesamiento del lenguaje natural. Sin embargo tienen varias **desventajas**
# - Requieren **conocimiento previo de la lengua**
# - Las herramientas son especificas de la lengua
# - Los fenomenos que se presentan son muy amplios y difícilmente se pueden abarcar con reglas formales (muchos casos especiales)
# - Las reglas tienden a ser rigidas y no admiten incertidumbre en el resultado

# ### Sintaxis

# ![](https://imgs.xkcd.com/comics/formal_languages_2x.png)
#
# **[audience looks around] 'What just happened?' 'There must be some context we're missing.'**

# #### Parsing basado en reglas

# - Gramaticas libres de contexto:
#
# $G = (T, N, O, R)$
# * $T$ símbolos terminales.
# * $N$ símbolos no terminales.
# * $O$ simbolo inicial o nodo raíz.
# * $R$ reglas de la forma $X \longrightarrow \gamma$ donde $X$ es no terminal y $\gamma$ es una secuencia de terminales y no terminales

import nltk
import pandas as pd
import numpy as np

plain_grammar = """
S -> NP VP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
PP -> P NP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
"""

# +
grammar = nltk.CFG.fromstring(plain_grammar)
# Cambiar analizador y trace
analyzer = nltk.ChartParser(grammar)

sentence = "I shot an elephant in my pajamas".split()
trees = analyzer.parse(sentence)
# -

for tree in trees:
    print(tree, type(tree))
    print('\nBosquejo del árbol:\n')
    print(tree.pretty_print(unicodelines=True, nodedist=1))

# ## Perspectiva estadística

# - Puede integrar aspectos de la perspectiva formal
# - Lidia mejor con la incertidumbre y es menos rigida que la perspectiva formal
# - No requiere conocimiento profundo de la lengua. Se pueden obtener soluciones de forma no supervisada

# ## Modelos estadísticos

# - Las **frecuencias** juegan un papel fundamental para hacer una descripción acertada del lenguaje
# - Las frecuencias nos dan información de la **distribución de tokens**, de la cual podemos estimar probabilidades.
# - Existen **leyes empíricas del lenguaje** que nos indican como se comportan las lenguas a niveles estadísticos
# - A partir de estas leyes y otras reglas estadísticas podemos crear **modelos del lenguaje**; es decir, asignar probabilidades a las unidades lingüísticas

# ### Probabilistic Context Free Grammar

taco_grammar = nltk.PCFG.fromstring("""
O    -> FN FV     [0.7]
O    -> FV FN     [0.3]
FN   -> Sust      [0.6]
FN   -> Det Sust  [0.4]
FV   -> V FN      [0.8]
FV   -> FN V      [0.2]
Sust -> 'Juan'    [0.5]
Sust -> 'tacos'   [0.5]
Det  -> 'unos'    [1.0]
V    -> 'come'    [1.0]
""")
viterbi_parser = nltk.ViterbiParser(taco_grammar)

sentences = [
    "Juan come unos tacos",
    "unos tacos Juan come"
]
for sent in sentences:
    for tree in viterbi_parser.parse(sent.split()):
        print(tree)
        print("Versión bosque")
        tree.pretty_print(unicodelines=True, nodedist=1)

# ### Parseo de dependencias

# Un parseo de dependencias devuelve las dependencias que se dan entre los tokens de una oración. Estas dependencias suelen darse entre pares de tokens. Esto es, que relaciones tienen las palabras con otras palabras.

# ##### Freeling - https://nlp.lsi.upc.edu/freeling/demo/demo.php

import spacy
from spacy import displacy

# !python -m spacy download es_core_news_lg

nlp = spacy.load("es_core_news_lg")

doc = nlp("La niña come un suani")

displacy.render(doc, style="dep")

for chunk in doc.noun_chunks:
    print("text::", chunk.text)
    print("root::", chunk.root.text)
    print("root dep::", chunk.root.dep_)
    print("root head::", chunk.root.head.text)
    print("="*10)

for token in doc:
    print("token::", token.text)
    print("dep::", token.dep_)
    print("head::", token.head.text)
    print("head POS::", token.head.pos_)
    print("CHILDS")
    print([child for child in token.children])
    print("="*10)

# #### Named Entity Recognition (NER)

# El etiquetado NER consiste en identificar "objetos de la vida real" como organizaciones, paises, personas, entre otras y asignarles su etiqueta correspondiente. Esta tarea es del tipo *sequence labeling* ya que dado un texto de entrada el modelo debe identificar los intervalos del texto y etiquetarlos adecuadamente con la entidad que le corresponde. Veremos un ejemplo a continuación.

# !pip install datasets

from datasets import load_dataset

from huggingface_hub import login
login()

# !hf auth whoami

data = load_dataset("MarcOrfilaCarreras/spanish-news")

# +
# data?

# +
import random

random.seed(42)
corpus = random.choices(data["train"]["text"], k=3)
docs = list(nlp.pipe(corpus))
for j, doc in enumerate(docs):
    print(f"DOC #{j+1}")
    doc.user_data["title"] = " ".join(doc.text.split()[:10])
    for i, ent in enumerate(doc.ents):
        print(" -"*10, f"Entity #{i}")
        print(f"\tTexto={ent.text}")
        print(f"\tstart/end={ent.start_char}-{ent.end_char}")
        print(f"\tLabel={ent.label_}")
# -

displacy.render(docs, style="ent")

# [Available labels](https://spacy.io/models/en)

# ## Leyes estadísticas

# Bibliotecas
from collections import Counter
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [10, 6]

mini_corpus = """Humanismo es un concepto polisémico que se aplica tanto al estudio de las letras humanas, los
estudios clásicos y la filología grecorromana como a una genérica doctrina o actitud vital que
concibe de forma integrada los valores humanos. Por otro lado, también se denomina humanis-
mo al «sistema de creencias centrado en el principio de que las necesidades de la sensibilidad
y de la inteligencia humana pueden satisfacerse sin tener que aceptar la existencia de Dios
y la predicación de las religiones», lo que se aproxima al laicismo o a posturas secularistas.
Se aplica como denominación a distintas corrientes filosóficas, aunque de forma particular,
al humanismo renacentista1 (la corriente cultural europea desarrollada de forma paralela al
Renacimiento a partir de sus orígenes en la Italia del siglo XV), caracterizado a la vez por su
vocación filológica clásica y por su antropocentrismo frente al teocentrismo medieval
"""
words = mini_corpus.replace("\n", " ").split(" ")
len(words)

vocabulary = Counter(words)
vocabulary.most_common(10)

len(vocabulary)


# +
def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

def plot_frequencies(frequencies: list, title="Freq of words", log_scale=False):
    x = list(range(1, len(frequencies)+1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)


# -

frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)

plot_frequencies(frequencies, log_scale=True)

# **¿Qué pasará con más datos? 📊**

# ### Ley Zipf

# Utilizaremos un corpus en español, entre más grande mejor.

dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

# +
# dataset?
# -

row = next(iter(dataset))

print(row["title"])
print(row["text"])

corpus = dataset.take(10000)

# +
import re

def normalize_corpus(example):
    example["text"] = re.sub(r"[\W]", " ", example["text"])
    example["text"] = example["text"].lower()
    return example


# +
from datasets.iterable_dataset import IterableDataset


def count_words(corpus: IterableDataset) -> Counter:
    word_counts = Counter()
    normalized_corpus = corpus.map(normalize_corpus)
    for row in normalized_corpus:
        text = row["text"]
        word_counts.update(text.split())
    return word_counts


# -

# %%time
words = count_words(corpus)

words.most_common(10)

# +
import pandas as pd

def counter_to_pandas(counter: Counter) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(counter, orient="index").reset_index()
    df.columns = ["word", "count"]
    df.sort_values("count", ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# -

corpus_freqs = counter_to_pandas(words)

corpus_freqs.head(10)

corpus_freqs[corpus_freqs["word"] == "barriga"]

corpus_freqs["count"].plot(marker="o", legend=False)
plt.title('Ley de Zipf en el CREA')
plt.xlabel('rank')
plt.ylabel('freq')
plt.show()

corpus_freqs['count'].plot(loglog=True, legend=False)
plt.title('Ley de Zipf en el CREA (log-log)')
plt.xlabel('log rank')
plt.ylabel('log frecuencia')
plt.show()

# - Notamos que las frecuencias entre lenguas siguen un patrón
# - Pocas palabras (tipos) son muy frecuentes, mientras que la mayoría de palabras ocurren pocas veces
#
# De hecho, la frecuencia de la palabra que ocupa la posición r en el rank, es proporcional a $\frac{1}{r}$ (La palabra más frecuente ocurrirá aproximadamente el doble de veces que la segunda palabra más frecuente en el corpus y tres veces más que la tercer palabra más frecuente del corpus, etc)
#
# $$f(w_r) \propto \frac{1}{r^α}$$
#
# Donde:
# - $r$ es el rank que ocupa la palabra en el corpus
# - $f(w_r)$ es la frecuencia de la palabra en el corpus
# - $\alpha$ es un parámetro, el valor dependerá del corpus o fenómeno que estemos observando

# #### Formulación de la Ley de Zipf:

# $f(w_{r})=\frac{c}{r^{\alpha }}$
#
# En la escala logarítimica:
#
# $log(f(w_{r}))=log(\frac{c}{r^{\alpha }})$
#
# $log(f(w_{r}))=log (c)-\alpha log (r)$

# #### ❓ ¿Cómo estimar el parámetro $\alpha$?

# Podemos hacer una regresión lineal minimizando la suma de los errores cuadráticos:
#
# $J_{MSE}=\sum_{r}^{}(log(f(w_{r}))-(log(c)-\alpha log(r)))^{2}$

# +
from scipy.optimize import minimize

ranks = np.array(corpus_freqs.index) + 1
frecs = np.array(corpus_freqs['count'])

# Inicialización
a0 = 1

# Función de minimización:
func = lambda a: sum((np.log(frecs)-(np.log(frecs[0])-a*np.log(ranks)))**2)

# Apliando minimos cuadrados
a_hat = minimize(func, a0).x[0]

print('alpha:', a_hat, '\nMSE:', func(a_hat))


# -

def plot_generate_zipf(alpha: np.float64, ranks: np.array, freqs: np.array) -> None:
    plt.plot(np.log(ranks),  np.log(freqs[0]) - alpha*np.log(ranks), color='r', label='Aproximación Zipf')


plot_generate_zipf(a_hat, ranks, frecs)
plt.plot(np.log(ranks),np.log(frecs), color='b', label='Distribución CREA')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# #### 📊 Ejercicio: Verificando ley de Zipf
#
# - Busca un corpus en otra lengua (no español) en hugging face y descargalo
#     - Si es muy grande toma una muestra
# - Estima su parámetro $\alpha$
# - Verifica a ojo de buen cubero si se cumple la ley de Zipf

# ### Ley de Heap

# Relación entre el número de **tokens** y **tipos** de un corpus
#
# $$T \propto N^b$$
#
# Dónde:
#
# - $T = $ número de tipos
# - $N = $ número de tokens
# - $b = $ parámetro  

# - **TOKENS**: Número total de palabras dentro del texto (incluidas repeticiones)
# - **TIPOS**: Número total de palabras únicas en el texto

# #### 📊 Ejercicio: Muestra el plot de tokens vs types para el corpus CREA
#
# **HINT:** Obtener tipos y tokens acumulados

# ## Diversidad lingũística 


