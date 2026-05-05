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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Preprocesamiento y tokenización

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/6_tokenization.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
# <img src="https://2.bp.blogspot.com/-oDvCIkIjwXw/VdWWxfvmq3I/AAAAAAAARUE/r0MrmbNzMz8/s1600/inputoutput.jpg" width=500>

# %% [markdown]
# ## Objetivos

# %% [markdown]
# - Listar algunos pasos comúnes para el preprocesamiento de texto
#   - Aplicar preprocesamiento a un corpus
# - Entender el funcionamiento de algoritmos de sub-word tokenization
#   - BPE
#   - Word-piece
#   - Sentecepiece

# %% [markdown]
# ## El lenguaje, datos inherentemente desarreglados

# %% [markdown]
# Los datos con los que trabajamos son inherentemente **desestructurados** y en general contienen ruido, irregularidades, e inconsistencias.
#
# Los modelos de NLP son sensibles a estos problemas y en realidad no pueden utilizar el texto directamente. Dado que el texto es una representación simbólica de la información, es necesario convertirlo a una representación numérica que el modelo pueda utilizar. Esta representación generalmente serán **vectores** con valores reales llamados *embeddings*.
#
# El objetivo es crear un *pipeline* que transforme el texto crudo en *embeddigs* que serán utilizados como entrada para crear modelos que resuelvan alguna tarea de NLP.

# %% [markdown]
# ### *Pipelines*

# %% [markdown]
# ![](https://i.makeagif.com/media/11-05-2015/x60GaR.gif)

# %% [markdown]
# Al crear sistemas de *NLP* nos enfrentamos con problemas complejos. Conviene entonces separar dichos problemas en pequeños problemas que podamos manejar y resolver por separado.
#
# El preprocessamiento es el primer paso de este proceso. Otros elementos pueden ser los siguientes:
#
# - Definición del problema
# - Adquisición de datos
# - Ingeniería de características (*feature engineering*)
# - Modelado
#     - Definición de hiperparametros
# - Entrenamiento
# - Evaluación
# - Puesta en producción
# - Monitorización y actualización del modelo

# %% [markdown]
# ## Elementos del preprocesamiento

# %% [markdown]
# - Limpieza del texto
#     - Quitar etiquetas de marcado (HTML, XML, MD), metadatos y asegurarnos que todo esta en UTF-8
#     - Eliminar header, footers o titulos que no aportan información
# - Normalización
#     - Pasar todo a minúsculas
#     - Pasar texto a cierta norma ortográfica
#     - Expansión de contracciones o abreviaciones
# - Quitar stopwords y lematización/stemming
# - Tokenización
#     - Por palabra
#     - Por letras
#     - Por sub-palabras
# - *Embeddings*
#     - Los modelos solo entienden números, por lo que hay que convertir el texto a una representación vectorial

# %% [markdown]
# ### Limpieza de textos

# %% [markdown]
# Es común usar regex o bibliotecas como `BeautifulSoup` para limpiar el texto de etiquetas de marcado

# %%
import requests
from bs4 import BeautifulSoup
from rich import print as rprint

# %%
url = "https://elotl.mx/blog/index.html"
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")
posts = soup.find("div", class_="widget_onetone_recent_posts")

if posts:
    rows = posts.find_all("li")
    for row in rows:
        text = row.get_text()
        print(text)

# %%
import nltk

nltk.download("gutenberg")

# %%
from nltk.corpus import gutenberg

moby = gutenberg.raw("melville-moby_dick.txt")

# %%
print(moby[:30000])

# %%
import re
from nltk.tokenize import sent_tokenize


def clean_and_extract_sentences(text: str) -> list[str]:
    """Clean preamble text from Moby Dick and extract sentences from the novel."""
    novel_start_patterns = [
        r"CHAPTER\s+1\b",  # Matches "CHAPTER 1"
        r"Call\s+me\s+Ishmael",  # Matches "Call me Ishmael"
    ]

    # Buscamos el índice donde comienza la novelaS
    novel_start = None
    for pattern in novel_start_patterns:
        match = re.search(pattern, text)
        if match and (novel_start is None or match.start() < novel_start):
            novel_start = match.start()

    if novel_start is None:
        return []

    # Descartamos el preambulo
    novel_text = text[novel_start:]

    # Limpiamos el texto
    novel_text = re.sub(r"CHAPTER\s+\d+", "", novel_text)
    novel_text = re.sub(r"\s+", " ", novel_text).strip()
    novel_text = re.sub(r"\[.*?\]", "", novel_text)

    # Extraemos las oraciones
    sentences = sent_tokenize(novel_text)

    # Limpieza adicional
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = re.sub(r"^[^a-zA-Z]+", "", sentence)
        sentence = re.sub(r"[^a-zA-Z]+$", "", sentence)

        if sentence:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


# %%
sentences = clean_and_extract_sentences(moby)
for i, sentence in enumerate(sentences[:10], 1):
    rprint(f"{i}. {sentence}")

# %% [markdown]
# ### Normalización

# %% [markdown]
# <center><img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimg1.wikia.nocookie.net%2F__cb20140504152558%2Fspongebob%2Fimages%2Fe%2Fe3%2FThe_spongebob.jpg&f=1&nofb=1&ipt=28368023b54a7c84c9100025981b1042d0f4ca3ceaac53be42094cc1c3794348&ipo=images" height=300 width=300></center>

# %%
import unicodedata


def strip_accents(s: str) -> str:
    """Remove diacritical marks from characters in a Unicode string.

    Uses Unicode NFD (Normalization Form Decomposition) normalization to decompose accented characters into their
    base character + combining mark, then filters out combining marks (Mark, Nonspacing, Mn category).
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# %%
rprint(
    strip_accents("""Éxtasis
E-, e-, e-, e-, e-, e-, e-
Éxtasis
Éxtasis

Aquí no existe el bajón
Tamos' de fiestón, ya sabes
Despierta la inspiración
Si sacamos las suaves""")
)

# %% [markdown]
# - https://www.unicode.org/reports/tr44/#GC_Values_Table
#
# > And keep in mind, these manipulations may significantly alter the meaning of the text. Accents, Umlauts etc. are not "decoration".
# - [oefe](https://stackoverflow.com/users/49793/oefe) - [source](https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string)

# %% [markdown]
# #### ¿Para otras lenguas?

# %% [markdown]
# - No hay muchos recursos :(
# - Pero para el nahuatl esta `pyelotl` :)

# %% [markdown]
# #### Normalizando el Nahuatl

# %%
# !pip install elotl

# %%
import elotl.corpus
import elotl.nahuatl.orthography

# %%
axolotl = elotl.corpus.load("axolotl")

# %%
# Tres posibles normalizadores: sep, inali, ack
# Sauce: https://pypi.org/project/elotl/

nahuatl_normalizer = elotl.nahuatl.orthography.Normalizer("sep")

# %%
rprint(axolotl[1][1])

# %%
rprint(nahuatl_normalizer.normalize(axolotl[1][1]))

# %%
nahuatl_normalizer.to_phones(axolotl[1][1])

# %% [markdown]
# ### Stopwords

# %%
from nltk.corpus import stopwords

# %%
nltk.download("stopwords")

# %%
rprint(stopwords.words("spanish")[:15])


# %% [markdown]
# ### Definiendo una función de preprocesamiento

# %%
def preprocess(words: list[str], regex: str=r"[^\w+]", lang: str="en", remove_stops: bool = False, remove_accents: bool = False) -> list[str]:
    """Preprocess step for list of words in corpus
    """
    stop_lang = "english" if lang=="en" else "spanish"
    result = []
    for word in words:
        word = re.sub(regex, "", word).lower()
        if remove_stops and word in stopwords.words(stop_lang) or (len(word) < 2):
            continue

        if remove_accents:
            word = strip_accents(word)
        result.append(word)
    return result


# %% [markdown]
# ## ¿Cuántas palabras hay en las siguientes oraciones?

# %%
sentence_trapo = "Quitan el trapo y no lo ponen. ¿Por qué quitan el trapo? Si es una cosa que debe estar ahí."

# %%
sentence_trapo.split()

# %%
sentence_sad = "Mmmmm haz lo que quieras... pero no me digas que no te lo advertí 😓"

# %%
len(sentence_sad.split())

# %% [markdown]
# - A estas alturas tenemos cierta información acerca de las palabras:
#     - **typos:** Número de palabras únicas en un corpus. *AKA* vocabulario
#     - **tokens:** Número total de palabras. *AKA* instancias

# %% [markdown]
# ## ¿Qué es una palabra?

# %% [markdown]
# - Técnicas de procesamiento del lenguaje depende de las palabras y las oraciones.
#   - Debemos identificar estos elementos para poder procesarlos
# - Este paso de identificación de palabras y oraciones se le llama segmentación de texto o **tokenización** (*tokenization*)
# - Además de la identificación de unidades aplicaremos transformaciones al texto

# %% [markdown]
# ### Más que mil palabras

# %% [markdown]
# Aunque la definición de lo que es una palabra puede parecer obvia a la hora de diseñar sistemas de PLN puede ser tremendamente difícil.

# %% [markdown]
# - I'm
# - we'd
# - I've
# - Diego's Bicycle

# %% [markdown]
# En lenguas donde los espacios no son utilizados para marcar posibles delimitaciones entre palabras la cosa se pone más dura:
#
# - 姚明进入总决赛 - yáo míng jìn rù zong jué sài
# - *"Yao Ming llegó a las finales"*
#
# > Tomado de (Jurafsky, 2026)

# %% [markdown]
# Chinese Treebank:
#
# 1. 姚明 - Yao Ming
# 2. 进入 - llego a
# 3. 总决赛 - finales

# %% [markdown]
# Peking University:
#
# 1. 姚 - Yao
# 2. 明 - Ming
# 3. 进入 - llego
# 4. 总 - generales
# 5. 决赛 - finales

# %% [markdown]
# Caracteres como límites
#
# 1. 姚 - Yao
# 2. 明 - Ming
# 3. 进 - entrar
# 4. 入 - entrar
# 5. 总 - generales
# 6. 决 - decisión
# 7. 赛 - juego

# %% [markdown]
# Otro problema a considerar es la cantidad de palabras con la que tendran que lidiar los modelos que diseñemos. Por más texto que tengamos a disposición siempre habrán palabras que el modelo no habrá visto (*AKA* **Out of Vocabulary, OOV** o **\<UNK\>**)

# %% [markdown]
# ### Recordando los morfemas

# %% [markdown]
# - Con la morfología podemos identificar como se modifica el significado variando la estructura de las palabras
# - Tambien las reglas para producir:
#     - niño -> niños
#     - niño -> niña
# - Tenemos elementos mínimos, intercambiables que varian el significado de las palabras: **morfemas**
#
# > Un morfema es la unidad mínima con significado en la producción lingüística (Mijangos, 2020)

# %% [markdown]
# #### Tipos de morfemas

# %% [markdown]
# - Bases: Subcadenas que aportan información léxica de la palabra
#     - sol
#     - frasada
# - Afijos: Subcadenas que se adhieren a las bases para añadir información (flexiva, derivativa)
#     - Prefijos
#         - *in*-parable
#     - Subfijos
#         - pan-*ecitos*, come-*mos*

# %% [markdown]
# ## Tokenización

# %% [markdown]
# ### Word-base tokenization

# %%
text = """
¡¡¡Mamá prendele a la grabadora!!!, ¿llamaste a las vecinas? Corre la voz porque, efectivamente, !estoy en la tele! 📺
"""

# %%
text.split()

# %%
# [a-zA-Z_]\
regex = r"\w+|[?¿!¡]"
re.findall(regex, text)

# %%
re.findall(regex, "El valor de PI es 3.14159")

# %% [markdown]
# <img src="http://images.wikia.com/battlebears/images/2/2c/Troll_Problem.jpg" with="250" height="250">

# %% [markdown]
# - Vocabularios gigantescos difíciles de procesar
# - Generalmente, entre más grande es el vocabulario más pesado será nuestro modelo
#
# **Ejemplo:**
# - Si queremos representaciones vectoriales de nuestras palabras obtendríamos vectores distintos para palabras similares
#     - niño = `v1(39, 34, 5,...)`
#     - niños = `v2(9, 4, 0,...)`
#     - niña = `v3(2, 1, 1,...)`
#     - ...
# - Tendríamos tokens con bajísima frecuencia
#     - merequetengue = `vn(0,0,1,...)`

# %% [markdown]
# ### Una solución: Steaming/Lematización (AKA la vieja confiable)

# %% [markdown]
# ![](https://i.pinimg.com/736x/77/df/89/77df89e6ff57d332ba4e5d7bff723133--meme.jpg)

# %%
nltk.download("brown")

# %%
from nltk.corpus import brown

# %%
brown_corpus = preprocess(brown.words()[:100000], lang="en", remove_stops=True)

# %%
rprint(brown_corpus[:10])

# %%
from collections import Counter

rprint("[bright_yellow]Brown Vanilla")
rprint("Tokens:", len(brown.words()))
rprint("Tipos:", len(Counter(brown.words())))

rprint("[bright_green]Brown Preprocess")
rprint("Tokens:", len(brown_corpus))
rprint("Tipos:", len(Counter(brown_corpus)))

# %% [markdown]
# #### Steamming

# %% [markdown]
# - Chiquitititos - chico - Chiqu

# %%
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

# %%
stemmed_brown = [stemmer.stem(word) for word in brown_corpus]

# %%
stemmed_brown[:10]

# %% [markdown]
# #### Lematización

# %%
# !python -m spacy download en_core_web_md
# !python -m spacy download es_core_news_md

# %%
import spacy


def lemmatize(words: list, lang: str = "en") -> list:
    model = "en_core_web_md" if lang == "en" else "es_core_news_md"
    nlp = spacy.load(model)
    nlp.max_length = 1500000
    lemmas = []
    for doc in nlp.pipe([" ".join(words)], batch_size=500):
        lemmas.extend([token.lemma_ for token in doc if not token.is_space])
    return lemmas


# %%
lemmatized_brown = lemmatize(brown_corpus, lang="en")

# %%
lemmatized_brown[:10]

# %%
rprint("Tipos ([bright_magenta]word-based[/]):", len(Counter(brown_corpus)))
rprint("Tipos ([bright_yellow]Steamming[/]):", len(Counter(stemmed_brown)))
rprint("Tipos ([bright_green]Lemmatized[/]):", len(Counter(lemmatized_brown)))

# %% [markdown]
# #### More problems?
#
# <img src="https://uploads.dailydot.com/2019/10/Untitled_Goose_Game_Honk.jpeg?auto=compress%2Cformat&ixlib=php-3.3.0" width="250" height="250">

# %% [markdown]
# - Métodos dependientes de las lenguas
# - Se pierde información
# - Ruled-based

# %% [markdown]
# ## Subword-tokenization salva el día 🦸🏼‍♀️

# %% [markdown]
# - Segmentación de palabras en unidades más pequeñas (*sub-words*)
# - Obtenemos tipos menos variados y con mayores frecuencias
#     - Esto le gusta modelos basados en métodos estadísticos
# - Palabras frecuentes no deberían separarse
# - Palabras largas y raras debería descomponerse en sub-palabras significativas
# - Los métodos estadisticos que no requieren conocimiento a priori de las lenguas

# %%
text = "Let's do tokenization!"
result = ["Let's", "do", "token", "ization", "!"]
print(f"Objetivo: {text} -> {result}")

# %% [markdown]
# ### Algoritmos

# %% [markdown]
# Existen varios algoritmos para hacer *subword-tokenization* como los que se listan a continuación:
#
# - Byte-Pair Encoding (BPE)
# - WordPiece
# - Unigram

# %% [markdown]
# #### BPE

# %% [markdown]
# - Segmenmentación iterativa, comienza segmentando en secuencias de caracteres
# - Junta los pares más frecuentes (*merge operation*)
# - Termina cuando se llega al número de *merge operations* especificado o número de vocabulario deseado (*hyperparams*, depende de la implementación)
# - Introducido en el paper: [Neural Machine Translation of Rare Words with Subword Units, (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)

# %%
# %%HTML
<iframe width="960" height="515" src="https://www.youtube.com/embed/HEikzVL-lZU"></iframe>

# %% [markdown]
# ### Implementación de BPE
#
# > Basado en el tutorial de HF - TODO LINK

# %%
corpus = clean_and_extract_sentences(moby)

# %%
from nltk.tokenize import word_tokenize

# %%
from collections import defaultdict


word_freqs = defaultdict(int)

for sent in corpus:
    words = word_tokenize(sent)
    for word in words:
        word_freqs[word] += 1

# %%
print(word_freqs)

# %%
alphabet = set()

for word in word_freqs.keys():
    for char in word:
        if char not in alphabet:
            alphabet.add(char)

# %%
print(alphabet)

# %%
splits = {word: [c for c in word] for word in word_freqs.keys()}

# %%
print(splits)


# %% [markdown]
# #### Creando el modelo

# %%
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


# %%
pair_freqs = compute_pair_freqs(splits)

# %%
for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# %%
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)


# %% [markdown]
# Aplicamos el merge más común

# %%
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


# %%
vocab = []
merges = {}
vocab_size = 500

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])


# %%
def tokenize(text):
    words = word_tokenize(text)
    splits = [[c for c in word] for word in words]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])


# %%
for token in tokenize("This is necessary for you my heaviest friendly friend"):
    print(token)

# %%
# !pip install transformers

# %%
SENTENCE = "Let's do this tokenization to enable hypermodernization on my tokens tokenized 👁️👁️👁️!!!"

# %%
from transformers import GPT2Tokenizer

bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(bpe_tokenizer.tokenize(SENTENCE))

# %%
encoded_tokens = bpe_tokenizer(SENTENCE)
rprint(encoded_tokens["input_ids"])

# %%
rprint(bpe_tokenizer.decode(encoded_tokens["input_ids"]))

# %% [markdown]
# - En realidad GPT-2 usa *Byte-Level BPE*
#     - Evitamos vocabularios de inicio grandes (Ej: unicode)
#     - Usamos bytes como vocabulario base
#     - Evitamos *Out Of Vocabulary, OOV* (aka `[UKW]`)

# %% [markdown]
# #### WordPiece

# %% [markdown]
# - Descrito en el paper: [Japanese and Korean voice search, (Schuster et al., 2012) ](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
# - Similar a BPE, inicia el vocabulario con todos los caracteres y aprende los merges
# - En contraste con BPE, no elige con base en los pares más frecuentes si no los pares que maximicen la probabilidad de aparecer en los datos una vez que se agregan al vocabulario
#
# $$score(a_i,b_j) = \frac{f(a_i,b_j)}{f(a_i)f(b_j)}$$
#
# - Esto quiere decir que evalua la perdida de realizar un *merge* asegurandoce que vale la pena hacerlo
#
# - Algoritmo usado en `BERT`

# %%
# %%HTML
<iframe width="960" height="500" src="https://www.youtube.com/embed/qpv6ms_t_1A"></iframe>

# %%
from transformers import BertTokenizer

SENTENCE = "🌽" + SENTENCE + "🔥"
wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
rprint(wp_tokenizer.tokenize(SENTENCE))

# %% [markdown]
# <center><img src="https://us-tuna-sounds-images.voicemod.net/9cf541d2-dd7f-4c1c-ae37-8bc671c855fe-1665957161744.jpg"></center>

# %%
rprint(wp_tokenizer(SENTENCE))

# %% [markdown]
# #### Unigram

# %% [markdown]
# - Algoritmo de subpword tokenization introducido en el paper: [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018)](https://arxiv.org/pdf/1804.10959.pdf)
# - En contraste con BPE o WordPiece, este algoritmo inicia con un vocabulario muy grande y va reduciendolo hasta llegar tener un vocabulario deseado
# - En cada iteración se calcula la perdida de quitar cierto elemento del vocabulario
#     - Se quitará `p%` elementos que menos aumenten la perdida en esa iteración
# - El algoritmo termina cuando se alcanza el tamaño deseado del vocabulario

# %% [markdown]
# Sin embargo, *Unigram* no se usa por si mismo en algun modelo de Hugging Face:
# > "Unigram is not used directly for any of the models in the transformers, but it’s used in conjunction with SentencePiece." - Hugging face guy

# %% [markdown]
# #### SentencePiece
#

# %% [markdown]
# - No asume que las palabras estan divididas por espacios
# - Trata la entrada de texto como un *stream* de datos crudos. Esto incluye al espacio como un caractér a usar
# - Utiliza BPE o Unigram para construir el vocabulario

# %%
# https://github.com/google/sentencepiece#installation
# !pip install sentencepiece

# %%
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
rprint(tokenizer.tokenize(SENTENCE))

# %% [markdown]
# #### Objetivo de los subword tokenizers
#

# %% [markdown]
# - Buscamos que modelos de redes neuronales tenga datos mas frecuentes
# - Esto ayuda a que en principio "aprendan" mejor
# - Reducir el numero de tipos
# - Reducir el numero de OOV

# %%
