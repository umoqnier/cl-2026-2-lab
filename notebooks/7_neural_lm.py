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
# # 7. Modelos:  BPE, Embeddings, Neural LM

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/7_neural_lm.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
# ## Objetivos
#
# - Entrenar modelos para sub-word tokenization
#   - Aplicar BPE a corpus
# - Entrenar modelos para *embeddings*
#   - Word2Vec
#   - Glove
# - Implementación de modelo del lenguaje Neuronal de Bengio
#   - Generación de lenguaje

# %%
import os
import re
from rich import print as rprint
from nltk import word_tokenize
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

# %% [markdown]
# ## Funciones de preprocesamiento

# %%
from nltk.corpus import stopwords
import unicodedata


def strip_accents(s: str) -> str:
    """Remove diacritical marks from characters in a Unicode string.

    Uses Unicode NFD (Normalization Form Decomposition) normalization to decompose accented characters into their
    base character + combining mark, then filters out combining marks (Mark, Nonspacing, Mn category).
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_words(
    words: list[str],
    regex: str = r"[^\w+]",
    lang: str = "en",
    remove_stops: bool = False,
    remove_accents: bool = False,
) -> list[str]:
    """Preprocess step for list of words in corpus"""
    stop_lang = "english" if lang == "en" else "spanish"
    result = []
    for word in words:
        word = re.sub(regex, "", word).lower()
        if remove_stops and word in stopwords.words(stop_lang) or (len(word) < 2):
            continue

        if remove_accents:
            word = strip_accents(word)
        result.append(word)
    return result


def preprocess_text(text: str, to_lower: bool = True) -> str:
    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    if to_lower:
        text = text.lower()

    # 3. Collapse all whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    # 4. Clean up leading/trailing whitespace
    text = text.strip()

    return text


# %% [markdown]
# ## Byte-Pair Encoding

# %% [markdown]
# ### Vamos a tokenizar 🌈
# ![](https://i.pinimg.com/736x/58/6b/88/586b8825f010ce0e3f9c831f568aafa8.jpg)

# %%
BASE_PATH = "."
CORPORA_PATH = f"{BASE_PATH}/data/"
MODELS_PATH = f"{BASE_PATH}/models/"

os.makedirs(CORPORA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


# %%
TOKENIZERS_DATA_PATH = f"{MODELS_PATH}/tokenization"
TOKENIZERS_MODEL_PATH = f"{TOKENIZERS_DATA_PATH}/sub-word"

os.makedirs(TOKENIZERS_DATA_PATH, exist_ok=True)
os.makedirs(TOKENIZERS_MODEL_PATH, exist_ok=True)

# %% [markdown]
# ### Corpus en español: Wikipedia

# %%
from datasets import load_dataset, load_dataset_builder

# %%
data_builder = load_dataset_builder("wikimedia/wikipedia", "20231101.es")

# %%
rprint(data_builder.info)

# %%
dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.es", split="train", streaming=True
)

# %%
wiki_words = []
for article in dataset.take(1):
    rprint(preprocess_text(article["text"][:1000], to_lower=False))

# %%
# %%time

wiki_file_path = f"{CORPORA_PATH}/wikipedia_es_plain.txt"
with open(wiki_file_path, "w", encoding="utf-8") as f:
    for article in dataset.take(1000):
        f.write(preprocess_text(article["text"]))
        f.write("\n")

# %%
# !head -n 10 {CORPORA_PATH}/wikipedia_es_plain.txt

# %% [markdown]
# ### Entrenando nuestro modelo con BPE
#
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fd565618bb1217a7c435579d9172270d0%2Ftenor.gif%3Fitemid%3D3379322&f=1&nofb=1&ipt=9719714edb643995ce9d978c8bab77f5310204960093070e37e183d5372096d9&ipo=images)

# %%
# !pip install subword-nmt

# %%
# !ls {CORPORA_PATH}

# %%
# !subword-nmt --help

# %%
# !subword-nmt learn-bpe --help

# %%
# %%time

# !subword-nmt learn-bpe --num-workers -1 -s 300 < \
#  {CORPORA_PATH}/wikipedia_es_plain.txt > \
#   {MODELS_PATH}/wiki_es_300.model

# %%
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c {MODELS_PATH}/wiki_es_300.model

# %%
# %%time

# !subword-nmt learn-bpe --num-workers -1 -s 10000 < \
# data/tokenization/wikipedia_es_plain.txt > \
#  models/sub-word/wiki_es_10k.model

# %%
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c models/sub-word/wiki_es_10k.model

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(
    " ".join(
        tokenizer.tokenize(
            "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano"
        )
    ).replace("Ġ", "@@")
)

# %% [markdown]
# ### Aplicandolo a otros corpus: La biblia 📖🇻🇦

# %%
BIBLE_FILE_NAMES = {
    "spa": "spa-x-bible-reinavaleracontemporanea",
    "eng": "eng-x-bible-kingjames",
}

# %%
import requests


def get_bible_corpus(lang: str) -> str:
    """Download bible file corpus from GitHub repo"""
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(
        f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt"
    )
    return r.text


def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    """Write file text on disk"""
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)


# %% [markdown]
# #### Tokenizando biblia en español

# %%
spa_bible_raw = get_bible_corpus("spa")
spa_bible_plain_text = preprocess_text(spa_bible_raw)

# %%
write_plain_text_corpus(spa_bible_plain_text, f"{CORPORA_PATH}/bible-spa")

# %%
# !subword-nmt apply-bpe -c {MODELS_PATH}/wiki_es_10k.model < \
#  {CORPORA_PATH}/bible-spa.txt > \
#  {CORPORA_PATH}/bible-spa-tokenized.txt

# %% [markdown]
# #### Comparando resultados

# %%
spa_bible_words = word_tokenize(spa_bible_plain_text)

# %%
spa_bible_words[:10]

# %%
len(spa_bible_words)

# %%
spa_bible_types = Counter(spa_bible_words)
len(spa_bible_types)

# %%
spa_bible_types.most_common(30)

# %%
with open(f"{CORPORA_PATH}/bible-spa-tokenized.txt", "r") as f:
    tokenized_text = f.read()
spa_bible_tokenized = tokenized_text.split()

# %%
spa_bible_tokenized[:10]

# %%
len(spa_bible_tokenized)

# %%
spa_bible_tokenized_types = Counter(spa_bible_tokenized)
len(spa_bible_tokenized_types)

# %%
spa_bible_tokenized_types.most_common(40)

# %%
rprint("Biblia español")
rprint(f"Tipos ([bright_magenta]word-base[/]): {len(spa_bible_types)}")
rprint(f"Tipos ([bright_green]sub-word[/]): {len(spa_bible_tokenized_types)}")

# %% [markdown]
# #### OOV: out of vocabulary

# %% [markdown]
# Palabras que se vieron en el entrenamiento pero no estan en el test

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    spa_bible_words, test_size=0.3, random_state=42
)
rprint(len(train_data), len(test_data))

# %%
s_1 = {"a", "b", "c", "d", "e"}
s_2 = {"a", "x", "y", "d"}
rprint(s_1 - s_2)
rprint(s_2 - s_1)

# %%
oov_test = set(test_data) - set(train_data)

# %%
for word in list(oov_test)[:3]:
    rprint(f"{word} in train: {word in set(train_data)}")

# %%
train_tokenized, test_tokenized = train_test_split(
    spa_bible_tokenized, test_size=0.3, random_state=42
)
rprint(len(train_tokenized), len(test_tokenized))

# %%
oov_tokenized_test = set(test_tokenized) - set(train_tokenized)

# %%
rprint("OOV ([bright_magenta]word-base[/]):", len(oov_test))
rprint("OOV ([bright_green]sub-word[/]):", len(oov_tokenized_test))

# %% [markdown]
# ### Type-token Ratio (TTR)
#
# - Una forma de medir la variación del vocabulario en un corpus
# - Este se calcula como $TTR = \frac{len(types)}{len(tokens)}$
# - Puede ser útil para monitorear la variación lexica de un texto

# %%
stemmer = SnowballStemmer("spanish")
spa_bible_stemmed = [stemmer.stem(word) for word in spa_bible_words]
spa_bible_stemmed_types = set(spa_bible_stemmed)

# %%
rprint("Bible Spanish Information")
rprint("Tokens:", len(spa_bible_words))
rprint("Types ([bright_magenta]word-base[/]):", len(spa_bible_types))
rprint("Types ([bright_yellow]stemmed[/])", len(spa_bible_stemmed_types))
rprint("Types ([bright_green]BPE[/]):", len(spa_bible_tokenized_types))
rprint(
    "TTR ([bright_magenta]word-base[/]):", len(spa_bible_types) / len(spa_bible_words)
)
rprint(
    "TTR ([bright_yellow]stemmed[/]):",
    len(spa_bible_stemmed_types) / len(spa_bible_stemmed),
)
rprint(
    "TTR ([bright_green]BPE[/]):",
    len(spa_bible_tokenized_types) / len(spa_bible_tokenized),
)

# %% [markdown]
# ## Word Embeddings (W2V, GloVe) 

# %% [markdown]
# Vamos a entrenar nuestras propias representaciones vectoriales

# %% [markdown]
# ![we](https://miro.medium.com/v2/resize:fit:2000/1*SYiW1MUZul1NvL1kc1RxwQ.png)

# %% [markdown]
# ### Datos: Noticias en Español

# %%
news_databuilder = load_dataset_builder("LeoCordoba/CC-NEWS-ES", "mx")

# %%
rprint(news_databuilder.info)

# %%
news_dataset = load_dataset(
    "LeoCordoba/CC-NEWS-ES", "mx", split="train", streaming=True
)

# %%
for post in news_dataset.take(1):
    rprint(post["text"])

# %%
from gensim.utils import simple_preprocess

rprint(simple_preprocess(post["text"], deacc=True)[:10])

# %%
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from tqdm.notebook import tqdm
from datasets import load_dataset
from gensim.utils import simple_preprocess


class CCNewsExtractor:
    """
    Iterador optimizado para CC-NEWS-ES + Word2Vec.
    Diseñado para alta velocidad y compatibilidad con los epochs de Gensim.
    """

    def __init__(self, lang: str = "mx", max_posts: int = -1):
        self.dataset = load_dataset(
            "LeoCordoba/CC-NEWS-ES", name=lang, split="train", streaming=True
        )
        self.max_posts = max_posts

        # Precompilar la expresión regular es considerablemente más rápido
        # en bucles anidados que inicializarla en cada pasada.
        self.sent_splitter = re.compile(r"[.!?\n]+")

    def __iter__(self):
        for item in tqdm(self.dataset.take(self.max_posts)):
            text = item.get("text", "")
            if not text:
                continue

            words = simple_preprocess(text, deacc=False, min_len=2)

            if not words:
                continue
            yield words


# %%
# Uso con tu función train_model
iterator = CCNewsExtractor(lang="mx", max_posts=3)

# %%
for i in iterator:
    rprint(i[:10])

# %%
# %%time
sentences = CCNewsExtractor(lang="mx", max_posts=10)

# %%
for sentence in sentences:
    print(sentence)

# %%
from gensim.models import word2vec

# %%
EMB_MODELS_DIR = os.path.join(MODELS_PATH, "embeddings")

os.makedirs(EMB_MODELS_DIR, exist_ok=True)

# %%
from enum import Enum


class Algorithms(Enum):
    CBOW = 0
    SKIP_GRAM = 1


# %%
def load_model(model_path: str):
    try:
        print(model_path)
        return word2vec.Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"[WARN] Model not found in path {model_path}")
        return None


# %%
def train_model(
    sentences: list,
    model_name: str,
    vector_size: int,
    window=5,
    workers=2,
    algorithm=Algorithms.CBOW,
):
    model_name_params = f"{model_name}-vs{vector_size}-w{window}-{algorithm.name}.model"
    model_path = os.path.join(EMB_MODELS_DIR, model_name_params)
    if load_model(model_path) is not None:
        print(f"Already exists the model {model_path}")
        return load_model(model_path)
    print(f"TRAINING: {model_path}")
    if algorithm in [Algorithms.CBOW, Algorithms.SKIP_GRAM]:
        model = word2vec.Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            workers=workers,
            sg=algorithm.value,
            seed=42,
        )
    else:
        print("[ERROR] algorithm not implemented yet :p")
        return
    try:
        model.save(model_path)
    except:
        print(f"[ERROR] Saving model at {model_path}")
    return model


# %% [markdown]
# ### CBOW

# %%
skipm_gram_model = load_model(
    os.path.join(EMB_MODELS_DIR, "eswiki-xl-300-SKIP_GRAM.model")
)

# %%
# %%time
cbow_model = train_model(
    CCNewsExtractor(lang="mx", max_posts=100_000),
    "eswiki",
    vector_size=100,
    window=3,
    workers=6,
    algorithm=Algorithms.CBOW,
)

# %% [markdown]
# ### Skip gram

# %%
# %%time
skip_gram_model = train_model(
    CCNewsExtractor(lang="mx", max_posts=100_000),
    "es_news_hf",
    300,
    5,
    workers=12,
    algorithm=Algorithms.SKIP_GRAM,
)


# %%
def report_stats(model) -> None:
    """Print report of a model"""
    print(
        "Number of words in the corpus used for training the model: ",
        model.corpus_count,
    )
    print("Number of words in the model: ", len(model.wv.index_to_key))
    print("Time [s], required for training the model: ", model.total_train_time)
    print("Count of trainings performed to generate this model: ", model.train_count)
    print("Length of the word2vec vectors: ", model.vector_size)
    print("Applied context length for generating the model: ", model.window)


# %%
report_stats(cbow_model)

# %%
report_stats(skip_gram_model)

# %% [markdown]
# ### Operaciones con los vectores entrenados
#
# Veremos operaciones comunes sobre vectores. Estos resultados dependeran del modelo que hayamos cargado en memoria

# %%
models = {
    Algorithms.CBOW: cbow_model,
    Algorithms.SKIP_GRAM: skip_gram_model,
}

# %%
model = models[Algorithms.SKIP_GRAM]

# %%
for index, word in enumerate(model.wv.index_to_key):
    if index == 100:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

# %%
gato_vec = model.wv["gato"]
print(gato_vec[:10])
print(len(gato_vec))

# %%
try:
    agustisidad_vec = model.wv["agusticidad"]
except KeyError:
    print("OOV founded!")


# %%
agustisidad_vec[:10]
len(agustisidad_vec)

# %%
model.wv.most_similar("mercado", topn=5)

# %% [markdown]
# Podemos ver como la similitud entre palabras decrece

# %%
word_pairs = [
    ("automóvil", "camión"),
    ("automóvil", "bicicleta"),
    ("automóvil", "cereal"),
    ("automóvil", "conde"),
]

for w1, w2 in word_pairs:
    print(f"{w1} - {w2} {model.wv.similarity(w1, w2)}")

# %%
# rey es a hombre como ___ a mujer
# londres es a inglaterra como ____ a vino
model.wv.most_similar(positive=["saltillo", "morelos"], negative=["cuernavaca"])

# %%
model.wv.doesnt_match(["disco", "música", "mantequilla", "cantante"])

# %%
model.wv.similarity("noche", "noches")

# %% [markdown]
# #### Evaluación

# %% [markdown]
# `Word2Vec` es una tarea de entrenamiento semi-supervisada, por lo tanto, es difícil evaluar el desempeño de un modelo. La evaluación dependerá de la tarea.
#
# Sin embargo, Google liberó un conjunto de evaluación con ejemplos semánticos/sintácticos. Se sigue la forma "A es a B como C es a D". Por ejemplo, "tokio es a japon como berlin es a alemania".
#
# Se tienen varias categorias como comparaciones sintácticas, capitales, miembros de una familia, etc.

# %%
from gensim.test.utils import datapath

model.wv.evaluate_word_analogies(datapath("questions-words.txt"))


# %% [markdown]
# ## Modelos del Lenguaje Neuronales (Bengio)

# %% [markdown]
# - [(Bengio et al 2003)](https://dl.acm.org/doi/10.5555/944919.944966) proponen una arquitecura neuronal como alternativa a los modelos del lenguaje estadísticos
# - Esta arquitectura lidia mejor con los casos donde las probabilidades se hacen cero, sin necesidad de aplicar una técnica de smoothing.

# %% [markdown]
# <p float="left">
#   <img src="https://toppng.com/public/uploads/preview/at-the-movies-will-smith-meme-tada-11562851401lnexjqtwf9.png" width="100" />
#   <img src="https://abhinavcreed13.github.io/assets/images/bengio-model.png" width="600"/>
# </p>

# %%
def lm_preprocess_corpus(corpus: list[str]) -> list[str]:
    """Función de preprocesamiento para LM

    Esta función está diseñada para preprocesar
    corpus para modelos del lenguaje neuronales.
    Agrega tokens de inicio y fin, normaliza
    palabras a minusculas
    """
    preprocessed_corpus = []
    for sent in corpus:
        result = [word.lower() for word in sent]
        # Al final de la oración
        result.append("<EOS>")
        result.insert(0, "<BOS>")
        preprocessed_corpus.append(result)
    return preprocessed_corpus


# %%
def get_words_freqs(corpus: list[list[str]]):
    """Calcula la frecuencia de las palabras en un corpus"""
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs


# %%
UNK_LABEL = "<UNK>"


def get_words_indexes(words_freqs: dict) -> dict:
    """Calcula los indices de las palabras dadas sus frecuencias"""
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        # Happax legomena happends
        if words_freqs[word] == 1:
            # Temp index for unknowns
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx

    return {word: idx for idx, word in enumerate(result.keys())}, {
        idx: word for idx, word in enumerate(result.keys())
    }


# %%
import nltk

nltk.download("gutenberg")
nltk.download("abc")
nltk.download("genesis")
nltk.download("inaugural")
nltk.download("state_union")
nltk.download("webtext")
nltk.download("punkt_tab")

# %%
from nltk.corpus import abc, genesis, gutenberg, inaugural, state_union, webtext

# Exploración del corpus
total_sents = 0
corpora = []

plaintext_corpora = {
    "abc": abc,
    "Gutenberg": gutenberg,
    "Genesis": genesis,
    "Inaugural": inaugural,
    "State Union": state_union,
    "Web": webtext,
}

for title, corpus in plaintext_corpora.items():
    corpus_sents = lm_preprocess_corpus(corpus.sents())
    corpus_len = len(corpus_sents)
    total_sents += corpus_len
    print(f"{title} sents={corpus_len}")
    corpora.extend(corpus_sents)
print(f"Total={total_sents}")

# %%
len(corpora)

# %%
corpora[42]

# %%
words_freqs = get_words_freqs(corpora)

# %%
words_freqs["the"]

# %%
len(words_freqs)

# %%
count = 0
for word, freq in words_freqs.items():
    if freq == 1 and count <= 10:
        print(word, freq)
        count += 1

# %%
words_indexes, index_to_word = get_words_indexes(words_freqs)

# %%
words_indexes["god"]

# %%
index_to_word[9573]

# %%
len(words_indexes)

# %%
len(index_to_word)


# %%
def get_word_id(words_indexes: dict, word: str) -> int:
    """Obtiene el id de una palabra dada

    Si no se encuentra la palabra se regresa el id
    del token UNK
    """
    unk_word_id = words_indexes[UNK_LABEL]
    return words_indexes.get(word, unk_word_id)


# %% [markdown]
# ### Obtenemos trigramas

# %% [markdown]
# Convertiremos los trigramas obtenidos a secuencias de idx, y preparamos el conjunto de entrenamiento $x$ y $y$
#
# - x: Contexto
# - y: Predicción de la siguiente palabra

# %%
from nltk import ngrams


def get_train_test_data(
    corpus: list[list[str]], words_indexes: dict, n: int
) -> tuple[list, list]:
    """Obtiene el conjunto de train y test

    Requerido en el step de entrenamiento del modelo neuronal
    """
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append(
                [get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)]
            )
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train


# %% [markdown]
# ### Preparando Pytorch
#
# $x' = e(x_1) \oplus e(x_2)$
#
# $h = \tanh(W_1 x' + b)$
#
# $y = softmax(W_2 h)$

# %%
# cargamos bibliotecas
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# %%
# Setup de parametros
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
BATCH_SIZE = 256
H = 100
torch.manual_seed(42)
# Tamaño del Vocabulario
V = len(words_indexes)

# %%
x_train, y_train = get_train_test_data(corpora, words_indexes, n=3)

# %%
import numpy as np

train_set = np.concatenate((x_train, y_train), axis=1)
# partimos los datos de entrada en batches
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)


# %% [markdown]
# ### Creamos la arquitectura del modelo

# %%
# Trigram Neural Network Model
class TrigramModel(nn.Module):
    """Clase padre: https://pytorch.org/docs/stable/generated/torch.nn.Module.html"""

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        # x': concatenation of x1 and x2 embeddings   -->
        # self.embeddings regresa un vector por cada uno de los índices que se les pase como entrada.
        # view() les cambia el tamaño para concatenarlos
        embeds = self.embeddings(inputs).view(
            (-1, self.context_size * self.embedding_dim)
        )
        # h: tanh(W_1.x' + b)  -->
        out = torch.tanh(self.linear1(embeds))
        # W_2.h                 -->
        out = self.linear2(out)
        # log_softmax(W_2.h)      -->
        # dim=1 para que opere sobre renglones, pues al usar batchs tenemos varios vectores de salida
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


# %%
# Seleccionar la GPU si está disponible
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# %%
NN_MODELS_PATH = os.path.join(MODELS_PATH, "nn")

os.makedirs(NN_MODELS_PATH, exist_ok=True)

LM_PATH = os.path.join(NN_MODELS_PATH, "trigrams_nlm_cpu_epoch3.pt")

# %%
torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")

# 1. Pérdida. Negative log-likelihood loss
loss_function = nn.NLLLoss()

# 2. Instanciar el modelo y enviarlo a device
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# 3. Optimización. ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# ------------------------- TRAIN & SAVE MODEL ------------------------
EPOCHS = 1
for epoch in range(EPOCHS):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch))
    for it, data_tensor in enumerate(train_loader):
        # Mover los datos al dispositivo
        context_tensor = data_tensor[:, 0:2].to(device)
        target_tensor = data_tensor[:, 2].to(device)

        model.zero_grad()

        # FORWARD:
        log_probs = model(context_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # BACKWARD:
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(
                "Training Iteration {} of epoch {} complete. Loss: {}; Time taken (s): {}".format(
                    it, epoch, loss.item(), (time.time() - st)
                )
            )
            st = time.time()

    # saving model
    model_path = os.path.join(
        NN_MODELS_PATH, f"lm_large_{device}_context_{CONTEXT_SIZE}_epoch_{epoch}.dat"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for epoch={epoch} at {model_path}")


# %%
model


# %%
def get_model(path: str) -> TrigramModel:
    """Obtiene modelo de pytorch desde disco"""
    model_loaded = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H)
    model_loaded.load_state_dict(torch.load(path))
    model_loaded.eval()
    return model_loaded


# %%
# model = get_model(PATH)
W1 = "<BOS>"
W2 = "my"

IDX1 = get_word_id(words_indexes, W1)
IDX2 = get_word_id(words_indexes, W2)

# Obtenemos Log probabidades p(W3|W2,W1)
probs = model(torch.tensor([[IDX1, IDX2]]).to(device)).detach().tolist()

# %%
len(probs[0])

# %%
# Creamos diccionario con {idx: logprob}
model_probs = {}
for idx, p in enumerate(probs[0]):
    model_probs[idx] = p

# Sort:
model_probs_sorted = sorted(
    ((prob, idx) for idx, prob in model_probs.items()), reverse=True
)

# Printing word  and prob (retrieving the idx):
topcandidates = 0
for prob, idx in model_probs_sorted:
    # Retrieve the word associated with that idx
    word = index_to_word[idx]
    print(idx, word, prob, np.exp(prob))

    topcandidates += 1

    if topcandidates > 10:
        break

# %%
print(index_to_word.get(model_probs_sorted[0][1]))


# %% [markdown]
# ### Generacion de lenguaje

# %%
def get_likely_words(
    model: TrigramModel,
    context: str,
    words_indexes: dict,
    index_to_word: dict,
) -> list[tuple]:
    model_probs = {}
    words = context.split()
    idx_word_1 = get_word_id(words_indexes, words[0])
    idx_word_2 = get_word_id(words_indexes, words[1])
    probs = model(torch.tensor([[idx_word_1, idx_word_2]]).to(device)).detach().tolist()

    for idx, p in enumerate(probs[0]):
        model_probs[idx] = p

    # Strategy: Sort and get top-K words to generate text
    return sorted(
        ((prob, index_to_word[idx]) for idx, prob in model_probs.items()), reverse=True
    )


# %%
sentence = "this is"
get_likely_words(model, sentence, words_indexes, index_to_word)[:3]

# %%
import random
from random import randint

def get_next_top_p_word(words: list[tuple[float, str]], p: float = 0.8) -> str:
    """
    Selecciona la siguiente palabra utilizando Nucleus Sampling (Top-p).
    
    Params:
    - words: Lista de tuplas (palabra, probabilidad).
    - p: Umbral de masa de probabilidad acumulada (típicamente entre 0.8 y 0.95).
    """
    if not words:
        return "<EOS>"
        
    # Aseguramos que la lista esté ordenada de mayor a menor probabilidad
    # sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
    
    valid_words = []
    valid_probs = []
    cumulative_prob = 0.0
    
    # Recolectamos palabras hasta que la suma de probabilidades alcance el umbral 'p'
    for log_prob, word in words:
        prob = np.exp(log_prob)
        valid_words.append(word)
        valid_probs.append(prob)
        cumulative_prob += prob
        
        if cumulative_prob >= p:
            break
            
    # Muestreamos una palabra del subconjunto válido (núcleo) usando sus probabilidades como pesos.
    # random.choices devuelve una lista, por lo que extraemos el elemento [0]
    return random.choices(valid_words, weights=valid_probs, k=1)[0]


def get_next_word(words: list[tuple[float, str]]) -> str:
    # From a top-K list of words get a random word
    return words[randint(0, len(words) - 1)][1]


# %%
get_next_top_p_word(get_likely_words(model, sentence, words_indexes, index_to_word))

# %%
MAX_TOKENS = 50
TOP_P = 0.7


def generate_text(
    model: TrigramModel,
    history: str,
    words_indexes: dict,
    index_to_word: dict,
    tokens_count: int = 0,
) -> None:
    next_word = get_next_top_p_word(
        get_likely_words(
            model, history, words_indexes, index_to_word
        ), p=TOP_P
    )
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(
        model,
        history.split()[1] + " " + next_word,
        words_indexes,
        index_to_word,
        tokens_count,
    )


# %%
sentence = "god said"
print(sentence, end=" ")
generate_text(model, sentence, words_indexes, index_to_word)

# %% [markdown]
# # Práctica 4: Evaluación de modelos del lenguaje
#
# **Fecha: 5 de Mayo 2026 11:59pm**
#
# La calidad de un modelo del lenguaje puede ser evaluado por medio de su perplejidad (perplexity)
#
# - Investigar como calcular la perplejidad de un modelo del lenguaje y como evaluarlo con esa medida
#     - Incluir en el `README.md` de su entrega una síntesis de esta investigación (Un par de parrafos)
# - Evalua el modelo entrenado en clase con los corpus de `nltk`
# - Entrena un nuevo modelo del lenguaje neuronal con los corpus de `nltk` aplicando previamente sub-word tokenization al corpus 
#     - Puedes utilizar un modelo de tokenizacion pre-entrenado o entrenar uno desde cero
#     - TODO: Test de evauación
# - Evalua tu modelo calculando su perplejidad
#     - Compara los resultados de la evaluación de los ambos modelos.
#     - ¿Cúal es mejor? ¿Por qué?
#
# ## EXTRA
#
# - Diseña una estrategia de generación de usando el modelo del lenguaje entrenado con sub-word tokenization
# - Se deben generar secuencias de palabras (no subwords)

# %% [markdown]
#
