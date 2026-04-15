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
# # 7. Modelos de Sub-word Tokenization y Modelos del Lenguaje Neuronales

# %% [markdown]
# ## Objetivos
#
# - Entrenar modelos para sub-word tokenization
#   - Aplicar BPE a corpus

# %%
import os
import re
import nltk
from rich import print as rprint

# %% [markdown]
# ### Función de preprocesamiento

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

def preprocess_words(words: list[str], regex: str=r"[^\w+]", lang: str="en", remove_stops: bool = False, remove_accents: bool = False) -> list[str]:
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


def preprocess_text(text: str, to_lower: bool = True) -> str:
    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)
    
    if to_lower:
        text = text.lower()
    
    # 3. Collapse all whitespace/newlines into a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Clean up leading/trailing whitespace
    text = text.strip()
    
    return text


# %% [markdown]
# ## Vamos a tokenizar 🌈
# ![](https://i.pinimg.com/736x/58/6b/88/586b8825f010ce0e3f9c831f568aafa8.jpg)

# %%
BASE_PATH = "."
CORPORA_PATH = f"{BASE_PATH}/data/tokenization"
MODELS_PATH = f"{BASE_PATH}/models/sub-word"


os.makedirs(CORPORA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# %% [markdown]
# #### Corpus en español: Wikipedia

# %%
from datasets import load_dataset, load_dataset_builder

# %%
data_builder = load_dataset_builder("wikimedia/wikipedia", "20231101.es")

# %%
rprint(data_builder.info)

# %%
dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

# %%
wiki_words = []
for article in dataset.take(1):
    print(preprocess_text(article["text"][:1000]))

# %%
# %%time

wiki_file_path = f"{CORPORA_PATH}/wikipedia_es_plain.txt"
with open(wiki_file_path, "w", encoding="utf-8") as f:
    for article in dataset.take(1000):
        f.write(preprocess_text(article["text"]))

# %% [markdown]
# #### Corpus Inglés: Gutenberg

# %%
nltk.download("punkt_tab")

# %%
from nltk.corpus import gutenberg

gutenberg_words = gutenberg.words()[:200000]

# %%
rprint(" ".join(gutenberg_words[:30]))

# %%
gutenberg_plain_text = " ".join(preprocess(gutenberg_words))

rprint(gutenberg_plain_text[:100])

# %%
gutenberg_preprocessed_words = gutenberg_plain_text.split()

# %%
with open(f"{CORPORA_PATH}/gutenberg_plain.txt", "w") as f:
    f.write(gutenberg_plain_text)

# %% [markdown]
# #### Ejercicio: Aplica un tokenizando para el español con Hugging face

# %%
from transformers import AutoTokenizer

spanish_tokenizer = AutoTokenizer.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-uncased"
)

# %%
rprint(spanish_tokenizer.tokenize(cess_plain_text[1000:1400]))

# %%
cess_types = Counter(cess_words)

# %%
rprint(cess_types.most_common(10))

# %%
cess_tokenized = spanish_tokenizer.tokenize(cess_plain_text)
rprint(cess_tokenized[:10])
cess_tokenized_types = Counter(cess_tokenized)

# %%
rprint(cess_tokenized_types.most_common(30))

# %%
cess_lemmatized_types = Counter(lemmatize(cess_words, lang="es"))

# %%
rprint(cess_lemmatized_types.most_common(30))

# %%
rprint("CESS")
rprint(f"Tipos ([bright_magenta]word-base[/]): {len(cess_types)}")
rprint(f"Tipos ([bright_yellow]lemmatized[/]): {len(cess_lemmatized_types)}")
rprint(f"Tipos ([bright_green]sub-word[/]): {len(cess_tokenized_types)}")

# %% [markdown]
# #### Tokenizando para el inglés

# %%
gutenberg_types = Counter(gutenberg_words)

# %%
gutenberg_tokenized = wp_tokenizer.tokenize(gutenberg_plain_text)
gutenberg_tokenized_types = Counter(gutenberg_tokenized)

# %%
rprint(gutenberg_tokenized_types.most_common(10))

# %%
gutenberg_lemmatized_types = Counter(lemmatize(gutenberg_preprocessed_words))

# %%
rprint(gutenberg_lemmatized_types.most_common(20))

# %%
rprint("Gutenberg")
rprint(f"Tipos ([bright_magenta]word-base[/]): {len(gutenberg_types)}")
rprint(f"Tipos ([bright_yellow]lemmatized[/]): {len(gutenberg_lemmatized_types)}")
rprint(f"Tipos ([bright_green]sub-word[/]): {len(gutenberg_tokenized_types)}")

# %% [markdown]
# #### OOV: out of vocabulary

# %% [markdown]
# Palabras que se vieron en el entrenamiento pero no estan en el test

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    gutenberg_words, test_size=0.3, random_state=42
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
    gutenberg_tokenized, test_size=0.3, random_state=42
)
rprint(len(train_tokenized), len(test_tokenized))

# %%
oov_tokenized_test = set(test_tokenized) - set(train_tokenized)

# %%
rprint("OOV ([yellow]word-base):", len(oov_test))
rprint("OOV ([green]sub-word):", len(oov_tokenized_test))

# %% [markdown]
# ## Entrenando nuestro modelo con BPE
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fd565618bb1217a7c435579d9172270d0%2Ftenor.gif%3Fitemid%3D3379322&f=1&nofb=1&ipt=9719714edb643995ce9d978c8bab77f5310204960093070e37e183d5372096d9&ipo=images)

# %%
# !pip install subword-nmt

# %%
# !ls data/tokenization

# %%
# !head -c 1000 data/tokenization/wikipedia_es_plain.txt

# %%
# %%time

# !subword-nmt learn-bpe -s 300 < \
#  data/tokenization/wikipedia_es_plain.txt > \
#   models/sub-word/wiki_es_small.model

# %%
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c models/sub-word/wiki_es_small.model

# %%
# %%time

# !subword-nmt learn-bpe -s 1500 < \
# data/tokenization/wikipedia_es_plain.txt > \
#  models/sub-word/wiki_es_high.model

# %%
# !echo "ando haciendo un análisis para claramente ver si puedes procesar esta oración mano" \
# | subword-nmt apply-bpe -c models/sub-word/wiki_es_high.model

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.tokenize("ando haciendo un análisis para claramente ver si puedes procesar esta oración mano"))

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
# #### Biblia en Inglés

# %%
eng_bible_plain_text = get_bible_corpus("eng")
eng_bible_words = eng_bible_plain_text.lower().replace("\n", " ").split()

# %%
print(eng_bible_words[:10])

# %%
len(eng_bible_words)

# %%
eng_bible_types = Counter(eng_bible_words)

# %%
rprint(eng_bible_types.most_common(30))

# %%
eng_bible_lemmas_types = Counter(lemmatize(eng_bible_words, lang="en"))

# %%
write_plain_text_corpus(eng_bible_plain_text, f"{CORPORA_PATH}/eng-bible")

# %%
# !subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg_high.model < \
#  {CORPORA_PATH}/eng-bible.txt > \
#  {CORPORA_PATH}/eng-bible-tokenized.txt

# %%
with open(f"{CORPORA_PATH}/eng-bible-tokenized.txt", "r") as f:
    tokenized_data = f.read()
eng_bible_tokenized = tokenized_data.split()

# %%
rprint(eng_bible_tokenized[:10])

# %%
len(eng_bible_tokenized)

# %%
eng_bible_tokenized_types = Counter(eng_bible_tokenized)
len(eng_bible_tokenized_types)

# %%
eng_bible_tokenized_types.most_common(30)

# %% [markdown]
# #### ¿Qué pasa si aplicamos el modelo aprendido con Gutenberg a otras lenguas?

# %%
spa_bible_plain_text = get_bible_corpus("spa")
spa_bible_words = spa_bible_plain_text.replace("\n", " ").lower().split()

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
spa_bible_lemmas_types = Counter(lemmatize(spa_bible_words, lang="es"))
len(spa_bible_lemmas_types)

# %%
write_plain_text_corpus(spa_bible_plain_text, f"{CORPORA_PATH}/spa-bible")

# %%
# !subword-nmt apply-bpe -c {MODELS_PATH}/gutenberg_high.model < \
#  {CORPORA_PATH}/spa-bible.txt > \
#  {CORPORA_PATH}/spa-bible-tokenized.txt

# %%
with open(f"{CORPORA_PATH}/spa-bible-tokenized.txt", "r") as f:
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

# %% [markdown]
# ## Type-token Ratio (TTR)
#
# - Una forma de medir la variación del vocabulario en un corpus
# - Este se calcula como $TTR = \frac{len(types)}{len(tokens)}$
# - Puede ser útil para monitorear la variación lexica de un texto

# %%
rprint("Información de la biblia en Inglés")
rprint("Tokens:", len(eng_bible_words))
rprint("Types ([bright_magenta]word-base):", len(eng_bible_types))
rprint("Types ([bright_yellow]lemmatized)", len(eng_bible_lemmas_types))
rprint("Types ([bright_green]BPE):", len(eng_bible_tokenized_types))
rprint("TTR ([bright_magenta]word-base):", len(eng_bible_types) / len(eng_bible_words))
rprint("TTR ([bright_green]BPE):", len(eng_bible_tokenized_types) / len(eng_bible_tokenized))

# %%
rprint("Bible Spanish Information")
rprint("Tokens:", len(spa_bible_words))
rprint("Types ([bright_magenta]word-base):", len(spa_bible_types))
rprint("Types ([bright_yellow]lemmatized)", len(spa_bible_lemmas_types))
rprint("Types ([bright_green]BPE):", len(spa_bible_tokenized_types))
rprint("TTR ([bright_magenta]word-base):", len(spa_bible_types) / len(spa_bible_words))
rprint("TTR ([bright_green]BPE):", len(spa_bible_tokenized_types) / len(spa_bible_tokenized))


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
def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Función de preprocesamiento

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

    return {word: idx for idx, word in enumerate(result.keys())}, {idx: word for idx, word in enumerate(result.keys())}


# %%
corpus = preprocess_corpus(reuters.sents())

# %%
len(corpus)

# %%
words_freqs = get_words_freqs(corpus)

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
words_indexes["the"]

# %%
index_to_word[16]

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
def get_train_test_data(corpus: list[list[str]], words_indexes: dict, n: int) -> tuple[list, list]:
    """Obtiene el conjunto de train y test

    Requerido en el step de entrenamiento del modelo neuronal
    """
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append([get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)])
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
x_train, y_train = get_train_test_data(corpus, words_indexes, n=3)

# %%
import numpy as np

train_set = np.concatenate((x_train, y_train), axis=1)
# partimos los datos de entrada en batches
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE)


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
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
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
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# %%
#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")

# 1. Pérdida. Negative log-likelihood loss
loss_function = nn.NLLLoss()

# 2. Instanciar el modelo y enviarlo a device
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# 3. Optimización. ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr = 2e-3)

# ------------------------- TRAIN & SAVE MODEL ------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch))
    for it, data_tensor in enumerate(train_loader):
        # Mover los datos a la GPU
        context_tensor = data_tensor[:,0:2].to(device)
        target_tensor = data_tensor[:,2].to(device)

        model.zero_grad()

        # FORWARD:
        log_probs = model(context_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # BACKWARD:
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print("Training Iteration {} of epoch {} complete. Loss: {}; Time taken (s): {}".format(it, epoch, loss.item(), (time.time()-st)))
            st = time.time()

    # saving model
    model_path = f'drive/MyDrive/LM_neuronal/model_{device}_context_{CONTEXT_SIZE}_epoch_{epoch}.dat'
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
PATH = "drive/MyDrive/LM_neuronal/model_cuda_4.dat"

# %%
#model = get_model(PATH)
W1 = "<BOS>"
W2 = "my"

IDX1 = get_word_id(words_indexes, W1)
IDX2 = get_word_id(words_indexes, W2)

#Obtenemos Log probabidades p(W3|W2,W1)
probs = model(torch.tensor([[IDX1,  IDX2]]).to(device)).detach().tolist()

# %%
len(probs[0])

# %%
# Creamos diccionario con {idx: logprob}
model_probs = {}
for idx, p in enumerate(probs[0]):
  model_probs[idx] = p

# Sort:
model_probs_sorted = sorted(((prob, idx) for idx, prob in model_probs.items()), reverse=True)

# Printing word  and prob (retrieving the idx):
topcandidates = 0
for prob, idx in model_probs_sorted:
  #Retrieve the word associated with that idx
  word = index_to_word[idx]
  print(idx, word, prob)

  topcandidates += 1

  if topcandidates > 10:
    break

# %%
print(index_to_word.get(model_probs_sorted[0][1]))


# %% [markdown]
# ### Generacion de lenguaje

# %%
def get_likely_words(model: TrigramModel, context: str, words_indexes: dict, index_to_word: dict, top_count: int=10) -> list[tuple]:
    model_probs = {}
    words = context.split()
    idx_word_1 = get_word_id(words_indexes, words[0])
    idx_word_2 = get_word_id(words_indexes, words[1])
    probs = model(torch.tensor([[idx_word_1, idx_word_2]]).to(device)).detach().tolist()

    for idx, p in enumerate(probs[0]):
        model_probs[idx] = p

    # Strategy: Sort and get top-K words to generate text
    return sorted(((prob, index_to_word[idx]) for idx, prob in model_probs.items()), reverse=True)[:top_count]


# %%
sentence = "this is"
get_likely_words(model, sentence, words_indexes, index_to_word, 3)

# %%
from random import randint

def get_next_word(words: list[tuple[float, str]]) -> str:
    # From a top-K list of words get a random word
    return words[randint(0, len(words)-1)][1]


# %%
get_next_word(get_likely_words(model, sentence, words_indexes, index_to_word))

# %%
MAX_TOKENS = 50
TOP_COUNT = 10
def generate_text(model: TrigramModel, history: str, words_indexes: dict, index_to_word: dict, tokens_count: int=0) -> None:
    next_word = get_next_word(get_likely_words(model, history, words_indexes, index_to_word, top_count=TOP_COUNT))
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(model, history.split()[1]+ " " + next_word, words_indexes, index_to_word, tokens_count)


# %%
sentence = "mexico is"
print(sentence, end=" ")
generate_text(model, sentence, words_indexes, index_to_word)

# %% [markdown]
#
