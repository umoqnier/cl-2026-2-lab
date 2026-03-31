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
# # 5. Modelos del lenguaje (estadísticos)

# %% [markdown]
# ![](https://lena-voita.github.io/resources/lectures/lang_models/examples/morphosyntax-min.png)
#
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %% [markdown]
# ## Objetivo

# %% [markdown]
# - Implementar modelos del lenguaje estadísticos
# - Aplicaciones

# %% [markdown]
# ## Modelos del lenguaje

# %% [markdown]
# > "Un modelo del lenguaje es un modelo estadístico que asigna probabilidades a cadenas dentro de un lenguaje" - Parafraseando a mi compadre [Jurafsky, 2026](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
#
# $$ \mu = (\Sigma, A, \Pi)$$
#
# Donde:
# - $\mu$ es el modelo del lenguaje
# - $\Sigma$ es el vocabulario
# - $A$ es el tensor que guarda las probabilidades
# - $\Pi$ guarda las probabilidades iniciales

# %% [markdown]
# - Este modelo busca estimar la probabilidad de una secuencia de tokens
# - Pueden ser palabras, caracteres o tokens
# - Se pueden considerar varios escenarios para la creación de estos modelos
#     - Si podemos estimar la probabilidad de una unidad lingüística (palabras, tokens, oraciones, etc), podemos usarlar de formas insospechadas

# %% [markdown]
# ### Probabilidad de una oración

# %% [markdown]
# - El objetivo es estimar las probabilidades de unidades lingüísticas que reflejen el comportamiento del lenguaje natural
# - Esto es, por ejemplo, las oraciones que tengan mayor probabilidad de ocurrir

# %% [markdown]
# ####  Muchas oraciones probablemente no ocurran en nuestro corpus. ¿Cómo lidiamos con eso❓

# %% [markdown]
# #### I saw a cat in a mat

# %% [markdown]
# <img src="https://lena-voita.github.io/resources/lectures/lang_models/general/i_saw_a_cat_prob.gif">
#
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %% [markdown]
# Sean $y_1, y_2, \dots, y_n$ tokens y $P(y_1, y_2, \dots, y_n)$ la probabilidad de verlos en ese orden. Si aplicamos la regla de la cadena obtenemos:
#
# $$
# P(y_1, y_2, \dots, y_n)=P(y_1)\cdot P(y_2|y_1) \cdot P(y_3|y_1, y_2)\cdot\dots\cdot P(y_n|y_1, \dots, y_{n-1})=
#         \prod \limits_{t=1}^n P(y_t|y_{<t}).
# $$

# %% [markdown]
# Con esto modelamos la probabilidad de que un conjunto de tokens ocurran como una **probabilidad condicional** $P(y_n|y_1, \dots, y_{n-1})$. Podemos estimar esta probabilidad de multiples formas:
#
# - N-gramas
# - Modelos neuronales

# %% [markdown]
# ### Sobre bigramas y N-gramas

# %% [markdown]
# - Para bigramas tenemos la propiedad de Markov
# - Para $n > 2$ las palabras dependen de mas elementos
#     - Trigramas
#     - 4-gramas
# - En general para un modelo de n-gramas se toman en cuenta $n-1$ elementos

# %% [markdown]
# ![](https://lena-voita.github.io/resources/lectures/lang_models/ngram/example_cut_3gram-min.png)
#
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %% [markdown]
# ## Programando nuestros modelos del lenguaje

# %% [markdown]
# Utilizaremos un [corpus](https://www.nltk.org/book/ch02.html) en inglés disponible en NLTK

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
import numpy as np

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
    corpus_sents = corpus.sents()
    corpus_len = len(corpus_sents)
    total_sents += corpus_len
    print(f"{title} sents={corpus_len}")
    corpora.append(corpus_sents)
print(f"Total={total_sents}")


# %%
def preprocess_sent(sent: list[str]) -> list[str]:
    """Función de preprocesamiento

    Agrega tokens de inicio y fin a la oración y normaliza todo a minusculas

    Params
    ------
    sent: list[str]
        Lista de palabras que componen la oración

    Return
    ------
    Las oración preprocesada
    """
    result = [word.lower() for word in sent]
    # Al final de la oración
    result.append("</s>")
    # Al inicio de la oración
    result.insert(0, "<s>")
    return result


# %%
print(preprocess_sent(corpora[0][0]))

# %%
for i, sent in enumerate(corpora[0][:10]):
    print(i, " ".join(sent))

# %%
for i, sent in enumerate(corpora[0][:10]):
    print(i, " ".join(preprocess_sent(sent)))

# %%
from nltk import ngrams

for ngram in list(ngrams(preprocess_sent(corpora[0][10]), 2))[:10]:
    print(ngram)

# %% [markdown]
# ### Construyendo el modelo del lenguaje

# %% [markdown]
# ![](https://imgs.xkcd.com/comics/predictive_models_2x.png)

# %%
from collections import Counter, defaultdict  # ❤️‍🔥❤️‍🔥❤️‍🔥

test = defaultdict(Counter)

# %%
test[("el", "gato")]["salta"] += 1

# %%
test

# %%
type NgramModel = defaultdict[tuple[str, str], Counter[str]]


# %%
def build_trigram_model(data: list[list[list[str]]]) -> NgramModel:
    # Initialize model as a nested dict with default behavior
    model: NgramModel = defaultdict(Counter)
    for corpus in data:
        for sentence in corpus:
            for w1, w2, w3 in ngrams(preprocess_sent(sentence), 3):
                model[(w1, w2)][w3] += 1
    return model


# %%
# %%time
trigram_model = build_trigram_model(corpora)

# %%
trigram_model["<s>", "in"]

# %% [markdown]
# ### Calculo de probabilidades con conteos de trigramas

# %%
for i, key in enumerate(trigram_model):
    print(key)
    if i == 5:
        break


# %%
def calculate_model_probs(model: NgramModel) -> NgramModel:
    model_probs: NgramModel = defaultdict(Counter)
    # Por cada prefijo del modelo
    for prefix in model:
        # Todas las veces que ocurre prefix seguido de cualquier palabra
        total = float(sum(model[prefix].values()))
        # Por cada palabra w que haya ocurrido con prefix
        for w in model[prefix]:
            # Obtenemos la probabilidad
            model_probs[prefix][w] = model[prefix][w] / total
    return model_probs


# %% [markdown]
# ## Smoothing 🥤

# %% [markdown]
# ![](https://lena-voita.github.io/resources/lectures/lang_models/ngram/prob_cat_on_a_mat-min.png)
#
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %% [markdown]
# ¿Qué pasaría se no sucede la secuencia de tokens del númerador/denominador? Para evitar este problema se utiliza una técnica llamada *smoothing* que redistribuye la función de masa de probabilidad.

# %% [markdown]
# #### ¿Cómo agregamos smoothing❓

# %% [markdown]
# La forma más sencilla es pretender que vimos al menos una vez todos los n-gramas. Esto es sumar 1 a todas las cuentas. Algo más general sería agregar una cantidad $\delta$:
#
# $$
# P(mat| cat\ on\ a) = \frac{\delta + N(cat\ on\ a\ mat)}{\delta \cdot |V| + N(cat\ on\ a)}
# $$

# %% [markdown]
# #### Ejercicio 🤹🏽: Implementa el smoothing de Laplace agregando $\delta$ a todas las cuentas de los n-gramas
#
# - *Hint*: Obten todos los tokens, despues el vocabulario y el tamaño del vocabulario

# %%
# Calcula el vocabulario (tipos) acá
TOKENS = []
for _, corpus in plaintext_corpora.items():
    for sent in corpus.sents():
        for word in sent:
            TOKENS.append(word.lower())
VOCABULARY = set(TOKENS)
# +2 por los tokens <s> y </s>
VOCABULARY_SIZE = len(VOCABULARY) + 2


# %%
VOCABULARY_SIZE

# %%
len(trigram_model)


# %%
def calculate_smooth_probs(model: NgramModel, vocab_size: int, delta: float = 1.0) -> NgramModel:
    model_probs = defaultdict(Counter)
    for prefix in model:
        total = float(sum(model[prefix].values()))
        for w in model[prefix]:
            model_probs[prefix][w] = (model[prefix][w] + delta) / (
                total + delta * vocab_size
            )
    return model_probs



# %%
trigram_probs = calculate_model_probs(trigram_model)

# %%
trigram_smooth = calculate_smooth_probs(trigram_model, VOCABULARY_SIZE)

# %%
sorted(dict(trigram_probs["<s>", "the"]).items(), key=lambda x: -1 * x[1])

# %%
sorted(dict(trigram_smooth["<s>", "the"]).items(), key=lambda x: -1 * x[1])

# %%
trigram_smooth["<s>", "the"]


# %% [markdown]
# ### Aplicaciones

# %% [markdown]
# - Generación de texto
# - Completado de texto
# - Speech To Text (STT)

# %% [markdown]
# ![](https://lena-voita.github.io/resources/lectures/lang_models/examples/suggest-min.png)
#
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %% [markdown]
# ### Generación de texto 📨

# %% [markdown]
# <video src="https://lena-voita.github.io/resources/lectures/lang_models/general/generation_example.mp4" controls loop>

# %% [markdown]
# > Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# %%
def get_likely_words(
    model: NgramModel, context: str, top_count: int = 10
) -> list[tuple]:
    """Dado un contexto obtiene las palabras más probables

    Params
    ------
    model: NgramModel
        Modelo con sus probabilidades calculadas
    context: str
        Contexto con el cual calcular las palabras más probables siguientes
    top_count: int
        Cantidad de palabras más probables. Default 10
    """
    history = tuple(context.split())
    return model[history].most_common(top_count)


# %%
get_likely_words(trigram_probs, "<s> the", top_count=10)

# %%
import random

# Strategy here
def get_next_word(words: list[tuple]) -> str:
    return words[0][0]

def get_next_random_word(words: list) -> str:
    if not words:
        return "</s>"
    return random.choice(words)[0]

def get_next_top_p_word(words: list[tuple[str, float]], p: float = 0.8) -> str:
    """
    Selecciona la siguiente palabra utilizando Nucleus Sampling (Top-p).
    
    Params:
    - words: Lista de tuplas (palabra, probabilidad).
    - p: Umbral de masa de probabilidad acumulada (típicamente entre 0.8 y 0.95).
    """
    if not words:
        return "</s>"
        
    # Aseguramos que la lista esté ordenada de mayor a menor probabilidad
    # sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
    
    valid_words = []
    valid_probs = []
    cumulative_prob = 0.0
    
    # Recolectamos palabras hasta que la suma de probabilidades alcance el umbral 'p'
    for word, prob in words:
        valid_words.append(word)
        valid_probs.append(prob)
        cumulative_prob += prob
        
        if cumulative_prob >= p:
            break
            
    # Muestreamos una palabra del subconjunto válido (núcleo) usando sus probabilidades como pesos.
    # random.choices devuelve una lista, por lo que extraemos el elemento [0]
    return random.choices(valid_words, weights=valid_probs, k=1)[0]



# %%
get_next_word(get_likely_words(trigram_probs, "emma was", 50))

# %%
sentence = "<s> the"
for i in range(10):
    print(i, get_next_random_word(get_likely_words(trigram_probs, sentence, 50)))

# %% [markdown]
# #### Ejercicio 🤺: Genera lenguaje
#
#
# - Utilizando el modelo de trigramas, diseña una estrategia para generación del lenguaje
# - Implementa la fución `generate_text()` que reciba un modelo de n-gramas y una historia y genere texto utilizando la estrategia implementada.
# - Agrega los parámetros que consideres necesarios a tu función de generación de texto
#
# **Ejemplo:**
#
# ```python
# sentence = "god was"
# generate_text(trigram_probs, sentence)
# >> god was evil 🐲
# ```

# %%
import time
from random import uniform
from typing import Callable

def generate_text(
    model: NgramModel,
    history: str,
    strategy: Callable,
    tokens_count: int = 0,
    top_n: int = 10,
    max_tokens: int = 50,
    use_gpu: bool = False,
):
    next_word = strategy(get_likely_words(model, history, top_count=top_n))

    if not use_gpu:
        time.sleep(uniform(0.1, 0.3))
    
    print(next_word, end=" ")
    
    tokens_count += 1
    
    if tokens_count == max_tokens or next_word == "</s>":
        return

    new_history = history.split()[-1] + " " + next_word
    
    return generate_text(
        model,
        new_history,
        strategy,
        tokens_count,
        top_n,
        max_tokens,
        use_gpu,
    )



# %%
sentence = "science is"
print(sentence, end=" ")
generate_text(trigram_probs, sentence, get_next_top_p_word, 0, max_tokens=10, top_n=15, use_gpu=False)


# %% [markdown]
# ### Calculando la probabilidad de una oración

# %%
def calculate_sentence_prob(sentence: list[str], model: NgramModel) -> float:
    prob = 0
    for w1, w2, w3 in ngrams(sentence, n=3):
        try:
            prob += np.log(model[w1, w2][w3])
        except KeyError:
            # OOV
            prob += 0.0
    return prob


# %%
nltk.download("reuters")

# %%
from nltk.corpus import reuters

# %%
news_sentence = preprocess_sent(reuters.sents()[6701])
gutenberg_sentence = preprocess_sent(gutenberg.sents()[100])
sentences = [news_sentence, gutenberg_sentence, preprocess_sent(gutenberg.sents()[101])]

for sent in sentences:
    print(f"PROB={calculate_sentence_prob(sent, trigram_smooth)}: '{' '.join(sent)}'")

# %%
i = 0

for j, sent in enumerate(reuters.sents()):
    sent = preprocess_sent(sent)
    if calculate_sentence_prob(sent, trigram_smooth) != -np.inf:
        print(
            f"{j} PROB={calculate_sentence_prob(sent, trigram_smooth)}: '{' '.join(sent)}'"
        )
        i += 1
    if i > 30:
        break

# %% [markdown]
# ### Usando un dataset grandesito

# %%
from datasets import load_dataset

dataset = load_dataset("Helsinki-NLP/opus-100", "en-es", split="train")

# %%
from nltk import word_tokenize


def build_vocabulary(dataset, vocab_size=30000):
    """
    Construye los diccionarios de mapeo para limitar el consumo de memoria.
    """
    word_counts = Counter()
    
    # Contamos frecuencias (solo procesamos el texto en español)
    for item in dataset:
        # Tokenización básica por espacios (puedes usar nltk.word_tokenize si lo prefieres)
        tokens = word_tokenize(item['translation']['es'].lower(), language="spanish")
        word_counts.update(tokens)
        
    # Nos quedamos solo con las 'vocab_size' palabras más frecuentes
    most_common = word_counts.most_common(vocab_size)
    
    # Inicializamos con nuestros tokens especiales
    word2idx = {"<UNK>": 0, "<s>": 1, "</s>": 2}
    idx2word = {0: "<UNK>", 1: "<s>", 2: "</s>"}
    
    # Asignamos un ID único a cada palabra del vocabulario
    for idx, (word, _) in enumerate(most_common, start=3):
        word2idx[word] = idx
        idx2word[idx] = word
        
    return word2idx, idx2word



# %%
# Construimos el vocabulario (Esto puede tomar un minuto)
word2idx, idx2word = build_vocabulary(dataset, vocab_size=1_000_000)

# %%
VOCAB_INT_SIZE = len(word2idx)
print(f"Vocabulario creado. Tamaño: {len(word2idx)}")


# %%
def encode_sentence(tokens: list[str], word2idx: dict) -> list[int]:
    """Convierte una lista de strings a enteros usando el diccionario."""
    return [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]


# %%
def build_integer_trigram_model(dataset, word2idx) -> NgramModel:
    """
    Construye el modelo de trigramas almacenando únicamente enteros.
    """
    # La estructura sigue siendo la misma, pero ahora almacena int -> int
    model: NgramModel = defaultdict(Counter)
    
    for item in dataset:
        raw_tokens = word_tokenize(item['translation']['es'].lower(), language="spanish")
        
        # Agregamos los IDs de los tokens <s> y </s>
        encoded_tokens = [word2idx["<s>"]] + encode_sentence(raw_tokens, word2idx) + [word2idx["</s>"]]
        
        # Construimos el modelo con las secuencias numéricas
        for w1, w2, w3 in ngrams(encoded_tokens, 3):
            model[(w1, w2)][w3] += 1
            
    return model



# %%
# Entrenamos el modelo "eficiente" en memoria (esto tomará ~2 minutos)
integer_trigram_model = build_integer_trigram_model(dataset, word2idx)

# %%
int_prob_trigram_model = calculate_model_probs(integer_trigram_model)


# %%
def generate_text_int(
    model: dict,
    context_idx: tuple[int, int],  # El historial ahora es una tupla de IDs numéricos
    idx2word: dict,                # Diccionario para decodificar
    strategy: Callable,
    tokens_count: int = 0,
    top_n: int = 10,
    max_tokens: int = 50,
    use_gpu: bool = False,
):
    predictions = model.get(context_idx, {})
    
    # Si el modelo llega a un callejón sin salida, nos detenemos
    if not predictions:
        return

    # Obtenemos los candidatos principales y normalizamos sus conteos a probabilidades
    top_candidates = predictions.most_common(top_n)
    
    # La estrategia elige el siguiente ID numérico
    next_word_idx = strategy(top_candidates)
    next_word_str = idx2word[next_word_idx]
    
    if not use_gpu:
        time.sleep(random.uniform(0.1, 0.3))
        
    print(next_word_str, end=" ")
    tokens_count += 1
    
    if tokens_count == max_tokens or next_word_str == "</s>":
        return
        
    # Actualizamos la ventana (desplazamos a la izquierda y añadimos el nuevo ID)
    new_context_idx = (context_idx[1], next_word_idx)
    
    return generate_text_int(
        model,
        new_context_idx,
        idx2word,
        strategy,
        tokens_count,
        top_n,
        max_tokens,
        use_gpu,
    )


# %%
text_prompt = "el gobierno"
# Procesamos la entrada del usuario
tokens = text_prompt.lower().split()
print(text_prompt, end=" ")
    
# Extraemos las dos últimas palabras del prompt y las codificamos
idxs = encode_sentence(tokens, word2idx)

w1 = idxs[-2]
w2 = idxs[-1]
context_idx = (w1, w2)

generate_text_int(
    model=int_prob_trigram_model,
    context_idx=context_idx,
    idx2word=idx2word,
    strategy=get_next_random_word,
    max_tokens=20
)

# %% [markdown]
#
# ## Referencias
#
# - [Maravilloso curso de Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html) ⚡
