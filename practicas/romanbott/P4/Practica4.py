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
# # Práctica 4: Evaluación de modelos del lenguaje neuronales

# %% [markdown]
# ## Investigación: Perplejidad en modelos del lenguaje
#
# ### ¿Qué es la perplejidad?
#
# La perplejidad es una medida de incertidumbre de una distribución de probabilidad discreta.
# Se puede usar como una métrica para evaluar el desempeño de un modelo de lenguaje.
# Intuitivamente representa el grado de incertidumbre o "sorpresa" de un modelo al predecir la siguiente palabra, dado un contexto.
#
# Algunos ejemplos que pueden dar más sentido es la perplejidad de un lanzamiento de una moneda, que es dos,
# o la de un lanzamiento de un dado, que es seis.
# Es decir, intuitivamente la perplejidad indica que la incertidumbre al elegir la siguiente palabra es comparable
# a la de tirar un dado de esa misma cantidad de caras.
#
# En general se considera que un modelo es mejor si su perplejidad es lo más baja posible.
#
#
# ### Fórmula matemática
# Para una sucesión de $N$ palabras $W = (w_1, w_2, ..., w_N)$, la perplejidad se define como
# el inverso de la media geométrica de las probabilidades predichas por el modelo:
#
# $$PPL(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \frac{1}{\sqrt[N]{P(w_1, w_2, ..., w_N)}}$$
#
# Utilizando la regla de la cadena y aplicando logaritmos (como se implementa usualmente para evitar errores de precisión numérica), se relaciona con la **entropía cruzada** ($H$):
#
# $$PPL(W) = e^{H(W)} = e^{-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i | w_{<i})}$$
#
# Donde:
# *   $P(w_1, ..., w_N)$: Probabilidad total asignada por el modelo a la secuencia completa.
# *   $N$: Número total de palabras (tokens) en el texto evaluado.
# *   $P(w_i | w_{<i})$: Probabilidad que el modelo asigna a la palabra actual dado el contexto anterior.
#
#
# ### Como métrica para modelos de lenguaje
# Existe una correlación inversa entre la perplejidad y la capacidad predictiva:
# *   **Perplejidad baja:** El modelo es preciso y asigna altas probabilidades a las secuencias de texto reales.
# *   **Perplejidad alta:** El modelo es errático o está "confundido", lo que sugiere que no ha capturado correctamente los patrones del lenguaje.
#
# ### Ventajas y limitaciones
#
# La perplejidad se ha usado extensivamente para evaluar modelos de lenguaje, pues entre otras cosas,
# es una métrica sencilla de calcular, y hasta cierto punto interpretable.
#
# Sin embargo, algunas desventajas que presenta es que no provee información sobre el desempeño semántico del modelo,
# es decir, es difícil que detecte cuando el modelo produce oraciones sintácticamente correctas, pero que carecen de sentido.
#
# Otra desventaja es que depende de alguna forma del tamaño del vocabulario.
# Por ejemplo, un modelo con perplejidad máxima, sería aquel que asigna las misma probabilidad a todas las palabras del vocabulario, en cuyo caso la perplejidad sería el tamaño del vocabulario.
# Es así que el tamaño de vocabulario funciona como una cota superior a la perplejidad, por lo que ésta depende de dicho tamaño y esto dificulta hacer comparaciones entre modelos entrenados con vocabularios distintos.
#

# %%
import os
import math
import time
import glob
import re
import pathlib
import pandas as pd
import numpy as np
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import nltk
from nltk.util import ngrams
from nltk.corpus import abc, genesis, gutenberg, inaugural, state_union, webtext
from transformers import AutoTokenizer

nltk.download("gutenberg", quiet=True)
nltk.download("abc", quiet=True)
nltk.download("genesis", quiet=True)
nltk.download("inaugural", quiet=True)
nltk.download("state_union", quiet=True)
nltk.download("webtext", quiet=True)
nltk.download("punkt_tab", quiet=True)

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Usando dispositivo: {device}")

# %%
NN_MODELS_PATH = pathlib.Path("models/nn")
os.makedirs(NN_MODELS_PATH, exist_ok=True)

# %%
# Hiperparámetros globales
EMBEDDING_DIM = 200
H = 100
# Dependiendo de la memoria del dispositivo puede ser necesario usar un valor más bajo
BATCH_SIZE = 8192
# BATCH_SIZE = 512
MAX_EPOCHS = 5
UNK_LABEL = "<UNK>"

# %%
plaintext_corpora = {
    "abc": abc,
    "Gutenberg": gutenberg,
    "Inaugural": inaugural,
    "State Union": state_union,
    "Web": webtext,
}

# %%
corpus_strings = []      # Para entrenar tokenizers de subwords
corpora_words = []       # Para el modelo de palabras completas

# %%
def lm_preprocess_corpus(corpus: list[list[str]]) -> list[list[str]]:
    """Agrega tokens de inicio/fin y pasa a minúsculas."""
    preprocessed = []
    for sent in corpus:
        result = [word.lower() for word in sent]
        result.append("<EOS>")
        result.insert(0, "<BOS>")
        preprocessed.append(result)
    return preprocessed

# %%
print("--- Procesando Corpus ---")
for title, corpus in plaintext_corpora.items():
    sents = corpus.sents()
    # Para subwords (Strings puros)
    for sent in sents:
        corpus_strings.append(" ".join(sent))
    # Para palabras completas (Listas procesadas)
    corpora_words.extend(lm_preprocess_corpus(sents))

print(f"Total de oraciones extraídas: {len(corpus_strings)}")

# %%
# Diccionario de frecuencias e índices para palabras completas
def get_words_indexes(corpus: list[list[str]]) -> tuple[dict, dict]:
    freqs = {}
    for sentence in corpus:
        for word in sentence:
            freqs[word] = freqs.get(word, 0) + 1
            
    result = {}
    for idx, word in enumerate(freqs.keys()):
        if freqs[word] == 1:
            result[UNK_LABEL] = len(freqs)
        else:
            result[word] = idx
            
    w2i = {word: idx for idx, word in enumerate(result.keys())}
    i2w = {idx: word for idx, word in enumerate(result.keys())}
    return w2i, i2w

# %%
words_indexes, index_to_word = get_words_indexes(corpora_words)

# %% [markdown]
# ### Tokenizers
# Entrenamos tokenizadores usando la biblioteca `transformers`

# %%
def train_custom_tokenizer(corpus_strings, vocab_size, base_model="bert-base-uncased"):
    print(f"Entrenando Tokenizer (Vocab: {vocab_size})...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model)
    custom_tokenizer = base_tokenizer.train_new_from_iterator(corpus_strings, vocab_size=vocab_size)
    return custom_tokenizer

# %%
TOKENIZERS = {}
vocab_sizes = [5000, 10000, 15000]

print("\n--- Generando Tokenizers ---")
for v_size in vocab_sizes:
    TOKENIZERS[v_size] = train_custom_tokenizer(corpus_strings, v_size)

# %% [markdown]
# #### Funciones auxiliares
# Para crear los dataloaderes necesarios y poder reutilizarlos según sea el caso.

# %%
def create_dataloader_words(corpus, words_indexes, n, batch_size):
    x_train, y_train = [], []
    unk_id = words_indexes[UNK_LABEL]
    
    for sent in corpus:
        if len(sent) < n: continue
        for gram in ngrams(sent, n):
            ids = [words_indexes.get(w, unk_id) for w in gram]
            x_train.append(ids[:-1]) # Contexto
            y_train.append([ids[-1]]) # Objetivo
            
    train_set = np.concatenate((x_train, y_train), axis=1)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)

# %%
def create_dataloader_subwords(corpus_strings, tokenizer, n, batch_size):
    x_train, y_train = [], []
    
    for sent_str in corpus_strings:
        token_ids = tokenizer.encode(sent_str, truncation=True, max_length=512)
        if len(token_ids) < n: continue
        for gram in ngrams(token_ids, n):
            x_train.append(list(gram[:-1]))
            y_train.append([gram[-1]])
            
    train_set = np.concatenate((x_train, y_train), axis=1)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)

# %%
DL_SW = {}

def get_dl_sw(vocab, ngrams_val):
    """
    Obtiene el DataLoader para subwords bajo demanda.
    Si ya fue calculado, lo devuelve del diccionario DL_SW.
    Si no, lo crea, lo guarda y lo devuelve.
    """
    key = (vocab, ngrams_val)
    
    if key not in DL_SW:
        print(f"Generando DataLoader: Vocab={vocab}, N-grams={ngrams_val}...")
        DL_SW[key] = create_dataloader_subwords(
            corpus_strings,
            TOKENIZERS[vocab],
            ngrams_val,
            BATCH_SIZE
        )
    
    return DL_SW[key]


# %%
DL_W = {}

def get_dl_words(ngrams):
    """
    Obtiene el DataLoader para palabras completas bajo demanda.
    Si ya existe para ese valor de n-gramas, lo recupera; de lo contrario, lo crea.
    """
    
    if ngrams not in DL_W:
        print(f"Generando DataLoader: N-grams={ngrams}...")
        DL_W[ngrams] = create_dataloader_words(
            corpora_words, 
            words_indexes, 
            ngrams, 
            BATCH_SIZE
        )
    
    return DL_W[ngrams]


# %% [markdown]
# #### Modelo
# Arquitectura genérica para los diferentes modelos a entrenar.

# %%
class NGramLanguageModel(nn.Module):
    """Adaptación del modelo original para soportar tamaños de contexto variables"""
    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(NGramLanguageModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)

# %%
def train_or_load(mode, target_epoch, n_gram, vocab_size, dataloader, device, retrain=False):
    """
    Función para cargar o entrenar modelos.
    Guarda los archivos indicando explícitamente sus parámetros.
    """
    context_size = n_gram - 1
    model_name = f"lm_{mode}_vocab_{vocab_size}_ngram_{n_gram}"
    model_pattern = os.path.join(NN_MODELS_PATH, f"{model_name}_epoch_*.dat")
    target_path = os.path.join(NN_MODELS_PATH, f"{model_name}_epoch_{target_epoch}.dat")

    model = NGramLanguageModel(vocab_size, EMBEDDING_DIM, context_size, H).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    loss_function = nn.NLLLoss()

    if not retrain and os.path.exists(target_path):
        model.load_state_dict(torch.load(target_path, map_location=device, weights_only=True))
        model.eval()
        return model

    start_epoch = 0
    existing_files = glob.glob(model_pattern)
    
    if not retrain and existing_files:
        epochs_found = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in existing_files]
        last_epoch = max(epochs_found)
        if last_epoch < target_epoch:
            start_epoch = last_epoch + 1
            last_model_path = os.path.join(NN_MODELS_PATH, f"{model_name}_epoch_{last_epoch}.dat")
            model.load_state_dict(torch.load(last_model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(target_path, map_location=device, weights_only=True))
            model.eval()
            return model

    print(f"\nEntrenando: Modo={mode} | Vocab={vocab_size} | N-Grama={n_gram} | Épocas={start_epoch} a {target_epoch}")
    for epoch in range(start_epoch, target_epoch + 1):
        if epoch == 0: continue
        model.train()
        total_loss = 0
        
        for data_tensor in dataloader:
            context_tensor = data_tensor[:, :-1].to(device)
            target_tensor = data_tensor[:, -1].to(device)

            model.zero_grad()
            log_probs = model(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        save_path = os.path.join(NN_MODELS_PATH, f"{model_name}_epoch_{epoch}.dat")
        torch.save(model.state_dict(), save_path)
        print(f"Época {epoch}/{target_epoch} guardada -> Loss Promedio: {total_loss/len(dataloader):.4f}")

    model.eval()
    return model

# %%
def generate_text_words(model, n_grams, words_indexes, index_to_word, seed_text, max_new_tokens=30, device="cpu"):
    """Generación basada en palabras completas."""
    model.eval()
    context_size = n_grams - 1
    
    seed_words = [w.lower() for w in seed_text.split()]
    unk_id = words_indexes[UNK_LABEL]
    generated_ids = [words_indexes.get(w, unk_id) for w in seed_words]
    
    # Rellenar si la semilla es más corta que el contexto necesario
    while len(generated_ids) < context_size:
        generated_ids.insert(0, words_indexes.get("<BOS>", unk_id))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-context_size:]
            context_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)
            
            log_probs = model(context_tensor)
            predicted_id = torch.argmax(log_probs, dim=1).item()
            generated_ids.append(predicted_id)
            
            if index_to_word.get(predicted_id) == "<EOS>":
                break
                
    return " ".join([index_to_word.get(idx, UNK_LABEL) for idx in generated_ids])

# %%
def generate_text_subwords(model, n_grams, tokenizer, seed_text, max_new_tokens=30, device="cpu"):
    """Generación basada en subwords."""
    model.eval()
    context_size = n_grams - 1
    generated_ids = tokenizer.encode(seed_text, add_special_tokens=True)
    
    # Rellenar con tokens de padding si es necesario
    while len(generated_ids) < context_size:
        generated_ids.insert(0, tokenizer.pad_token_id or 0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-context_size:]
            context_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)
            
            log_probs = model(context_tensor)
            predicted_id = torch.argmax(log_probs, dim=1).item()
            generated_ids.append(predicted_id)
            
            if predicted_id == tokenizer.sep_token_id:
                break
                
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# %% [markdown]
# #### Cálculo de perplejidades

# %%
genesis_sents_words = lm_preprocess_corpus(genesis.sents())
genesis_sents_strings = [" ".join(sent) for sent in genesis.sents()]

# %%
def calculate_perplexity(model, mode, n, meta_data, device):
    """Calcula la perplejidad sobre el corpus Genesis"""
    model.eval()
    total_loss, total_tokens = 0, 0
    context_size = n - 1
    loss_fn = nn.NLLLoss(reduction='sum')
    
    with torch.no_grad():
        if mode == "words":
            words_indexes = meta_data
            unk_id = words_indexes[UNK_LABEL]
            for sent in genesis_sents_words:
                if len(sent) < n: continue
                for gram in ngrams(sent, n):
                    ids = [words_indexes.get(w, unk_id) for w in gram]
                    x = torch.tensor([ids[:-1]]).to(device)
                    y = torch.tensor([ids[-1]]).to(device)
                    total_loss += loss_fn(model(x), y).item()
                    total_tokens += 1
        else: # subwords
            tokenizer = meta_data
            for sent_str in genesis_sents_strings:
                token_ids = tokenizer.encode(sent_str, truncation=True, max_length=512)
                if len(token_ids) < n: continue
                for gram in ngrams(token_ids, n):
                    x = torch.tensor([list(gram[:-1])]).to(device)
                    y = torch.tensor([gram[-1]]).to(device)
                    total_loss += loss_fn(model(x), y).item()
                    total_tokens += 1
                    
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

# %%
def compute_perplexities():
    results = []
    
    # Calcula métricas con palabras
    vocab_size_words = len(words_indexes)
    for n in [3, 4, 5]:
        print(f"\n--- Preparando DataLoader (Palabras | N={n}) ---")
        dl = create_dataloader_words(corpora_words, words_indexes, n, BATCH_SIZE)
        
        model = train_or_load("words", MAX_EPOCHS, n, vocab_size_words, dl, device)
        ppl = calculate_perplexity(model, "words", n, words_indexes, device)
        
        results.append({
            "Tipo": "Palabras Completas",
            "N-Gram": n,
            "Vocabulario": vocab_size_words,
            "Épocas": MAX_EPOCHS,
            "Perplexity (Genesis)": ppl
        })
        
    # Calcula métricas con subpalabras
    for v_size in [5000, 10000, 15000]:
        tok = TOKENIZERS[v_size]
        actual_v_size = len(tok)
        
        for n in [3, 4, 5]:
            print(f"\n--- Preparando DataLoader (Subwords | Vocab={v_size} | N={n}) ---")
            dl = create_dataloader_subwords(corpus_strings, tok, n, BATCH_SIZE)
            
            model = train_or_load("subwords", MAX_EPOCHS, n, actual_v_size, dl, device)
            ppl = calculate_perplexity(model, "subwords", n, tok, device)
            
            results.append({
                "Tipo": "Subwords",
                "N-Gram": n,
                "Vocabulario": actual_v_size,
                "Épocas": MAX_EPOCHS,
                "Perplexity (Genesis)": ppl
            })
            
    df = pd.DataFrame(results)
    return df

# %% [markdown]
# Ejecutar la siguiente celda evalua todos los modelos. En caso de no encontrar algún modelo empieza por entrenarlo.

# %%
# df_results = compute_perplexities()

# %% [markdown]
# ## Análisis comparativo.
#
# En un primer intento, utilicé el tokenizador preentrenado de la librería `transformers`, y entrené los modelos por 30 épocas, lo que tomó bastante tiempo. Al comparar el desempeño de ambos por época noté que la perplejidad del modelo con subpalabras era mucho mayor, y más aún incrementaba con la época de entrenamiento.
#
# Después de revisar con más detalle y buscar causas posibles a este resultado, encontré que el tokenizador preentrenado (que usa WordPiece) tiene un vocabulario fijo de 30,522 tokens, por lo que pensé que podía ser demasiado grande para nuestro corpus.
#
# Otro problema que se presentaba, muy notorio al generar texto, era que un contexto de 3 palabras es relativamente mayor a un contexto de únicamente 3 subpalabras. Es decir, dado que las unidades léxicas son mucho menores en el caso de subpalabras, podríamos pensar que contienen menor información, y por lo tanto requieren un contexto mayor.
#
# Fue entonces que decidí entrenar un tokenizador con nuestro corpus para que la comparación fuese más realista, aunque aún así se presenta el problema de que interpretar la perplejidad cuando el tamaño de vocabulario es distinto no es tan sencillo.
#
# Además pensé en comparar modelos con contextos de 3 a 5 n-gramas.
#
# Finalmente esto produjo una gran diversidad de modelos (lo que no permitió entrenarlos por muchas épocas) y la comparación de perplejidades es como sigue:
#

# %%
# df_results.to_csv("perplexity.csv")

# %% [markdown]
# Los resultados obtenidos los subimos al repositorio:

# %%
df = pd.read_csv("perplexity.csv", index_col=0)

# %%
df

# %% [markdown]
# En términos generales vemos que la perplejidad aumenta con el tamaño de vocabulario (cosa que no es extraña), pero la otra cosa notable es que los modelos de subpalabras tienen perplejidad considerablemente más alta.
# Me gustaría entrenar los modelos por más épocas para ver cómo cambia este resultado.
#

# %% [markdown]
# Una posible hipótesis para explicar por qué el modelo de palabras tiene una perplejidad mucho más baja a pesar de tener un vocabulario mucho más grande, es que ocurran muchos OOV, que al manejarse manualmente como "UNK" podría alterar artificialmente la perplejidad.

# %%
# Cálculo para el corpus de palabras
total_w, oov_w = 0, 0
for sent in genesis_sents_words:
    for word in sent:
        total_w += 1
        if word not in words_indexes:
            oov_w += 1
oov_rate_words = (oov_w / total_w) * 100 if total_w > 0 else 0

# Cálculo para los Tokenizers de Subwords (5k, 10k, 15k)
oov_subwords = {}
for v_size in [5000, 10000, 15000]:
    tok = TOKENIZERS[v_size]
    total_s, oov_s = 0, 0
    unk_id = tok.unk_token_id
    
    for sent_str in genesis_sents_strings:
        token_ids = tok.encode(sent_str, add_special_tokens=False)
        total_s += len(token_ids)
        oov_s += token_ids.count(unk_id)
    
    oov_subwords[v_size] = (oov_s / total_s) * 100 if total_s > 0 else 0

print(f"{'CONFIGURACIÓN':<25} | {'TASA OOV':<10}")
print("-" * 40)
print(f"{'Palabras Completas':<25} | {oov_rate_words:>8.2f}%")
print(f"{'Subwords (5k)':<25} | {oov_subwords[5000]:>8.2f}%")
print(f"{'Subwords (10k)':<25} | {oov_subwords[10000]:>8.2f}%")
print(f"{'Subwords (15k)':<25} | {oov_subwords[15000]:>8.2f}%")


# %% [markdown]
# ### Uso de modelos "preentrenados"
#
# Dado que volver a entrenar estos modelos consume muchos recursos, las siguientes funciones permiten descargar bajo demanda los modelos que obtuvimos.
#
# Posteriormente se pueden cargar con la función `train_or_load` y se pueden usar para calcular manualmente la perplejidad, o para hacer inferencia y generar texto.

# %%
def download_model(mode, target_epoch, n_gram, vocab_size, silent=False):
    """
    Descarga un modelo preentrenado desde el repositorio de GitHub si no existe localmente.
    Sigue el patrón de nombres: lm_[mode]_vocab_[V]_ngram_[N]_epoch_[E].dat
    """
    model_name = f"lm_{mode}_vocab_{vocab_size}_ngram_{n_gram}_epoch_{target_epoch}.dat"
    local_path = os.path.join(NN_MODELS_PATH, model_name)
    
    base_url = "https://github.com/romanbott/lm_perplexity_w/raw/refs/heads/main/models/nn/"
    url = f"{base_url}{model_name}?download="
    
    if os.path.exists(local_path):
        if not silent:
            print(f"El modelo {model_name} ya se encuentra en local.")
        return local_path

        print(f"Descargando modelo desde: {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Descarga completada: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error al descargar el modelo {model_name}: {e}")
        return None


# %% [markdown]
# #### Ejemplo

# %%
vocab = 10000
n_grams = 4
epoch = 5

# %%
download_model("subwords", epoch, n_grams, vocab)

# %%
dl = get_dl_sw(vocab, n_grams)
download_model("subwords", epoch, n_grams, vocab, silent=True)
model = train_or_load("subwords", epoch, n_grams, vocab, dl, "cpu")

# %%
calculate_perplexity(model, "subwords", n_grams, TOKENIZERS[vocab], "cpu")


# %% [markdown]
# #### Generación de texto
#
# Las siguientes funciones simplemente son "wrappers" de las funciones ya definidas y proveen una manera conveniente de generar texto con diferentes modelos:

# %%
def gen_text_wrapper_w(epoch, ngrams, seed):
    download_model("words", epoch, ngrams, 41392, silent=True)
    dl = get_dl_words(ngrams)
    model = train_or_load("words", epoch, ngrams, 41392, dl, "cpu")
    return generate_text_words(model, ngrams, words_indexes, index_to_word, seed, max_new_tokens=30, device="cpu")


# %%
def gen_text_wrapper_sw(epoch, ngrams, vocab, seed):
    dl = get_dl_sw(vocab, ngrams)
    download_model("subwords", epoch, ngrams, vocab, silent=True)
    model = train_or_load("subwords", epoch, ngrams, vocab, dl, "cpu")
    return generate_text_subwords(model, ngrams, TOKENIZERS[vocab], seed, device="cpu")
    


# %%
gen_text_wrapper_w(5, 5, "today i will")

# %%
gen_text_wrapper_sw(5, 5, 15000, "today i will")
