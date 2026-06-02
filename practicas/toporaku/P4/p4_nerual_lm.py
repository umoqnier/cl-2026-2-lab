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
# # Práctica 4. Modelo de Lenguaje Neuronal
# ### Lingüística Computacional 2026-2
# #### Cuerpo Académico
# **Dra.** María Ximena Gutiérrez Vasques
#
# **Ayud.** Ximena de la Luz Contreras Mendoza
#
# **Lab.** Diego Alberto Barriga Martínez
#
# #### Alumno
# Toporek Coca Eric - **314284987**

# %% [markdown]
# ## Objetivos
#
# - Entrenar un modelo de lenguaje neuronal (trigrama, estilo Bengio) sobre corpus de NLTK
# - Aplicar sub-word tokenization (BPE) al corpus de entrenamiento
# - Evaluar el modelo utilizando el corpus **genesis** de NLTK como conjunto de prueba
# - Calcular la **perplejidad** del modelo

# %% [markdown]
# ## 1. Instalación de dependencias y descarga de corpus

# %%
# %pip install tokenizers -q
# %pip install torch

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
import os
import math
import time
import numpy as np
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from rich import print as rprint
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from nltk.corpus import abc, genesis, gutenberg, inaugural, state_union, webtext

# %% [markdown]
# ## 2. Funciones de preprocesamiento
#
# Utilizamos las mismas funciones que se vieron en el notebook de referencia (7_neural_lm). Cada oración se delimita con tokens especiales `<BOS>` y `<EOS>`.

# %%
UNK_LABEL = "<UNK>"
BOS_LABEL = "<BOS>"
EOS_LABEL = "<EOS>"


def lm_preprocess_corpus(corpus_sents: list[list[str]]) -> list[list[str]]:
    """Preprocesa oraciones: lowercase + agrega BOS/EOS."""
    preprocessed_corpus = []
    for sent in corpus_sents:
        result = [w.lower() for w in sent]
        result.append(EOS_LABEL)
        result.insert(0, BOS_LABEL)
        preprocessed_corpus.append(result)
    return preprocessed_corpus


def flatten_corpus(corpus: list[list[str]]) -> str:
    """Aplana un corpus de oraciones a texto plano."""
    return "\n".join(" ".join(sent) for sent in corpus)


# %% [markdown]
# ## 3. Carga de corpus de NLTK
#
# Usamos los corpus de NLTK (abc, gutenberg, inaugural, state_union, webtext) como **entrenamiento** y reservamos el corpus **genesis** exclusivamente para **test**.

# %%
# --- Corpus de entrenamiento ---
train_corpora = []
total_sents = 0

plaintext_corpora = {
    "abc": abc,
    "Gutenberg": gutenberg,
    "Inaugural": inaugural,
    "State Union": state_union,
    "Web": webtext,
}

for title, corpus in plaintext_corpora.items():
    corpus_sents = lm_preprocess_corpus(corpus.sents())
    corpus_len = len(corpus_sents)
    total_sents += corpus_len
    print(f"{title} sents={corpus_len}")
    train_corpora.extend(corpus_sents)

print(f"Total train sents={total_sents}")

# --- Corpus de test (genesis) ---
test_corpora = lm_preprocess_corpus(genesis.sents())
print(f"Genesis (test) sents={len(test_corpora)}")

# %% [markdown]
# ## 4. Entrenamiento del tokenizador BPE (sub-word)
#
# Entrenamos un tokenizador BPE **desde cero** utilizando la librería `tokenizers` de Hugging Face sobre el texto del corpus de entrenamiento. Esto nos permite representar palabras menos frecuentes como combinaciones de sub-palabras, reduciendo el problema de OOV (out-of-vocabulary).

# %%
os.makedirs("models", exist_ok=True)

# Escribimos el corpus de entrenamiento a un archivo temporal para el BPE
TRAIN_TXT = "models/train_corpus.txt"
with open(TRAIN_TXT, "w") as f:
    f.write(flatten_corpus(train_corpora))

# Entrenamos el tokenizador BPE con un vocabulario de 5000 sub-words
VOCAB_SIZE = 5000
tokenizer = Tokenizer(BPE(unk_token=UNK_LABEL))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[UNK_LABEL, BOS_LABEL, EOS_LABEL],
    min_frequency=2,
)
tokenizer.train([TRAIN_TXT], trainer)

BPE_PATH = "models/bpe_tokenizer.json"
tokenizer.save(BPE_PATH)

print(f"Vocabulario BPE: {tokenizer.get_vocab_size()} sub-words")

# %%
# Demostración de la tokenización sub-word
demo_sents = ["the quick brown fox jumps", "uncharacteristically wonderful"]
for s in demo_sents:
    tokens = tokenizer.encode(s).tokens
    rprint(f"[bold]{s}[/bold] -> {tokens}")


# %% [markdown]
# ## 5. Tokenización sub-word del corpus
#
# Aplicamos el tokenizador BPE entrenado a cada oración de los corpus. Mantenemos los tokens especiales `<BOS>` y `<EOS>` intactos.

# %%
def tokenize_corpus_bpe(corpus: list[list[str]], tok: Tokenizer) -> list[list[str]]:
    """Aplica BPE a cada oración, preservando BOS/EOS."""
    result = []
    for sent in corpus:
        # El interior de la oración (sin BOS/EOS) se pasa por BPE
        inner = " ".join(sent[1:-1])
        bpe_tokens = tok.encode(inner).tokens
        result.append([BOS_LABEL] + bpe_tokens + [EOS_LABEL])
    return result


train_bpe = tokenize_corpus_bpe(train_corpora, tokenizer)
test_bpe = tokenize_corpus_bpe(test_corpora, tokenizer)

rprint(f"Ejemplo oración original: {train_corpora[42]}")
rprint(f"Ejemplo oración BPE:      {train_bpe[42]}")
print(f"\nTrain BPE sents: {len(train_bpe)}")
print(f"Test BPE sents:  {len(test_bpe)}")


# %% [markdown]
# ## 6. Construcción del vocabulario y datos de entrenamiento

# %%
def build_vocab(corpus: list[list[str]]) -> tuple[dict, dict]:
    """Construye token2idx e idx2token a partir del corpus BPE."""
    freqs = Counter(tok for sent in corpus for tok in sent)
    # Hapax legomena -> UNK
    token2idx = {}
    idx = 0
    for token, freq in freqs.items():
        if freq == 1:
            if UNK_LABEL not in token2idx:
                token2idx[UNK_LABEL] = idx
                idx += 1
        else:
            token2idx[token] = idx
            idx += 1
    if UNK_LABEL not in token2idx:
        token2idx[UNK_LABEL] = idx
    idx2token = {v: k for k, v in token2idx.items()}
    return token2idx, idx2token


token2idx, idx2token = build_vocab(train_bpe)
V = len(token2idx)
print(f"Tamaño del vocabulario (V): {V}")

# %%
from nltk import ngrams


def get_token_id(token2idx: dict, token: str) -> int:
    return token2idx.get(token, token2idx[UNK_LABEL])


def corpus_to_trigram_data(
    corpus: list[list[str]], token2idx: dict
) -> tuple[list, list]:
    """Convierte corpus BPE a pares (contexto, target) de trigramas."""
    x, y = [], []
    for sent in corpus:
        for w1, w2, w3 in ngrams(sent, 3):
            x.append([get_token_id(token2idx, w1), get_token_id(token2idx, w2)])
            y.append([get_token_id(token2idx, w3)])
    return x, y


x_train, y_train = corpus_to_trigram_data(train_bpe, token2idx)
print(f"Trigramas de entrenamiento: {len(x_train):,}")


# %% [markdown]
# ## 7. Arquitectura del modelo (Bengio Trigram)
#
# Implementamos la misma arquitectura del notebook de referencia:
#
# $x' = e(x_1) \oplus e(x_2)$
#
# $h = \tanh(W_1 x' + b)$
#
# $y = \text{log\_softmax}(W_2 h)$

# %%
class TrigramModel(nn.Module):
    """Modelo de lenguaje neuronal basado en trigramas (Bengio et al., 2003)."""

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(
            (-1, self.context_size * self.embedding_dim)
        )
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# %% [markdown]
# ## 8. Entrenamiento del modelo

# %%
# Hiperparámetros
EMBEDDING_DIM = 128
CONTEXT_SIZE = 2
HIDDEN_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 3
LR = 2e-3

torch.manual_seed(42)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Dispositivo: {device}")
print(f"Vocab size: {V}, Embedding: {EMBEDDING_DIM}, Hidden: {HIDDEN_SIZE}")

# %%
# Preparar DataLoader
train_set = np.concatenate((x_train, y_train), axis=1)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Modelo, loss y optimizador
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Ciclo de entrenamiento
os.makedirs("models/nn", exist_ok=True)

for epoch in range(EPOCHS):
    st = time.time()
    total_loss = 0.0
    n_batches = 0
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    for it, data_tensor in enumerate(train_loader):
        context = data_tensor[:, 0:2].to(device)
        target = data_tensor[:, 2].to(device)

        model.zero_grad()
        log_probs = model(context)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if it % 500 == 0:
            print(f"  Iter {it:>5d} | Loss: {loss.item():.4f} | Tiempo: {time.time()-st:.1f}s")
            st = time.time()

    avg_loss = total_loss / n_batches
    print(f"  -> Loss promedio epoch {epoch + 1}: {avg_loss:.4f}")

# Guardar modelo
MODEL_PATH = f"models/nn/trigram_bpe_{device}_e{EPOCHS}.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModelo guardado en: {MODEL_PATH}")

# %%
rprint(model)

# %% [markdown]
# ## 8.1 Exportación del modelo
#
# Exportamos **todos los artefactos** necesarios para reproducir el modelo en cualquier entorno:
# - `model_weights.pt`: pesos del modelo (state_dict)
# - `bpe_tokenizer.json`: tokenizador BPE entrenado
# - `vocab.json`: mapeos token↔idx
# - `config.json`: hiperparámetros de la arquitectura
#
# Esta carpeta `models/export/` se puede subir directamente a **Hugging Face Hub**, **Google Drive**, o comprimirse como `.tar.gz`.

# %%
import json
import shutil

EXPORT_DIR = "models/export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 1. Guardar pesos del modelo
torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "model_weights.pt"))

# 2. Copiar el tokenizador BPE
shutil.copy(BPE_PATH, os.path.join(EXPORT_DIR, "bpe_tokenizer.json"))

# 3. Guardar vocabulario (token2idx e idx2token)
vocab_data = {
    "token2idx": token2idx,
    "idx2token": {str(k): v for k, v in idx2token.items()},
}
with open(os.path.join(EXPORT_DIR, "vocab.json"), "w") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

# 4. Guardar configuración de hiperparámetros
config = {
    "model_type": "TrigramModel",
    "vocab_size": V,
    "embedding_dim": EMBEDDING_DIM,
    "context_size": CONTEXT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "bpe_vocab_size": VOCAB_SIZE,
    "epochs_trained": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "special_tokens": {"unk": UNK_LABEL, "bos": BOS_LABEL, "eos": EOS_LABEL},
}
with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

rprint(f"[bold green]Modelo exportado en:[/bold green] {EXPORT_DIR}/")
for fname in sorted(os.listdir(EXPORT_DIR)):
    fsize = os.path.getsize(os.path.join(EXPORT_DIR, fname))
    rprint(f"  📦 {fname} ({fsize / 1024:.1f} KB)")

# %%
# 5. Comprimir para distribución
ARCHIVE_PATH = "models/trigram_bpe_model"
shutil.make_archive(ARCHIVE_PATH, "gztar", EXPORT_DIR)
archive_size = os.path.getsize(f"{ARCHIVE_PATH}.tar.gz") / 1024
rprint(f"[bold]Archivo comprimido:[/bold] {ARCHIVE_PATH}.tar.gz ({archive_size:.1f} KB)")
rprint("\n[dim]Listo para subir a Hugging Face Hub, Google Drive, GitHub Releases, etc.[/dim]")


# %% [markdown]
# ## 9. Evaluación: Cálculo de Perplejidad sobre Genesis
#
# La perplejidad se calcula como:
#
# $$PP(W) = e^{H(W)}$$
#
# donde la entropía cruzada es:
#
# $$H(W) = -\frac{1}{N} \sum_{i=1}^N \log P(w_i | w_{i-2}, w_{i-1})$$
#
# En la práctica, acumulamos el **NLLLoss** promedio sobre todos los trigramas del test y luego exponenciamos.

# %%
def calculate_perplexity(
    model: TrigramModel,
    test_corpus: list[list[str]],
    token2idx: dict,
    device: str,
    batch_size: int = 512,
) -> tuple[float, float, int]:
    """Calcula la perplejidad del modelo sobre un corpus de test.

    1. Extrae trigramas del corpus de test.
    2. Calcula la NLL promedio con el modelo.
    3. Exponencia para obtener la perplejidad.
    """
    model.eval()
    x_test, y_test = corpus_to_trigram_data(test_corpus, token2idx)
    test_set = np.concatenate((x_test, y_test), axis=1)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0
    nll_fn = nn.NLLLoss(reduction="sum")

    with torch.no_grad():
        for data_tensor in test_loader:
            context = data_tensor[:, 0:2].to(device)
            target = data_tensor[:, 2].to(device)
            log_probs = model(context)
            loss = nll_fn(log_probs, target)
            total_loss += loss.item()
            total_tokens += target.size(0)

    avg_nll = total_loss / total_tokens  # H(W)
    perplexity = math.exp(avg_nll)       # PP(W) = e^{H(W)}
    return perplexity, avg_nll, total_tokens


# %%
ppl, avg_nll, n_tokens = calculate_perplexity(model, test_bpe, token2idx, device)

rprint(f"[bold green]Resultados sobre el corpus Genesis (test):[/bold green]")
rprint(f"  Trigramas evaluados: {n_tokens:,}")
rprint(f"  Entropía cruzada (H): {avg_nll:.4f}")
rprint(f"  [bold]Perplejidad (PP): {ppl:.2f}[/bold]")

# %% [markdown]
# ## 10. Análisis de resultados
#
# ### Interpretación de la perplejidad
#
# La perplejidad obtenida nos indica, en promedio, entre cuántas sub-palabras de nuestro vocabulario el modelo está "indeciso" al momento de predecir el siguiente token.
#
# Factores que influyen en el valor:
# - **Dominio del test**: el corpus *genesis* es texto bíblico, mientras que el entrenamiento incluye noticias, discursos y contenido web. Esta discrepancia de dominio incrementa la perplejidad.
# - **Tamaño del vocabulario BPE**: un vocabulario más grande puede capturar más palabras completas pero aumenta el espacio de búsqueda.
# - **Ventana de contexto**: nuestro modelo solo observa 2 tokens previos (trigrama), limitando la información contextual disponible.

# %% [markdown]
# ## 11. Modelo Base (word-level) del notebook de referencia
#
# Para realizar el análisis comparativo, implementamos el modelo **word-level**
# definido en `7_neural_lm.py`. Este modelo **no** usa tokenización BPE:
# opera directamente sobre palabras completas, reemplazando los *hapax legomena*
# (frecuencia = 1) por `<UNK>`.
#
# ### Diferencias clave con el Modelo Subword
#
# | Aspecto | Modelo Base (word-level) | Modelo Subword (BPE) |
# |---------|--------------------------|----------------------|
# | Tokenización | Palabras completas | Sub-words BPE (5 000) |
# | Embedding dim | 200 | 128 |
# | Hidden size | 100 | 64 |
# | Épocas | 5 | 3 |

# %%
# --- Funciones del modelo base (7_neural_lm.py) ---

def get_words_freqs(corpus: list[list[str]]) -> dict:
    """Calcula la frecuencia de las palabras en un corpus."""
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs


def get_words_indexes(words_freqs: dict) -> tuple[dict, dict]:
    """Calcula los índices de las palabras dadas sus frecuencias.

    Los hapax legomena se mapean a <UNK>.
    """
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        if words_freqs[word] == 1:
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx
    return (
        {word: idx for idx, word in enumerate(result.keys())},
        {idx: word for idx, word in enumerate(result.keys())},
    )


def get_word_id_base(words_indexes: dict, word: str) -> int:
    """Obtiene el id de una palabra; si no existe, devuelve el id de UNK."""
    return words_indexes.get(word, words_indexes[UNK_LABEL])


# %%
# --- Construir vocabulario word-level ---
words_freqs = get_words_freqs(train_corpora)
words_indexes, index_to_word = get_words_indexes(words_freqs)
V_base = len(words_indexes)

rprint(f"[bold]Vocabulario word-level (V_base):[/bold] {V_base}")
rprint(f"[bold]Vocabulario BPE       (V_bpe): [/bold] {V}")

# %%
# --- Preparar trigramas word-level ---
from nltk import ngrams as nltk_ngrams


def corpus_to_trigram_data_base(
    corpus: list[list[str]], words_indexes: dict
) -> tuple[list, list]:
    """Convierte corpus word-level a pares (contexto, target) de trigramas."""
    x, y = [], []
    for sent in corpus:
        for w1, w2, w3 in nltk_ngrams(sent, 3):
            x.append(
                [get_word_id_base(words_indexes, w1), get_word_id_base(words_indexes, w2)]
            )
            y.append([get_word_id_base(words_indexes, w3)])
    return x, y


x_train_base, y_train_base = corpus_to_trigram_data_base(train_corpora, words_indexes)
print(f"Trigramas word-level de entrenamiento: {len(x_train_base):,}")

# %%
# --- Modelo Base: misma arquitectura TrigramModel, distintos hiperparámetros ---
EMBEDDING_DIM_BASE = 200
HIDDEN_SIZE_BASE = 100
EPOCHS_BASE = 5

torch.manual_seed(42)

train_set_base = np.concatenate((x_train_base, y_train_base), axis=1)
train_loader_base = DataLoader(train_set_base, batch_size=BATCH_SIZE, shuffle=True)

model_base = TrigramModel(V_base, EMBEDDING_DIM_BASE, CONTEXT_SIZE, HIDDEN_SIZE_BASE).to(device)
loss_fn_base = nn.NLLLoss()
optimizer_base = optim.Adam(model_base.parameters(), lr=LR)

for epoch in range(EPOCHS_BASE):
    st = time.time()
    total_loss = 0.0
    n_batches = 0
    print(f"\n--- [Base] Epoch {epoch + 1}/{EPOCHS_BASE} ---")
    for it, data_tensor in enumerate(train_loader_base):
        context = data_tensor[:, 0:2].to(device)
        target = data_tensor[:, 2].to(device)
        model_base.zero_grad()
        log_probs = model_base(context)
        loss = loss_fn_base(log_probs, target)
        loss.backward()
        optimizer_base.step()
        total_loss += loss.item()
        n_batches += 1
        if it % 500 == 0:
            print(f"  Iter {it:>5d} | Loss: {loss.item():.4f} | Tiempo: {time.time()-st:.1f}s")
            st = time.time()
    avg_loss = total_loss / n_batches
    print(f"  -> Loss promedio epoch {epoch + 1}: {avg_loss:.4f}")

rprint("[bold green]Entrenamiento del Modelo Base completado.[/bold green]")
rprint(model_base)

# %% [markdown]
# ### 11.1 Perplejidad del Modelo Base sobre Genesis
#
# Evaluamos el modelo base **word-level** con la misma función `calculate_perplexity`,
# pero usando el corpus de test **sin** tokenización BPE y el vocabulario word-level.

# %%
def calculate_perplexity_base(
    model: TrigramModel,
    test_corpus: list[list[str]],
    words_indexes: dict,
    device: str,
    batch_size: int = 512,
) -> tuple[float, float, int]:
    """Calcula la perplejidad del modelo base (word-level) sobre un corpus de test."""
    model.eval()
    x_test, y_test = corpus_to_trigram_data_base(test_corpus, words_indexes)
    test_set = np.concatenate((x_test, y_test), axis=1)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0
    nll_fn = nn.NLLLoss(reduction="sum")

    with torch.no_grad():
        for data_tensor in test_loader:
            context = data_tensor[:, 0:2].to(device)
            target = data_tensor[:, 2].to(device)
            log_probs = model(context)
            loss = nll_fn(log_probs, target)
            total_loss += loss.item()
            total_tokens += target.size(0)

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, avg_nll, total_tokens


# %%
ppl_base, avg_nll_base, n_tokens_base = calculate_perplexity_base(
    model_base, test_corpora, words_indexes, device
)

rprint(f"[bold green]Resultados Modelo Base (word-level) sobre Genesis:[/bold green]")
rprint(f"  Trigramas evaluados: {n_tokens_base:,}")
rprint(f"  Entropía cruzada (H): {avg_nll_base:.4f}")
rprint(f"  [bold]Perplejidad (PP): {ppl_base:.2f}[/bold]")

# %% [markdown]
# ### 11.2 Tasa de OOV (Out-of-Vocabulary)
#
# Calculamos la proporción de tokens en el corpus de test que son desconocidos
# para cada modelo. Un modelo con BPE debería tener una tasa de OOV
# significativamente menor.

# %%
# --- OOV rate: Modelo Base (word-level) ---
test_tokens_word = [tok for sent in test_corpora for tok in sent]
oov_base = sum(1 for tok in test_tokens_word if tok not in words_indexes) / len(test_tokens_word)

# --- OOV rate: Modelo Subword (BPE) ---
test_tokens_bpe = [tok for sent in test_bpe for tok in sent]
oov_bpe = sum(1 for tok in test_tokens_bpe if tok not in token2idx) / len(test_tokens_bpe)

rprint(f"[bold]OOV Rate Modelo Base (word-level):[/bold] {oov_base:.4%}")
rprint(f"[bold]OOV Rate Modelo Subword (BPE):    [/bold] {oov_bpe:.4%}")

# %% [markdown]
# ## 12. Análisis Comparativo
#
# ### Tabla resumen
#
# A continuación se presenta la comparación cuantitativa entre ambos modelos:

# %%
# --- Tabla comparativa ---
print(f"{'Métrica':<30} {'Modelo Base':>15} {'Modelo Subword':>15}")
print("-" * 62)
print(f"{'Perplejidad (genesis)':<30} {ppl_base:>15.2f} {ppl:>15.2f}")
print(f"{'Entropía cruzada (H)':<30} {avg_nll_base:>15.4f} {avg_nll:>15.4f}")
print(f"{'Tamaño vocabulario':<30} {V_base:>15,} {V:>15,}")
print(f"{'OOV Rate (genesis)':<30} {oov_base:>15.4%} {oov_bpe:>15.4%}")
print(f"{'Embedding dim':<30} {EMBEDDING_DIM_BASE:>15} {EMBEDDING_DIM:>15}")
print(f"{'Hidden size':<30} {HIDDEN_SIZE_BASE:>15} {HIDDEN_SIZE:>15}")
print(f"{'Épocas entrenadas':<30} {EPOCHS_BASE:>15} {EPOCHS:>15}")
print(f"{'Trigramas test':<30} {n_tokens_base:>15,} {n_tokens:>15,}")

# %% [markdown]
# ### Discusión
#
# **Perplejidad:** El modelo subword (BPE) opera sobre un vocabulario más pequeño
# (~5 000 sub-words vs ~30 000+ palabras), lo que reduce el espacio de predicción
# y puede resultar en una perplejidad numérica menor. Sin embargo, la perplejidad
# de modelos con distintos vocabularios **no es directamente comparable**, ya que
# cada modelo predice unidades distintas (sub-words vs palabras completas).
#
# **OOV Rate:** El modelo BPE tiene una tasa de OOV mucho más baja, ya que puede
# descomponer palabras desconocidas en sub-unidades conocidas, mientras que el
# modelo word-level mapea todas las palabras no vistas a `<UNK>`.
#
# **Ventajas del Modelo Base:**
# - Predicciones directamente interpretables a nivel de palabra
# - Mayor embedding dim y hidden size → mayor capacidad de representación
# - Más épocas de entrenamiento
#
# **Ventajas del Modelo Subword (BPE):**
# - Vocabulario compacto y controlable
# - Tasa de OOV significativamente menor
# - Mejor generalización para palabras raras o no vistas
# - Se puede aplicar a cualquier texto sin preocuparse por palabras desconocidas
#
# **Recomendaciones para mejorar ambos modelos:**
# - Aumentar el contexto (n-gramas de orden mayor, e.g., 4-gramas o 5-gramas)
# - Usar arquitecturas más expresivas (LSTM, Transformer)
# - Entrenar más épocas con learning rate scheduling
# - Aumentar el corpus de entrenamiento
# - Para el modelo BPE: experimentar con distintos tamaños de vocabulario

# %% [markdown]
# ## EXTRA: Generación de palabras completas con sub-word tokenization
#
# ### Estrategia
#
# El modelo genera **sub-words** (tokens BPE), no palabras completas. Para reconstruir palabras necesitamos una estrategia de decodificación que fusione los sub-tokens en palabras reales.
#
# **Pipeline de generación:**
#
# 1. Se parte de un contexto semilla (e.g. `<BOS> the`)
# 2. El modelo genera sub-words una a una usando **top-p (nucleus) sampling**
# 3. Se acumulan los sub-word tokens generados en una lista
# 4. Se utiliza `tokenizer.decode()` del tokenizador BPE para reconstruir el texto con **palabras completas**, ya que el decodificador conoce las reglas de fusión de los sub-tokens
# 5. Se eliminan los tokens especiales (`<BOS>`, `<EOS>`) de la salida final

# %%
import random


def generate_words(
    model: TrigramModel,
    bpe_tokenizer: Tokenizer,
    seed: str,
    token2idx: dict,
    idx2token: dict,
    max_tokens: int = 60,
    top_p: float = 0.85,
    device: str = "cpu",
) -> str:
    """Genera texto con palabras completas a partir de sub-word tokens.

    Estrategia:
      1. Genera sub-words con nucleus sampling (top-p)
      2. Fusiona sub-words en palabras completas con tokenizer.decode()
    """
    model.eval()
    words = seed.split()
    generated_subwords = []

    for _ in range(max_tokens):
        idx1 = get_token_id(token2idx, words[-2])
        idx2 = get_token_id(token2idx, words[-1])

        with torch.no_grad():
            log_probs = model(torch.tensor([[idx1, idx2]]).to(device))

        probs = torch.exp(log_probs).squeeze()

        # --- Top-p (nucleus) sampling ---
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        # Crear máscara: conservar tokens hasta que la prob acumulada >= top_p
        mask = cumulative_probs - sorted_probs < top_p
        filtered_probs = sorted_probs * mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum()  # re-normalizar

        chosen = torch.multinomial(filtered_probs, 1).item()
        next_idx = sorted_indices[chosen].item()
        next_token = idx2token.get(next_idx, UNK_LABEL)

        if next_token == EOS_LABEL:
            break

        generated_subwords.append(next_token)
        words = [words[-1], next_token]

    # --- Reconstrucción de palabras completas ---
    # Convertir tokens de texto a IDs del tokenizador BPE
    bpe_ids = []
    for tok in generated_subwords:
        tid = bpe_tokenizer.token_to_id(tok)
        if tid is not None:
            bpe_ids.append(tid)

    # El decoder del tokenizador fusiona sub-words en palabras completas
    decoded_text = bpe_tokenizer.decode(bpe_ids)
    return decoded_text


# %% [markdown]
# ### Ejemplos de generación
#
# A continuación generamos 3 secuencias con diferentes semillas y parámetros de sampling.

# %%
# Ejemplo 1: Semilla narrativa
seed_1 = f"{BOS_LABEL} the"
text_1 = generate_words(model, tokenizer, seed_1, token2idx, idx2token, max_tokens=50, top_p=0.85, device=device)
rprint(f"[bold cyan]Semilla:[/bold cyan] 'the'")
rprint(f"[bold green]Generado:[/bold green] {text_1}\n")

# Ejemplo 2: Semilla con contexto político (presente en inaugural/state_union)
seed_2 = f"{BOS_LABEL} we"
text_2 = generate_words(model, tokenizer, seed_2, token2idx, idx2token, max_tokens=50, top_p=0.9, device=device)
rprint(f"[bold cyan]Semilla:[/bold cyan] 'we'")
rprint(f"[bold green]Generado:[/bold green] {text_2}\n")

# Ejemplo 3: Semilla literaria (presente en gutenberg)
seed_3 = f"{BOS_LABEL} god"
text_3 = generate_words(model, tokenizer, seed_3, token2idx, idx2token, max_tokens=50, top_p=0.8, device=device)
rprint(f"[bold cyan]Semilla:[/bold cyan] 'god'")
rprint(f"[bold green]Generado:[/bold green] {text_3}")

# %%
# Comparación: sub-words crudas vs palabras reconstruidas
model.eval()
seed = f"{BOS_LABEL} the"
words = seed.split()
raw_subwords = []

for _ in range(40):
    idx1 = get_token_id(token2idx, words[-2])
    idx2 = get_token_id(token2idx, words[-1])
    with torch.no_grad():
        log_probs = model(torch.tensor([[idx1, idx2]]).to(device))
    probs = torch.exp(log_probs).squeeze()
    next_idx = torch.multinomial(probs, 1).item()
    next_token = idx2token.get(next_idx, UNK_LABEL)
    if next_token == EOS_LABEL:
        break
    raw_subwords.append(next_token)
    words = [words[-1], next_token]

rprint(f"[bold yellow]Sub-words crudas:[/bold yellow]")
rprint(f"  {raw_subwords}\n")

bpe_ids = [tokenizer.token_to_id(t) for t in raw_subwords if tokenizer.token_to_id(t) is not None]
rprint(f"[bold green]Texto reconstruido:[/bold green]")
rprint(f"  {tokenizer.decode(bpe_ids)}")

# %%
# Verificación: cargar el modelo exportado desde cero
with open(os.path.join(EXPORT_DIR, "config.json")) as f:
    loaded_config = json.load(f)

loaded_model = TrigramModel(
    vocab_size=loaded_config["vocab_size"],
    embedding_dim=loaded_config["embedding_dim"],
    context_size=loaded_config["context_size"],
    hidden_size=loaded_config["hidden_size"],
)
loaded_model.load_state_dict(
    torch.load(os.path.join(EXPORT_DIR, "model_weights.pt"), weights_only=True)
)
loaded_model.eval()

# Verificar que la perplejidad es idéntica
ppl_check, _, _ = calculate_perplexity(loaded_model, test_bpe, token2idx, "cpu")
rprint(f"[bold]Perplejidad modelo cargado: {ppl_check:.2f}[/bold] (debe coincidir con {ppl:.2f})")
