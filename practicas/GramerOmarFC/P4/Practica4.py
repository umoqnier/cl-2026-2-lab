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
#
# > **Integrantes:** *Omar Fernando Gramer Muñoz*  
# > **Fecha de entrega:** 5 de Mayo 2026
# ---
#
# ## 📌 Objetivos de esta práctica
#
# 1. **Investigar** qué es la *perplejidad* (perplexity) y cómo se calcula.  
# 2. **Evaluar** el modelo neuronal trigrama (entrenado en clase) sobre el corpus `genesis` de NLTK, calculando su perplejidad, tamaño de vocabulario y tasa de OOV.  
# 3. **Entrenar** un nuevo modelo neuronal trigrama sobre los mismos corpus de NLTK, pero después de aplicar **tokenización sub‑palabra (BPE)**.  
# 4. **Comparar** ambos modelos mediante una tabla y analizar sus ventajas.  
# 5. **(Extra)** Diseñar una estrategia de generación de texto con el modelo sub‑palabra que produzca secuencias de palabras (no sub‑palabras) y mostrar ejemplos.

# %% [markdown]
# ## **1. Investigación: Perplejidad (Perplexity)**
#
# ### ¿Qué es la perplejidad?
#
# La **perplejidad** es una métrica muy utilizada para evaluar modelos del lenguaje. Mide qué tan "sorprendido" está el modelo ante un texto de prueba. Cuanto **menor** es la perplejidad, **mejor** es el modelo (porque predice mejor las palabras).
#
# ### Definición matemática
#
# Se define como la exponencial de la entropía cruzada promedio sobre el corpus de prueba:
#
# $$
# \text{Perplexidad}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid \text{contexto}) \right)
# $$
#
# En la práctica, cuando entrenamos minimizando la **Negative Log‑Likelihood (NLL)** con logaritmo natural, la perplejidad se calcula como:
#
# $$
# \text{PPL} = \exp\left( \frac{1}{M} \sum_{j=1}^{M} \text{NLL}(s_j) \right)
# $$
#
# donde:
# - $M$ es el número total de palabras (o trigramas) en el corpus de prueba.
# - $\text{NLL}(s_j) = -\log P(w_j \mid \text{contexto})$ es la pérdida para la palabra $w_j$.
#
# ### Interpretación
#
# - **Perplejidad = 10**: el modelo está tan "confundido" como si tuviera que elegir entre 10 palabras igualmente probables.
# - **Perplejidad = 100**: mucho más incierto, peor predicción.
#
# ### Relación con la calidad del modelo
#
# Un modelo que asigna probabilidades altas a las palabras reales tendrá una pérdida baja y, por lo tanto, una perplejidad baja. Es una medida indirecta de qué tan bien el modelo ha aprendido la distribución del lenguaje.
#
# ### Ventajas
#
# - Fácil de calcular a partir de la función de pérdida.
# - Permite comparar diferentes modelos si se evalúan sobre el mismo conjunto de prueba.
# - No depende del tamaño del corpus (está normalizada).
#
# ### Limitaciones
#
# - No captura directamente la coherencia semántica ni la fluidez del texto generado.
# - Puede favorecer modelos que asignan probabilidad muy alta a palabras frecuentes pero que no generan frases naturales.
# - No es directamente interpretable como una métrica de calidad humana.

# %% [markdown]
# ## **2. Preparación del entorno**
#
# Comenzamos instalando las librerías que no vienen por defecto y luego importamos todo lo necesario.

# %%
# Instalar paquetes requeridos
# !pip install subword-nmt
# !pip install torch
# !pip install nltk
# !pip install rich   # opcional, para mejor visualización

# %% [markdown]
# Ahora importamos los módulos. Algunos son estándar (`os`, `re`, `math`), otros son específicos para NLP (`nltk`, `torch`) y para manejo de datos.

# %%
# Imports y configuración inicial
import os
import re
import math
import random
import unicodedata
import subprocess
from collections import Counter
import torch.optim as optim
from tqdm.notebook import tqdm

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Descargar los recursos de NLTK que usaremos (corpus y tokenizador)
nltk.download('gutenberg')
nltk.download('abc')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('state_union')
nltk.download('webtext')
nltk.download('punkt_tab')

# Para mostrar resultados bonitos (opcional)
from rich import print as rprint

# Detectar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Crear carpetas para guardar modelos si no existen
BASE_PATH = "."
MODELS_PATH = os.path.join(BASE_PATH, "models")
NN_MODELS_PATH = os.path.join(MODELS_PATH, "nn")
os.makedirs(NN_MODELS_PATH, exist_ok=True)

print("Entorno listo.")

# %% [markdown]
# ## **3. Carga y preprocesamiento de los corpus NLTK**
#
# Cargamos los corpus de entrenamiento (varios) y el de prueba (genesis). Luego aplicamos el preprocesamiento estándar: minúsculas, tokens especiales `<BOS>` y `<EOS>`.

# %%
# Funciones de preprocesamiento para el modelo del lenguaje
UNK_LABEL = "<UNK>"

def preprocess_sentences(sentences):
    """
    Toma una lista de oraciones (cada oración es lista de palabras)
    y devuelve la misma lista pero con:
      - todas las palabras en minúsculas
      - añadido <BOS> al inicio
      - añadido <EOS> al final
    """
    processed = []
    for sent in sentences:
        lower_sent = [w.lower() for w in sent]
        processed.append(["<BOS>"] + lower_sent + ["<EOS>"])
    return processed


# %%
# Cargar los corpus
from nltk.corpus import abc, gutenberg, inaugural, state_union, webtext, genesis

# Corpus de entrenamiento
train_corpora = [abc, gutenberg, inaugural, state_union, webtext]
train_sents_raw = []
for corp in train_corpora:
    train_sents_raw.extend(corp.sents())

# Corpus de prueba (genesis)
test_sents_raw = genesis.sents()

print(f"Oraciones de entrenamiento (sin preprocesar): {len(train_sents_raw)}")
print(f"Oraciones de prueba (sin preprocesar): {len(test_sents_raw)}")

# %%
# Aplicar preprocesamiento
train_sents = preprocess_sentences(train_sents_raw)
test_sents = preprocess_sentences(test_sents_raw)

print("Ejemplo de oración preprocesada:")
print(train_sents[0][:15])   # mostramos los primeros 15 tokens


# %%
# Construir vocabulario a partir del entrenamiento
def build_vocab(sentences):
    """
    Construye diccionarios word2id e id2word.
    Palabras con frecuencia 1 se mapean a <UNK>.
    """
    freq = {}
    for sent in sentences:
        for w in sent:
            freq[w] = freq.get(w, 0) + 1
    
    word2id = {UNK_LABEL: 0}
    id2word = {0: UNK_LABEL}
    idx = 1
    for w, f in freq.items():
        if f > 1:   # ignoramos hapax legomena
            word2id[w] = idx
            id2word[idx] = w
            idx += 1
    return word2id, id2word

word2id, id2word = build_vocab(train_sents)
vocab_size = len(word2id)
print(f"Tamaño del vocabulario (word-level): {vocab_size}")


# %%
# Función para obtener ID de una palabra (con <UNK> si no existe)
def get_word_id(word):
    return word2id.get(word, word2id[UNK_LABEL])


# %% [markdown]
# ## **4. Definición del modelo trigrama (arquitectura de Bengio)**
#
# Implementamos la misma arquitectura neuronal utilizada en clase.

# %%
# Definimos la clase TrigramModel
class TrigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: (batch_size, context_size)
        embeds = self.embeddings(inputs)                  # (batch, context, emb_dim)
        embeds = embeds.view(-1, self.context_size * self.embedding_dim)
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# %%
# Parámetros (igual que en clase)
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
HIDDEN_DIM = 100

model_word = TrigramModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM).to(device)
print(model_word)

# %% [markdown]
# ## **5. Modelo word‑level**
#
# Vamos a **entrenar nuestro propio modelo trigrama** sobre un subconjunto de los corpus de NLTK. Esto nos permitirá:
#
# - Comprender el proceso completo de entrenamiento.
# - Controlar los parámetros y la cantidad de datos.
# - Obtener un modelo funcional para calcular su perplejidad y la tasa de OOV.
#

# %% [markdown]
# ### Preparación de los datos de entrenamiento
#
# Usaremos solo las primeras **5,000 oraciones** del corpus de entrenamiento para que el entrenamiento sea rápido (unos minutos con GPU). Esto es suficiente para que el modelo aprenda patrones básicos del lenguaje y podamos comparar con el modelo sub‑palabra.

# %%
# Usamos un subconjunto pequeño de datos para un entrenamiento rápido
small_train_sents = train_sents[:5000]   # solo las primeras 5,000 oraciones
print(f"📊 Usando {len(small_train_sents)} oraciones para entrenar")
print(f"📝 Ejemplo: {small_train_sents[0][:15]}")

# %% [markdown]
# ### Entrenamiento del modelo (código en la celda siguiente)
#
# El modelo tiene la misma arquitectura trigrama (embedding, capa oculta, softmax) que en clase:  
# - `embedding_dim = 200`  
# - `hidden_dim = 100`  
# - `batch_size = 512`  
# - `epochs = 3`
#
# Después del entrenamiento, guardaremos el modelo en la carpeta `models/nn/` por si queremos reutilizarlo más adelante.

# %%
# Parámetros del modelo
EMBEDDING_DIM = 200     
CONTEXT_SIZE = 2
HIDDEN_DIM = 100         
BATCH_SIZE = 512
EPOCHS = 3


# %%
# Función para preparar los trigramas
def prepare_trigrams(sentences):
    trigrams = []
    for sent in sentences:
        for i in range(len(sent) - 2):
            w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
            ctx = [get_word_id(w1), get_word_id(w2)]
            target = get_word_id(w3)
            trigrams.append((ctx, target))
    return trigrams

# Preparamos los datos de entrenamiento
train_trigrams = prepare_trigrams(small_train_sents)
print(f"Trigramas generados: {len(train_trigrams)}")


# Creamos el modelo
model_word = TrigramModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM).to(device)
print(model_word)

# Preparación para el entrenamiento con DataLoader
train_data = np.array([ctx + [tgt] for ctx, tgt in train_trigrams])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Configuramos la función de pérdida y el optimizador
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model_word.parameters(), lr=0.003)

# Iniciamos el entrenamiento
EPOCHS = 3
model_word.train()

for epoch in range(EPOCHS):
    total_loss = 0
    # Barra de progreso para los batches de esta época
    loop = tqdm(train_loader, desc=f"Época word‑level {epoch+1}/{EPOCHS}", unit="batch")
    for batch in loop:
        context = batch[:, :2].to(device)
        target = batch[:, 2].to(device)
        
        optimizer.zero_grad()
        log_probs = model_word(context)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Actualiza la barra con la pérdida actual
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    print(f"✅ Época {epoch+1}/{EPOCHS} completada. Pérdida promedio: {avg_loss:.4f}\n")

print("🎉 Modelo word‑level entrenado exitosamente.")


# %% [markdown]
# ### Evaluación del modelo word‑level entrenado
#
# Una vez que el modelo ha sido entrenado (aunque sea con un subconjunto de datos), podemos medir su calidad sobre un conjunto de **prueba** que no ha visto nunca: el corpus `genesis` de la Biblia.
#
# Las dos métricas principales que calcularemos son:
#
# 1. **Perplejidad (Perplexity)**  
#    Mide qué tan "sorprendido" está el modelo ante el texto de prueba.  
#    **A menor perplejidad, mejor**.
#
# 2. **Tasa de palabras desconocidas (OOV rate)**  
#    Porcentaje de palabras en `genesis` que **no aparecen** en el vocabulario de entrenamiento.  
#    Esto nos ayuda a entender cuánto afecta el vocabulario limitado al rendimiento del modelo.
#
# **Nota importante**: Como nuestro modelo fue entrenado con solo 5,000 oraciones, es probable que la perplejidad sea alta (miles) y la tasa OOV también significativa. Esto es normal y servirá como baseline para comparar con el modelo BPE, que debería reducir drásticamente los OOV.

# %%
# Calcular perplejidad y OOV para el modelo word-level

# Función para obtener el ID de una palabra (con <UNK>)
def get_word_id(word):
    return word2id.get(word, word2id[UNK_LABEL])

# Función que calcula la perplejidad a partir de una lista de oraciones
def compute_perplexity(model, sentences, batch_size=256):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    # Generar todos los trigramas (contexto de 2 palabras -> palabra objetivo)
    trigrams = []
    for sent in sentences:
        for i in range(len(sent) - 2):
            w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
            ctx = [get_word_id(w1), get_word_id(w2)]
            target = get_word_id(w3)
            trigrams.append((ctx, target))
    
    with torch.no_grad():
        for i in range(0, len(trigrams), batch_size):
            batch = trigrams[i:i+batch_size]
            ctx_batch = torch.tensor([p[0] for p in batch], device=device)
            tgt_batch = torch.tensor([p[1] for p in batch], device=device)
            log_probs = model(ctx_batch)
            loss = F.nll_loss(log_probs, tgt_batch, reduction='sum')
            total_nll += loss.item()
            total_tokens += len(batch)
    
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, total_nll, total_tokens

# Calcular perplejidad sobre el corpus de prueba (genesis)
ppl_word, nll_word, tokens_word = compute_perplexity(model_word, test_sents)
print(f"📊 Resultados del modelo word‑level (entrenado con 5000 oraciones):")
print(f"   - Perplejidad: {ppl_word:.2f}")
print(f"   - NLL total: {nll_word:.2f}")
print(f"   - Número de trigramas evaluados: {tokens_word}")

# %%
# Calcular tasa OOV para el modelo word-level
train_vocab = set(word2id.keys())
oov_count = 0
total_words = 0

for sent in test_sents:
    for word in sent:
        total_words += 1
        if word not in train_vocab:
            oov_count += 1

oov_rate = oov_count / total_words if total_words > 0 else 0
print(f"🔤 Tasa OOV (modelo word‑level): {oov_rate:.4%}")
print(f"   Palabras OOV: {oov_count} de {total_words} totales")

# %% [markdown]
# ## **6. Modelo del lenguaje con tokenización sub‑palabra (BPE)**
#
# El principal problema del modelo anterior es que muchas palabras en `genesis` no aparecen en nuestro vocabulario de entrenamiento (OOV). Para mitigarlo, usaremos **Byte Pair Encoding (BPE)**, que divide las palabras en sub‑palabras frecuentes.
#
# **Ventajas de BPE**:
# - Vocabulario más pequeño y manejable.
# - Mejor generalización.
#
# **Proceso**:
# 1. Guardamos nuestro corpus de entrenamiento (las mismas 5000 oraciones) en un archivo de texto plano (una línea por oración).
# 2. Aprendemos un modelo BPE con 2000 operaciones de fusión.
# 3. Tokenizamos tanto el corpus de entrenamiento como el de prueba (`genesis`) con ese modelo BPE.
# 4. Construimos un nuevo vocabulario a partir de las sub‑palabras resultantes.
# 5. Entrenamos un modelo trigrama idéntico al anterior, pero sobre secuencias de sub‑palabras.
# 6. Evaluamos la perplejidad y la tasa OOV (que debería ser casi 0).

# %%
# Guardar corpus de entrenamiento (5000 oraciones) en un archivo de texto plano
train_plain_path = "train_5000_plain.txt"
with open(train_plain_path, 'w', encoding='utf-8') as f:
    for sent in small_train_sents:   # small_train_sents es la variable que usamos antes (5000 oraciones)
        f.write(' '.join(sent) + '\n')
print(f"Corpus de entrenamiento guardado en {train_plain_path}")

# También guardamos el corpus de prueba (genesis) en plano
test_plain_path = "test_genesis_plain.txt"
with open(test_plain_path, 'w', encoding='utf-8') as f:
    for sent in test_sents:
        f.write(' '.join(sent) + '\n')
print(f"Corpus de prueba guardado en {test_plain_path}")

# %%
# Aprender BPE con 2000 merges (operaciones de fusión)
bpe_model_path = "bpe_2000.model"
cmd_learn = f"subword-nmt learn-bpe -s 2000 < {train_plain_path} > {bpe_model_path}"
print("Ejecutando:", cmd_learn)
subprocess.run(cmd_learn, shell=True, check=True)
print(f"Modelo BPE guardado en {bpe_model_path}")

# %% [markdown]
# ### Visualización de la tokenización BPE
#
# Para entender qué ha hecho BPE, compararemos el texto original (palabras separadas por espacios) con el mismo texto después de aplicar BPE. Veremos cómo las palabras raras o compuestas se dividen en sub‑palabras, marcadas con `@@` cuando no son el final de una palabra.

# %%
# Mostrar las primeras 3 oraciones del corpus de entrenamiento original (sin BPE)
print("="*60)
print("TEXTO ORIGINAL (primeras 3 oraciones)")
print("="*60)
for i, sent in enumerate(small_train_sents[:3]):
    print(f"Oración {i+1}: {' '.join(sent)}")
    print()

# Mostrar las mismas oraciones después de aplicar BPE
print("="*60)
print("MISMO TEXTO DESPUÉS DE BPE (sub‑palabras)")
print("="*60)
with open(train_bpe_path, 'r', encoding='utf-8') as f:
    bpe_lines = f.readlines()
    for i in range(min(3, len(bpe_lines))):
        print(f"Oración {i+1}: {bpe_lines[i].strip()}")
        print()


# %% [markdown]
# ### Entrenamiento del modelo trigrama con sub‑palabras (BPE)
#
# Usaremos la misma arquitectura que antes, pero ahora los datos de entrada son los índices de las sub‑palabras. El proceso es idéntico al del modelo word‑level, pero con un vocabulario diferente (probablemente más pequeño) y secuencias más largas.
#
# **Hiperparámetros** (iguales para comparación justa):
# - `EMBEDDING_DIM = 200`
# - `HIDDEN_DIM = 100`
# - `BATCH_SIZE = 512`
# - `EPOCHS = 3`
# - `learning_rate = 0.003`

# %%
# Cargar corpus tokenizados con BPE
def load_bpe_corpus(filepath):
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                corpus.append(tokens)
    return corpus

train_bpe_sents = load_bpe_corpus("train_5000_bpe.txt")
test_bpe_sents = load_bpe_corpus("test_genesis_bpe.txt")

print(f"Oraciones BPE entrenamiento: {len(train_bpe_sents)}")
print(f"Oraciones BPE prueba: {len(test_bpe_sents)}")
print("Ejemplo de sub‑palabras (primeros 10 tokens de la primera oración):")
print(train_bpe_sents[0][:10])

# %%
# Construir vocabulario BPE a partir de train_bpe_sents
word2id_bpe, id2word_bpe = build_vocab(train_bpe_sents)
vocab_size_bpe = len(word2id_bpe)
print(f"Tamaño vocabulario BPE: {vocab_size_bpe}")

def get_word_id_bpe(word):
    return word2id_bpe.get(word, word2id_bpe[UNK_LABEL])


# %%
# Ya tenemos las funciones definidas, pero las reutilizamos
print("Generando trigramas a partir de sub‑palabras...")

# Preparar trigramas BPE (similar a antes, pero usando get_word_id_bpe)
def prepare_trigrams_bpe(sentences):
    trigrams = []
    for sent in sentences:
        for i in range(len(sent) - 2):
            w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
            ctx = [get_word_id_bpe(w1), get_word_id_bpe(w2)]
            target = get_word_id_bpe(w3)
            trigrams.append((ctx, target))
    return trigrams

train_trigrams_bpe = prepare_trigrams_bpe(train_bpe_sents)
print(f"Trigramas BPE generados: {len(train_trigrams_bpe)}")

# Convertir a numpy array para DataLoader
train_data_bpe = np.array([ctx + [tgt] for ctx, tgt in train_trigrams_bpe])
train_loader_bpe = DataLoader(train_data_bpe, batch_size=BATCH_SIZE, shuffle=True)
print(f"DataLoader listo, batches: {len(train_loader_bpe)}")

# %%
# Crear modelo BPE
model_bpe = TrigramModel(vocab_size_bpe, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM).to(device)
print("Modelo BPE:")
print(model_bpe)

# Calcular número de parámetros
total_params = sum(p.numel() for p in model_bpe.parameters())
print(f"Parámetros totales: {total_params:,}")

# %% [markdown]
# #### Configuración del optimizador y función de pérdida
#
# Usaremos `Adam` con learning rate = 0.003 (igual que en el modelo word‑level) y `NLLLoss` porque nuestro modelo devuelve log‑probabilidades.

# %%
# Configurar optimizador y función de pérdida
optimizer_bpe = optim.Adam(model_bpe.parameters(), lr=0.003)
loss_fn = nn.NLLLoss()

print("Optimizador y loss configurados.")

# %% [markdown]
# Entrenamos durante 3 épocas (igual que el modelo word‑level) y mostramos una barra de progreso con `tqdm` para cada época. La pérdida (loss) debería disminuir con cada época.

# %%
# Entrenamiento
EPOCHS_BPE = 3
model_bpe.train()

print("🚀 Iniciando entrenamiento del modelo BPE...")

for epoch in range(EPOCHS_BPE):
    total_loss = 0
    loop = tqdm(train_loader_bpe, desc=f"Época BPE {epoch+1}/{EPOCHS_BPE}", unit="batch")
    
    for batch in loop:
        context = batch[:, :2].to(device)   # [w1, w2]
        target = batch[:, 2].to(device)     # w3
        
        optimizer_bpe.zero_grad()
        log_probs = model_bpe(context)
        loss = loss_fn(log_probs, target)
        loss.backward()
        optimizer_bpe.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader_bpe)
    print(f"✅ Época {epoch+1}/{EPOCHS_BPE} completada. Pérdida promedio: {avg_loss:.4f}")

print("🎉 Entrenamiento BPE finalizado.")

# %%
## Guardamos los pesos del modelo entrenado para poder reutilizarlo más tarde sin tener que volver a entrenar.
# Guardar modelo BPE
torch.save(model_bpe.state_dict(), os.path.join(NN_MODELS_PATH, "model_bpe_trigram.pt"))
print(f"💾 Modelo BPE guardado en {os.path.join(NN_MODELS_PATH, 'model_bpe_trigram.pt')}")


# %% [markdown]
# ## **7. Evaluación del modelo BPE sobre el corpus de prueba (genesis)**
#
# Ahora calculamos la **perplejidad** y la **tasa OOV** para el modelo BPE, usando el corpus `genesis` tokenizado con el mismo BPE.

# %%
def compute_perplexity_bpe(model, sentences, batch_size=512):
    model.eval()
    # Generar trigramas BPE del corpus de prueba
    trigrams = []
    for sent in sentences:
        for i in range(len(sent) - 2):
            w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
            ctx = [get_word_id_bpe(w1), get_word_id_bpe(w2)]
            target = get_word_id_bpe(w3)
            trigrams.append((ctx, target))
    
    total_nll = 0.0
    with torch.no_grad():
        for i in range(0, len(trigrams), batch_size):
            batch = trigrams[i:i+batch_size]
            ctx_batch = torch.tensor([p[0] for p in batch], device=device)
            tgt_batch = torch.tensor([p[1] for p in batch], device=device)
            log_probs = model(ctx_batch)
            loss = F.nll_loss(log_probs, tgt_batch, reduction='sum')
            total_nll += loss.item()
    
    avg_nll = total_nll / len(trigrams)
    perplexity = math.exp(avg_nll)
    return perplexity, total_nll, len(trigrams)

ppl_bpe, nll_bpe, tokens_bpe = compute_perplexity_bpe(model_bpe, test_bpe_sents)
print(f"📊 Resultados del modelo sub‑palabra (BPE):")
print(f"   - Perplejidad: {ppl_bpe:.2f}")
print(f"   - NLL total: {nll_bpe:.2f}")
print(f"   - Número de trigramas BPE evaluados: {tokens_bpe}")

# %%
# Calcular OOV BPE
train_vocab_bpe = set(word2id_bpe.keys())
oov_bpe = 0
total_subwords = 0
for sent in test_bpe_sents:
    for tok in sent:
        total_subwords += 1
        if tok not in train_vocab_bpe:
            oov_bpe += 1
oov_rate_bpe = oov_bpe / total_subwords if total_subwords > 0 else 0

print(f"🔤 Tasa de OOV (sub‑palabras no vistas en entrenamiento): {oov_rate_bpe:.4%}")
print(f"   - Sub‑palabras OOV: {oov_bpe} de {total_subwords} totales")

# %% [markdown]
# ## **8. Análisis de resultados comparativos**
#
# Reunimos todas las métricas de ambos modelos en una tabla para facilitar la comparación.

# %%
print("\n" + "="*70)
print("COMPARACIÓN DE MODELOS")
print("="*70)
print(f"{'Métrica':<35} {'Word‑level':<20} {'Sub‑palabra (BPE)':<20}")
print("-"*70)
print(f"{'Perplejidad (genesis)':<35} {ppl_word:<20.2f} {ppl_bpe:<20.2f}")
print(f"{'Tamaño del vocabulario':<35} {vocab_size:<20} {vocab_size_bpe:<20}")
print(f"{'Tasa OOV':<35} {oov_rate:<20.4%} {oov_rate_bpe:<20.4%}")

# %% [markdown]
# #### **Perplejidad**
# - **Word-level**: 926.84  
# - **BPE**: 829.31  
#
# La perplejidad del modelo sub-palabra es aproximadamente **un 10.5% menor** (mejor) que la del modelo word-level. Esto significa que el modelo BPE es menos "sorprendido" al predecir la siguiente sub-palabra dentro del corpus `genesis`. Aunque la diferencia ya no es tan extrema como en la versión anterior, BPE sigue mostrando una ventaja clara gracias a su capacidad para descomponer palabras raras o desconocidas en unidades más pequeñas y reutilizables.
#
# #### **Tamaño del vocabulario**
# - **Word-level**: 41,392 palabras  
# - **BPE**: 2,013 sub-palabras  
#
# El vocabulario BPE sigue siendo **más de 20 veces más pequeño** que el vocabulario word-level. Esto representa una ventaja importante en términos de memoria y eficiencia computacional: menos embeddings, menos parámetros y mejor capacidad de generalización. De hecho pudimos notar que el entrenamiento con BPE fue muchisimo más rápido. A pesar de su tamaño reducido, el modelo BPE puede representar prácticamente cualquier palabra mediante combinaciones de sub-palabras frecuentes. 
# #### **Tasa OOV (Out-Of-Vocabulary)**
# - **Word-level**: 36.6312%  
# - **BPE**: 5.5305%  
#
# La diferencia en tasa OOV continúa siendo muy significativa. Más de un tercio de las palabras del corpus `genesis` no aparecen en el vocabulario del modelo word-level, por lo que deben reemplazarse por el token `<UNK>`, provocando pérdida de información y peores predicciones.
#
# Por otro lado, el modelo BPE reduce la tasa OOV a solo **5.53%**, ya que puede segmentar palabras desconocidas en sub-componentes previamente observados durante el entrenamiento. Esto demuestra que los modelos basados en sub-palabras son mucho más robustos frente a vocabularios abiertos y palabras raras.
#
# ### Mejoras posibles para ambos modelos
# - Aumentar el tamaño del corpus de entrenamiento (usar todas las oraciones disponibles de NLTK en lugar de únicamente 5000).
# - Incrementar el número de merges en BPE (por ejemplo, 5000 o 10000) para capturar palabras frecuentes completas.
# - Aplicar suavizado más avanzado en los modelos de lenguaje para reducir el impacto de secuencias poco frecuentes.
# - Entrenar modelos de mayor orden (*n-gramas* más grandes) o utilizar arquitecturas neuronales modernas para mejorar la capacidad predictiva.

# %% [markdown]
# # **Parte Extra: Generación de texto con el modelo BPE**
#
# El modelo BPE genera sub‑palabras, no palabras completas. Para obtener texto legible, debemos **reconstruir palabras** cada vez que encontramos una sub‑palabra que **no termina en `@@`** (marca de continuación). La estrategia:
#
# 1. Partimos de un contexto inicial de dos sub‑palabras (por ejemplo, `["<BOS>", "the"]`).
# 2. Usamos el modelo para predecir la siguiente sub‑palabra, muestreando con **nucleus sampling (top‑p)** para añadir variedad.
# 3. Acumulamos sub‑palabras hasta formar una palabra completa.
# 4. Paramos al llegar a `max_tokens` o al token `<EOS>`.
# 5. Mostramos la frase reconstruida.

# %%
import torch.nn.functional as F
import random


def top_p_sampling(logits, p=0.9):
    """
    Nucleus (top-p) sampling: selecciona el siguiente token
    a partir del conjunto de tokens cuya masa de probabilidad acumulada
    es al menos p.
    """
    # Detach para quitar gradientes, luego exp y pasar a numpy
    probs = torch.exp(logits.detach()).cpu().numpy()[0]
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    mask = cumulative <= p
    # Aseguramos que al menos un token se seleccione
    if not mask.any():
        mask[0] = True
    indices = sorted_indices[mask]
    probs_filtered = sorted_probs[mask]
    probs_filtered = probs_filtered / probs_filtered.sum()
    next_idx = np.random.choice(indices, p=probs_filtered)
    return next_idx

def reconstruct_words(bpe_tokens):
    """
    Convierte una lista de sub‑palabras (con @@ al final si continúa)
    en una lista de palabras completas.
    Ejemplo: ["camin@@", "ando", "feliz"] -> ["caminando", "feliz"]
    """
    words = []
    current = ""
    for tok in bpe_tokens:
        if tok.endswith("@@"):
            current += tok[:-2]
        else:
            current += tok
            words.append(current)
            current = ""
    return words

def generate_text_bpe(model, start_context, max_new_tokens=30, top_p=0.8):
    """
    Genera texto a partir de un contexto inicial de dos sub‑palabras (strings).
    Devuelve la frase completa en palabras.
    """
    model.eval()
    # Convertir contexto a IDs
    ctx_ids = [get_word_id_bpe(w) for w in start_context]
    generated_bpe = start_context.copy()
    
    for _ in range(max_new_tokens):
        ctx_tensor = torch.tensor([ctx_ids[-2:]], device=device)
        log_probs = model(ctx_tensor)
        next_id = top_p_sampling(log_probs, p=top_p)
        next_token = id2word_bpe[next_id]
        generated_bpe.append(next_token)
        ctx_ids.append(next_id)
        if next_token == "<EOS>":
            break
    
    # Reconstruir palabras
    words = reconstruct_words(generated_bpe)
    return " ".join(words)


# %%
# Tres ejemplos con diferentes contextos iniciales
contexts = [
    ["<BOS>", "jesus"],
    ["he", "is"],
    ["my", "lord"],
    ["<BOS>", "they"],
    ["and", "if"],
    ["my", "lord"],
    ["caring", "people"],
    ["on", "heaven"],
]

print("🔮 Generación de texto con modelo BPE (top‑p=0.8):\n")
for ctx in contexts:
    text = generate_text_bpe(model_bpe, ctx, max_new_tokens=35, top_p=0.5)
    print(f"Contexto:[ {', '.join(ctx)}] -> {text}\n")

# %% [markdown]
# ## **Análisis de la generación de texto con el modelo BPE**
#
# Podemos notar ciertas limitaciones, es importante preguntarse el por qué:
#
# 1. **Modelo trigrama**: Solo mira las dos palabras anteriores. No puede mantener coherencia a larga distancia. Por eso las frases cambian de tema abruptamente. 
# 2. **Pocos datos de entrenamiento (5000 oraciones)**: El modelo no ha visto suficientes patrones lingüísticos reales. Palabras como "cobug", "refuive", "agut" son sub‑palabras o palabras malformadas porque el modelo no aprendió bien combinaciones.
#
# 3. **Top‑p = 0.8**: Introduce variedad, pero también ruido. 
#
# 4. **Sin mecanismo de búsqueda (beam search)**: Usamos muestreo aleatorio, que es más creativo pero menos coherente.
#
# ### ¿Cómo podría mejorar la generación?
#
# - **Aumentar los datos de entrenamiento**: Usar todas las oraciones de NLTK (en lugar de solo 5000) mejoraría drásticamente la calidad.
# - **Aumentar el tamaño del modelo**: Más dimensiones de embedding y capa oculta.
# - **Cambiar a un n-grama**: Podriamos utilizar un 5-grama en lugar de trigramas. O usar una red recurrente ( pero eso lo vimos despues )
#
# A pesar de su simplicidad, el modelo BPE **demuestra el principio**: puede generar palabras completas a partir de sub‑palabras y produce frases que, aunque imperfectas, tienen estructura gramatical básica (sujeto‑verbo‑objeto a veces, puntuación, etc.).
#
#

# %% [markdown]
# ## Resumen Final
#
# ### ¿Qué aprendí en esta práctica?
#
# 1. **Perplejidad**: Es la métrica estándar para evaluar modelos de lenguaje.  
#    Mide qué tan "sorprendido" está el modelo ante datos nuevos. Menor = mejor.
#
# 2. **BPE reduce el OOV**: Al tokenizar en sub-palabras, palabras desconocidas  
#    se descomponen en piezas conocidas, reduciendo el porcentaje de OOV.
#
# 3. **Trade-offs**: El modelo BPE tiene más sub-tokens que procesar (más trigramas),  
#    lo que aumenta el tiempo de entrenamiento, pero mejora la cobertura léxica.
#
# 4. **Reconstrucción BPE**: Para generar texto legible con modelos sub-word,  
#    debemos re-ensamblar los sub-tokens en palabras completas.
#
# ---
# *Práctica elaborada con PyTorch, NLTK, Gensim y subword-nmt*
#

# %%
