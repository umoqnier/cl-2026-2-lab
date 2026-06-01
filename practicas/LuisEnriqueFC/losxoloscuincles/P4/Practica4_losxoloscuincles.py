# %%
# !pip install transformers torch nltk


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nltk
import math
from transformers import GPT2Tokenizer
from collections import Counter


nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 10

CONTEXT_WORD = 2
CONTEXT_BPE = 5


# %% [markdown]
# ### Preprocesamiento y Partición del Corpus
# Filtrado de la señal de entrada para eliminar ruido (caracteres no alfanuméricos) y normalización a minúsculas. Se garantiza la separación ortogonal entre el conjunto de entrenamiento y el corpus de evaluación (`genesis`).

# %%
def clean_corpus(words_list):
    return [w.lower() for w in words_list if w.isalpha()]

# Conjunto de entrenamiento
raw_train = nltk.corpus.gutenberg.words() + nltk.corpus.inaugural.words()
train_data = clean_corpus(raw_train)

# Conjunto de prueba
raw_test = nltk.corpus.genesis.words()
test_data = clean_corpus(raw_test)

print(f"Tokens de entrenamiento: {len(train_data)}")
print(f"Tokens de prueba: {len(test_data)}")

# %% [markdown]
# ### Mapeo de Vocabulario y Tokenización
# Construcción de funciones biyectivas para el modelo basado en palabras (truncado por frecuencia) y carga del tokenizador BPE (Byte-Pair Encoding) para el modelo subword.

# %%
# Modelo Base (WORD)
VOCAB_SIZE = 10000
counts = Counter(train_data)
vocab_base = ["<UNK>"] + [w for w, c in counts.most_common(VOCAB_SIZE - 1)]

word2idx = {word: i for i, word in enumerate(vocab_base)}
idx2word = {i: word for i, word in enumerate(vocab_base)}

def encode_word_sequence(sequence):
    return [word2idx.get(w, word2idx['<UNK>']) for w in sequence]

# Modelo Subword (BPE)
tokenizer_subword = GPT2Tokenizer.from_pretrained("gpt2")

print(f"Tamaño Vocabulario Base: {len(vocab_base)}")
print(f"Tamaño Vocabulario Subword: {tokenizer_subword.vocab_size}")


# %% [markdown]
# ### Generación de Tensores de Contexto
# Deslizamiento de la ventana de contexto sobre las secuencias codificadas para generar los pares de entrada y objetivo $(X, Y)$ necesarios para la optimización de los tensores.

# %%
def create_ngram_tensors(encoded_sequence, context_size):
    ngrams = []
    for i in range(len(encoded_sequence) - context_size):
        context = encoded_sequence[i : i + context_size]
        target = encoded_sequence[i + context_size]
        ngrams.append((context, target))

    X = torch.tensor([n[0] for n in ngrams], dtype=torch.long)
    Y = torch.tensor([n[1] for n in ngrams], dtype=torch.long)
    return X, Y

# 1. Tensores para el modelo base (WORD)
train_encoded_words = encode_word_sequence(train_data)
X_train_word, Y_train_word = create_ngram_tensors(train_encoded_words, CONTEXT_WORD)

# 2. Tensores para el modelo Subword (BPE)
train_text_continuous = " ".join(train_data)
train_encoded_bpe = tokenizer_subword.encode(train_text_continuous)
X_train_bpe, Y_train_bpe = create_ngram_tensors(train_encoded_bpe, CONTEXT_BPE)

print(f"Tensores Word - X: {X_train_word.shape}, Y: {Y_train_word.shape}")
print(f"Tensores BPE  - X: {X_train_bpe.shape}, Y: {Y_train_bpe.shape}")


# %% [markdown]
# ### Arquitectura de la Red Neuronal (Feed-Forward)
# Implementación de la arquitectura probabilística de lenguaje basada en Bengio et al. (2003). La red consta de:
# 1. **Capa de Proyección (Embedding):** Transformación biyectiva de índices discretos a un espacio continuo $\mathbb{R}^d$.
# 2. **Capa Oculta:** Transformación afín seguida de una no-linealidad $\tanh$.
# 3. **Capa de Salida:** Proyección al espacio dimensional del vocabulario $|V|$.
#
# Se omitirá la aplicación de una función escalar softmax en el último paso *forward*. La propagación retornará logits puros, delegando el cálculo de la probabilidad logarítmica a la función de pérdida `CrossEntropyLoss` para optimizar la estabilidad numérica computacional.

# %%
class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(NNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Capa oculta
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        # Capa de salida
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # Extraemos embeddings y aplanamos
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = torch.tanh(self.linear1(embeds))
        return self.linear2(out)


# Instancia del Modelo base
model_word = NNLM(
    vocab_size=len(vocab_base),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    context_size=CONTEXT_WORD
).to(device)

# Instancia del Modelo subword
model_bpe = NNLM(
    vocab_size=tokenizer_subword.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    context_size=CONTEXT_BPE
).to(device)

print(f"Modelo Base asignado a la memoria del dispositivo.")
print(f"Modelo Subword asignado a la memoria del dispositivo.")

# %% [markdown]
# ### Bucle de Optimización Numérica (Training Loop)
# Implementación del algoritmo de retropropagación (*backpropagation*).
# Se utiliza el optimizador estocástico **Adam** para la actualización de los pesos y **CrossEntropyLoss** como función de costo. Esta última calcula implícitamente el logaritmo probabilístico ($LogSoftmax$) sobre los *logits* crudos devueltos por la red neuronal, garantizando la máxima estabilidad numérica durante el entrenamiento. Se implementa *Mini-batching* para optimizar la ocupación de memoria en hardware acelerado.

# %%
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 256

def train_model(model, X_train, Y_train, epochs, learning_rate=LEARNING_RATE):
    # Definimos la pérdida y el optimizador Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        # Bucle de entrenamiento principal
        for context, target in loader:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Época {epoch+1}/{epochs} | Pérdida: {avg_loss:.4f}")

    return model

print("INICIANDO OPTIMIZACIÓN: MODELO BASE")
model_word = train_model(model_word, X_train_word, Y_train_word, EPOCHS)

print("\n INICIANDO OPTIMIZACIÓN: MODELO SUBWORD")
model_bpe = train_model(model_bpe, X_train_bpe, Y_train_bpe, EPOCHS)


# %% [markdown]
# ### Evaluación Estricta: OOV Rate y Perplejidad
# En esta fase sometemos a los modelos entrenados a una prueba con datos nunca vistos (el corpus `genesis`).
# 1. **Tasa OOV (Out-Of-Vocabulary):** Calculada sobre la secuencia total de tokens para reflejar la cobertura real del vocabulario.
# 2. **Perplejidad ($PPL$):** Medida de la incertidumbre del modelo. Se define matemáticamente como la exponencial de la entropía cruzada promedio:
# $$PPL = \exp\left( \frac{1}{N} \sum_{i=1}^{N} -\ln P(w_i | contexto) \right)$$

# %%
def calculate_oov_rate(test_sequence, train_vocab_set):
    """Calcula la tasa OOV de forma rigurosa sobre el total de tokens."""
    if len(test_sequence) == 0: return 0
    oov_count = sum(1 for token in test_sequence if token not in train_vocab_set)
    return oov_count / len(test_sequence)

def evaluate_perplexity(model, X_test, Y_test):
    """Calcula la perplejidad del modelo en el conjunto de prueba."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # Procesamos por lotes para no saturar la memoria en el test
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    total_loss = 0
    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)
            logits = model(context)
            loss = criterion(logits, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return math.exp(avg_loss)

test_encoded_word = encode_word_sequence(test_data)
X_test_word, Y_test_word = create_ngram_tensors(test_encoded_word, CONTEXT_WORD)


test_text_continuous = " ".join(test_data)
test_encoded_bpe = tokenizer_subword.encode(test_text_continuous)
X_test_bpe, Y_test_bpe = create_ngram_tensors(test_encoded_bpe, CONTEXT_BPE)


# Métricas Modelo Base
oov_word = calculate_oov_rate(test_data, set(vocab_base))
ppl_word = evaluate_perplexity(model_word, X_test_word, Y_test_word)

# Métricas Modelo Subword
vocab_bpe_set = set(tokenizer_subword.get_vocab().keys())
oov_bpe = calculate_oov_rate(test_data, vocab_bpe_set) # Evaluamos cobertura de palabras
ppl_bpe = evaluate_perplexity(model_bpe, X_test_bpe, Y_test_bpe)

print(f"RESULTADOS: MODELO BASE")
print(f"Perplejidad: {ppl_word:.2f} | OOV Rate: {oov_word*100:.2f}%")

print(f"\n RESULTADOS: MODELO SUBWORD")
print(f"Perplejidad: {ppl_bpe:.2f} | OOV Rate: {oov_bpe*100:.2f}%")

# %% [markdown]
# ### Generación Estocástica de Texto
# Prueba cualitativa del poder predictivo de los modelos. Para la generación iterativa de secuencias, se implementa una técnica de **muestreo con temperatura** ($\tau = 0.8$).
#
# En lugar de utilizar una selección determinista (siempre elegir la probabilidad máxima mediante `argmax`, lo que provoca bucles infinitos de texto), la temperatura escala los *logits* antes de aplicar la función *softmax*. Esto permite al modelo transitar estocásticamente hacia palabras menos probables pero gramaticalmente viables, emulando la varianza natural del lenguaje humano.
#
# Posterior a ello se guarda el modelo generado.

# %%
import torch.nn.functional as F

def generate_text(model, initial_words, num_words, context_size, is_bpe=False, tau=0.8):
    """Genera una secuencia de texto iterativamente escalando logits por temperatura."""
    model.eval()

    if is_bpe:
        initial_text = " ".join(initial_words)
        context = tokenizer_subword.encode(initial_text)[-context_size:]
    else:
        context = encode_word_sequence(initial_words)[-context_size:]

    # Validación de dimensiones
    if len(context) < context_size:
        return f"Error: La semilla debe tener al menos {context_size} tokens."

    generated_indices = context.copy()

    with torch.no_grad():
        for _ in range(num_words):
            # Aseguramos que el tensor de entrada mantenga la dimensión del
            # contexto
            x = torch.tensor([context[-context_size:]], dtype=torch.long).to(device)
            logits = model(x)[0]

            # Escalamiento probabilístico
            scaled_logits = logits / tau
            probs = F.softmax(scaled_logits, dim=0)

            # Muestreo desde la distribución
            next_idx = torch.multinomial(probs, 1).item()

            generated_indices.append(next_idx)
            context.append(next_idx)

    # Convertir índices a texto
    if is_bpe:
        return tokenizer_subword.decode(generated_indices)
    else:
        return " ".join([idx2word.get(idx, "<UNK>") for idx in generated_indices])


semilla = ["and", "he", "said", "unto", "them", "that"]
NUM_TOKENS_A_GENERAR = 40

print("Texto Generado: MODELO BASE")
print(generate_text(model_word, semilla, NUM_TOKENS_A_GENERAR, CONTEXT_WORD, is_bpe=False))

print("\n Texto Generado: MODELO SUBWORD")
print(generate_text(model_bpe, semilla, NUM_TOKENS_A_GENERAR, CONTEXT_BPE, is_bpe=True))

# %%
torch.save(model_word.state_dict(), 'modelo_word_base.pt')
torch.save(model_bpe.state_dict(), 'modelo_bpe_subword.pt')
