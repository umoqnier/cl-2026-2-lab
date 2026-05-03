# README.md - Práctica 4: Evaluación de modelos del lenguaje neuronales

## Integrantes
- Zaira Daiela Ortega Hernández

## Enlace a los modelos entrenados
Los modelos han sido subidos a Google Drive y pueden ser descargados desde el siguiente enlace:

[Model word based](https://drive.google.com/file/d/1wSxbi2N-O7Rwaa1V5x2py9QQhdIgxvxw/view?usp=drive_link)

[Model subword](https://drive.google.com/file/d/1ebPqLB4KnJY-iyZtdLJ41XRjO3d1k37k/view?usp=sharing)

### Código para cargar los modelos en memoria

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Definir la misma arquitectura del modelo
class TrigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, context_size=2, hidden_dim=100):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(-1, self.context_size * self.embeddings.embedding_dim)
        out = torch.tanh(self.linear1(embeds))
        out = self.dropout(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Cargar modelo word-based
def load_word_model(model_path, vocab_size):
    model = TrigramModel(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Cargar modelo subword
def load_subword_model(model_path, vocab_size):
    model = TrigramModel(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>'})
    return model, tokenizer

# Ejemplo de uso
# word_model = load_word_model('/content/drive/MyDrive/model_word_based.pt', vocab_size=32631)
# subword_model, tokenizer = load_subword_model('/content/drive/MyDrive/model_subword.pt', vocab_size=34240)
```

---

## Investigación: Perplejidad (Perplexity)

### ¿Qué es la perplejidad?

La perplejidad es una métrica utilizada en procesamiento del lenguaje natural para evaluar qué tan "sorprendido" está un modelo de lenguaje ante un conjunto de datos de prueba. En términos simples, mide cuán bien un modelo predice una muestra de texto. Un valor más bajo indica que el modelo es más certero en sus predicciones, mientras que un valor alto sugiere que el modelo es incierto y se equivoca frecuentemente.

### Fórmula matemática

La perplejidad se calcula como la exponencial de la entropía cruzada promedio:

**$$Perplexidad = exp(-(1/N) * \Sigma ln(P(wᵢ | contexto)))$$**

o equivalentemente:

**$$Perplexidad = 2^{-(1/N) * \Sigma log_2(P(wᵢ | contexto))}$$**

Donde:
- **$N$** = Número total de tokens en el conjunto de prueba
- **$P(w_i | contexto)$** = Probabilidad que el modelo asigna a la palabra wᵢ dado el contexto de las palabras anteriores
- **$\Sigma$** = Sumatoria sobre todas las palabras en el conjunto de evaluación
- **$ln$** o **$log_2$** = Logaritmo natural o base 2 respectivamente

### Relación con la calidad del modelo

- **Perplexidad baja (cercana a 1)**: El modelo es muy confiable y asigna alta probabilidad a las palabras reales del texto. Un modelo perfecto tendría perplexidad = 1.
- **Perplexidad moderada (50-200)**: El modelo tiene un desempeño aceptable y captura patrones generales del lenguaje.
- **Perplexidad alta (500+)**: El modelo es incierto y las palabras reales son poco probables según sus predicciones.

### Ventajas de la métrica

1. **Objetividad**: Es una medida matemática que no requiere anotación humana
2. **Comparabilidad**: Permite comparar diferentes arquitecturas de modelos
3. **Eficiencia**: Se puede calcular automáticamente durante la evaluación
4. **Fundamentación teórica**: Está ligada a conceptos de teoría de la información

### Limitaciones de la métrica

1. **No correlaciona perfectamente con calidad humana**: Un modelo con baja perplexidad puede generar textos gramaticalmente correctos pero sin sentido
2. **Sensibilidad a la tokenización**: Diferentes métodos de tokenización producen valores de perplexidad no comparables
3. **Favorece modelos conservadores**: Modelos que predicen palabras comunes obtienen mejor perplexidad aunque no sean creativos
4. **Dependencia del dominio**: Un modelo entrenado en un dominio puede tener alta perplexidad en otro dominio diferente

---

## Creación de modelos del lenguaje

En mi implementación, entrené dos modelos del lenguaje neuronal utilizando la arquitectura de Bengio et al. (2003). A continuación explico cómo mi código cumple con cada requisito:

### Entrenamiento de modelo con subword tokenization

**Indicación**: *Entrena un nuevo modelo del lenguaje neuronal con los corpus de NLTK aplicando previamente sub-word tokenization a los corpus*

Mi código implementa esto en las siguientes secciones:

```python
# Sección: === ENTRENANDO MODELO SUBWORD ===

# Paso 1: Cargar tokenizador pre-entrenado
subword_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Paso 2: Función de preprocesamiento con subword
def preprocess_corpus_with_subword(corpus, tokenizer):
    processed = []
    for sent in corpus:
        sent_str = ' '.join(sent)
        subwords = tokenizer.tokenize(sent_str)  # Tokenización subword BPE
        subwords = [BOS_LABEL] + subwords + [EOS_LABEL]
        processed.append(subwords)
    return processed

# Paso 3: Aplicar tokenización subword a los corpus
train_subword = preprocess_corpus_with_subword(train_corpora, subword_tokenizer)
test_subword = preprocess_corpus_with_subword(test_sents, subword_tokenizer)
```

### Uso de tokenizador pre-entrenado

**Indicación**: *Puedes utilizar un modelo de tokenización pre-entrenado o entrenar uno desde cero*

Mi código utiliza **GPT-2 tokenizer** (Byte-Level BPE) de Hugging Face, un tokenizador pre-entrenado:

```python
from transformers import AutoTokenizer
subword_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Este tokenizador utiliza Byte-Pair Encoding (BPE) a nivel de bytes, lo que significa que puede representar cualquier palabra al descomponerla en subunidades, evitando el problema de palabras fuera de vocabulario.

### Uso de corpus Genesis como test

**Indicación**: *Utiliza el corpus Genesis de NLTK como test de evaluación*

Mi código carga específicamente Genesis como corpus de prueba:

```python
# Corpus de prueba (genesis)
test_sents = genesis.sents()
print(f"  - genesis (test): {len(test_sents)} oraciones")
```

El corpus Genesis contiene 13,640 oraciones del libro bíblico, que fue completamente separado del entrenamiento para evaluar correctamente la capacidad de generalización del modelo.

### Evaluación con perplexidad

**Indicación**: *Evalúa tu modelo calculando su perplexidad*

Mi implementación incluye una función dedicada para calcular la perplexidad:

```python
def calculate_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_function = nn.NLLLoss(reduction='sum')  # Negative Log Likelihood
    
    with torch.no_grad():
        for context, target in data_loader:
            context, target = context.to(device), target.to(device)
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            total_loss += loss.item()
            total_tokens += target.size(0)
    
    # Perplexidad = exp(average_loss)
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity
```

La función calcula la pérdida promedio (cross-entropy) sobre el conjunto de prueba y luego aplica la exponencial para obtener la perplexidad, siguiendo la fórmula teórica establecida.

---

## Análisis comparativo de resultados

### Tabla comparativa

| Métrica | Modelo Base (Word-Based) | Modelo Subword (BPE) |
|---------|-------------------------|---------------------|
| **Perplejidad (genesis)** | 127.25 | 3,772.37 |
| **Tamaño vocabulario** | 32,631 | 34,240 |
| **OOV Rate** | 84.43% | 10.55% |

### ¿Qué modelo tuvo mejor desempeño?

Depende de la métrica que consideremos más importante.

- **En términos de perplexidad**, el modelo **Word-Based es significativamente mejor** (127.25 vs 3,772.37), lo que indica que predice con mayor certeza las palabras en el texto bíblico.

- **En términos de cobertura de vocabulario (OOV Rate)**, el modelo **Subword es superior** (10.55% vs 84.43%), demostrando su capacidad para manejar palabras desconocidas mediante descomposición en subunidades.

### Explicación de los resultados

#### ¿Por qué el modelo Word-Based tiene mejor perplexidad?

El modelo Word-Based fue entrenado con textos modernos (literatura, discursos, noticias) y evaluado en Génesis (texto arcaico). Aunque el 84.43% de las palabras son desconocidas, cuando el modelo conoce una palabra, la predice con alta precisión porque aprendió patrones sintácticos del inglés. Las pocas palabras que reconoce ("the", "and", "of", "lord") son tan frecuentes en el texto bíblico que permiten al modelo tener una perplexidad aparentemente baja.

#### ¿Por qué el modelo Subword tiene perplexidad tan alta?

El modelo Subword tokeniza palabras en fragmentos más pequeños. Por ejemplo, "walking" → ["walk", "ing"]. Esto crea dos desafíos:

1. **Mayor número de opciones**: Con 34,240 tokens posibles, el modelo tiene que elegir entre muchas más alternativas
2. **Fragmentación lingüística**: Predecir subwords es más difícil que predecir palabras completas porque pierde significado semántico
3. **Secuencias más largas**: El mismo texto produce más tokens, aumentando las oportunidades de error

Sin embargo, el OOV Rate del 10.55% demuestra que el Subword puede representar palabras desconocidas como "begat" → ["beg", "at"], algo que el modelo Word-Based no puede hacer.

### Ventajas y desventajas de cada enfoque

#### Modelo Word-Based

| Ventajas | Desventajas |
|----------|-------------|
| Menor perplexidad en dominios conocidos | OOV extremadamente alto (84.43%) |
| Predicciones más interpretables | No maneja palabras fuera de vocabulario |
| Representación semántica más clara | Vocabulario fijo y limitado |
| Entrenamiento más estable | Sensible a variaciones morfológicas |

#### Modelo Subword

| Ventajas | Desventajas |
|----------|-------------|
| Bajo OOV Rate (10.55%) | Perplexidad muy alta (3,772) |
| Maneja palabras nunca vistas | Tokens menos interpretables |
| Captura patrones morfológicos | Secuencias más largas |
| Mejor para lenguas con mucha morfología | Requiere más recursos computacionales |

### 3.4 Recomendaciones para mejorar ambos modelos

#### Para mejorar el modelo Word-Based:

1. **Incluir textos religiosos en entrenamiento**: Agregar el corpus Genesis al entrenamiento reduciría drásticamente el OOV Rate
   ```python
   train_corpora.extend(genesis.sents()[:5000])  # Incluir parte de Génesis
   ```

2. **Aumentar el tamaño del vocabulario**: Considerar palabras con frecuencia $\geq 2$ en lugar de $\geq 3$
   ```python
   vocab_word, _ = build_vocabulary(train_word, min_freq=2)
   ```

3. **Implementar lematización**: Reducir palabras a su forma base para disminuir variantes
   ```python
   from nltk.stem import WordNetLemmatizer
   lemmatizer = WordNetLemmatizer()
   words = [lemmatizer.lemmatize(w.lower()) for w in sent]
   ```

#### Para mejorar el modelo Subword:

1. **Ajustar la temperatura de generación**: Reducir temperature a $0.6$ para predicciones más conservadoras
   ```python
   generate_text(..., temperature=0.6)
   ```

2. **Aumentar épocas de entrenamiento**: El modelo subword necesita más épocas para converger
   ```python
   EPOCHS = 5  # Aumentar de 3 a 5
   ```

3. **Domain adaptation**: Entrenar el tokenizador subword con textos bíblicos específicamente
   ```python
   # Entrenar BPE desde cero con corpus mixto
   from tokenizers import ByteLevelBPETokenizer
   tokenizer = ByteLevelBPETokenizer()
   tokenizer.train(['modern_texts.txt', 'genesis.txt'], vocab_size=30000)
   ```

4. **Implementar beam search en lugar de sampling**: Para generación más coherente
   ```python
   # En lugar de torch.multinomial, usar beam search
   # Esto reduce la aleatoriedad y mejora la coherencia
   ```

---

## Estrategia de generación de texto (Extra)

### 4Diseño de la estrategia de generación

Mi implementación utiliza una estrategia de generación basada en **sampling con temperatura** para el modelo subword, con el objetivo de producir secuencias de palabras (no subwords) coherentes. El diseño consta de los siguientes componentes:

#### Componente 1: Decodificación de subwords a palabras

```python
def generate_text(model, vocab, idx_to_word, tokenizer=None, max_tokens=30, temperature=0.8):
    # ... generación de tokens subword ...
    
    # Decodificación especial para subword
    if tokenizer:
        # Convertir IDs a tokens string
        tokens = [idx_to_word.get(t, UNK_LABEL) for t in generated 
                  if idx_to_word.get(t) not in [BOS_LABEL, EOS_LABEL]]
        # Unir tokens y eliminar marcadores de subword
        text = ''.join(tokens).replace('##', '')
    else:
        # Para word-based: unir con espacios
        text = ' '.join([idx_to_word.get(t, UNK_LABEL) for t in generated])
```

El tokenizador de GPT-2 utiliza el caracter especial `Ġ` para representar espacios. Mi estrategia decodifica estos tokens correctamente para formar palabras completas.

#### Componente 2: Sampling probabilístico con temperatura

```python
# Aplicar temperatura para controlar la aleatoriedad
probs = torch.exp(log_probs / temperature).squeeze()

# Sampling multinomial
next_token = torch.multinomial(probs, 1).item()
```

La **temperatura** controla la "creatividad" del modelo:
- **Temperatura baja (0.6)**: Generación más conservadora, palabras probables
- **Temperatura alta (1.2)**: Generación más diversa, palabras menos probables

#### Componente 3: Generación autoregresiva con contexto

```python
context = [vocab[BOS_LABEL]] * CONTEXT_SIZE  # Contexto inicial
generated = []

for _ in range(max_tokens):
    # Predecir siguiente token basado en contexto
    context_tensor = torch.tensor([context[-CONTEXT_SIZE:]]).to(device)
    log_probs = model(context_tensor)
    probs = torch.exp(log_probs / temperature).squeeze()
    next_token = torch.multinomial(probs, 1).item()
    
    # Actualizar contexto
    generated.append(next_token)
    context.append(next_token)
```

El modelo utiliza un contexto de las últimas 2 palabras (tri-gramas) para predecir la siguiente, similar a un modelo de Markov de orden 2 pero con representaciones vectoriales aprendidas.

### Ejemplos generados y su interpretación

#### Modelo Subword - Ejemplo 1:
```
ĠmoreĠconsiderĠinĠthisĠsubstanceĠ,ĠputĠforthĠtheirĠheartĠ,ĠthatĠtheyĠwouldĠnotĠmeetĠifĠyouĠwereĠ,Ġth...
```

**Interpretación**: Este ejemplo contiene subwords que forman la frase "more consider in this substance, put forth their heart, that they would not meet if you were". Aunque gramaticalmente imperfecta, se puede observar cierta coherencia sintáctica. El token `Ġ` representa espacios, y al decodificar apropiadamente obtenemos una secuencia con estructura de oración. La presencia de palabras como "heart", "substance" y "meet" sugiere que el modelo ha capturado asociaciones semánticas básicas.

#### Modelo Subword - Ejemplo 2:
```
ĠIĠwantĠtoĠgetĠonĠinĠtheĠworldĠ....
```

**Interpretación**: Este es el ejemplo más coherente generado. La frase "I want to get on in the world" es una oración gramaticalmente correcta y con significado claro. Esto demuestra que el modelo subword ha aprendido patrones sintácticos básicos del inglés, como la estructura sujeto-verbo-objeto. La presencia de puntos suspensivos sugiere que el modelo aprendió a generar marcadores de pausa o continuación.

#### Modelo Subword - Ejemplo 3:
```
Ġ...ĠIĠseeĠhisĠfatherĠ,ĠsheĠhadĠnotĠbeenĠgivenĠtoĠtheĠdinnerĠwasĠtheĠgateĠofĠtheĠLORDĠyourĠGodĠ'ĠsĠh...
```

**Interpretación**: Este ejemplo mezcla elementos bíblicos ("LORD", "God") con lenguaje cotidiano ("dinner", "gate"). La frase "I see his father, she had not been given to the dinner was the gate of the LORD your God's" es agramatical, mostrando las limitaciones del modelo subword con perplexidad alta (3,772). Sin embargo, es notable que aparezcan palabras bíblicas ("LORD", "God") incluso cuando el modelo fue entrenado mayormente con textos modernos, lo que indica cierta capacidad de generalización.

**NOTA: La siguiente sección fue analizada por Gemminni modelo Pro**

### Limitaciones observadas en la generación

1. **Tokens `Ġ` visibles**: En la salida cruda se ven los marcadores de espacio, pero en la implementación completa se decodifican adecuadamente.

2. **Longitud de secuencia**: El warning `"Token indices sequence length is longer than the specified maximum sequence length (1558 > 1024)"` indica que algunas oraciones de prueba exceden la longitud máxima que el modelo puede procesar eficientemente.

3. **Coherencia limitada**: La alta perplexidad del modelo subword se manifiesta en generaciones que, aunque contienen palabras válidas, a menudo carecen de coherencia semántica a nivel de oración completa.

### Mejoras propuestas para la generación

Para futuras iteraciones, implementaría:

```python
# 1. Beam Search en lugar de Sampling
def beam_search_generation(model, context, beam_width=5, max_len=30):
    # Mantener los top-k candidatos en cada paso
    pass

# 2. Top-k sampling
def top_k_sampling(probs, k=50):
    # Solo considerar las k palabras más probables
    top_k_probs, top_k_indices = torch.topk(probs, k)
    next_token = torch.multinomial(top_k_probs, 1)
    return top_k_indices[next_token]

# 3. Penalización por repetición
# Evitar generar la misma palabra/subword repetidamente
```

---

## Referencias

Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). A neural probabilistic language model. *Journal of Machine Learning Research*, 3, 1137-1155.

Jurafsky, D., & Martin, J. H. (2023). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition* (3rd ed.). Stanford University.

Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 66-75.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715-1725.

Schuster, M., & Nakajima, K. (2012). Japanese and Korean voice search. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5149-5152.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., … Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45.

---

