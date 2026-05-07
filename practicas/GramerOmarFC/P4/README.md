# Práctica 4: Evaluación de modelos del lenguaje neuronales

**Equipo:** Pro-Gramer  
**Integrantes:** Omar Fernando Gramer Muñoz 
**Fecha:** 7 de mayo de 2026

---

## Investigación: Perplejidad (Perplexity)

La **perplejidad** es una métrica utilizada para evaluar modelos probabilísticos del lenguaje. Mide qué tan "sorprendido" está el modelo ante un conjunto de prueba: cuanto menor es la perplejidad, mejor es el modelo.

### Definición matemática

Para un modelo que asigna probabilidad \(P(w_i | \text{contexto})\) a cada palabra \(w_i\), la perplejidad se define como:

\[
\text{PPL}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid \text{contexto}) \right)
\]

En la práctica, cuando entrenamos con *Negative Log-Likelihood* (NLL) usando logaritmo natural, la perplejidad se calcula como:

\[
\text{PPL} = \exp\left( \frac{1}{M} \sum_{j=1}^{M} \text{NLL}(s_j) \right)
\]

donde:
- \(M\) es el número total de palabras (o trigramas) en el corpus de prueba.
- \(\text{NLL}(s_j) = -\log P(w_j \mid \text{contexto})\) es la pérdida para la palabra \(w_j\).

### Relación con la calidad del modelo

- **Perplejidad baja** → el modelo asigna probabilidades altas a las palabras reales → mejor capacidad predictiva.
- **Perplejidad alta** → el modelo es incierto, similar a elegir entre muchas opciones equiprobables.

**Ejemplo:** una perplejidad de 10 equivale a que el modelo está tan confundido como si tuviera que adivinar entre 10 palabras igualmente probables.

### Ventajas y limitaciones

| Ventajas | Limitaciones |
|----------|---------------|
| Fácil de calcular a partir de la pérdida | No mide coherencia semántica ni fluidez |
| Permite comparar diferentes modelos | Favorece modelos que predicen palabras frecuentes |
| Independiente del tamaño del corpus (normalizada) | No es directamente interpretable como calidad humana |

---

## Modelos entrenados

Se entrenaron dos modelos trigrama (contexto de 2 palabras anteriores) sobre un subconjunto de los corpus de NLTK (5000 oraciones de entrenamiento de `abc`, `gutenberg`, `inaugural`, `state_union`, `webtext`). El corpus de prueba fue `genesis`.

### 1. Modelo word‑level
- Tokenización a nivel de palabra completa.
- Vocabulario: palabras con frecuencia > 1 (reemplazando palabras raras por `<UNK>`).

### 2. Modelo sub‑palabra (BPE)
- Se aplicó Byte Pair Encoding con 2000 operaciones de fusión.
- El corpus se tokenizó en sub‑palabras (ej. `camin@@` + `ando` → `caminando`).
- Se construyó un vocabulario de sub‑palabras.

---

## Evaluación y comparación

| Métrica               | Modelo word‑level | Modelo sub‑palabra (BPE) |
|-----------------------|------------------|--------------------------|
| **Perplejidad (genesis)** | 1302.76          | 820.13                   |
| **Tamaño del vocabulario** | 41,392           | 2,013                    |
| **Tasa OOV**          | 36.63%           | 5.53%                    |

### Análisis

El modelo **sub‑palabra (BPE)** supera claramente al modelo word‑level en todas las métricas:
- **Perplejidad un 37% menor**: predice mejor la siguiente sub‑palabra porque maneja bien palabras desconocidas.
- **Vocabulario 20 veces más pequeño**: reduce la complejidad del modelo y el riesgo de sobreentrenamiento.
- **Tasa OOV solo del 5.53%** frente al 36.63% del word‑level. Con un corpus de entrenamiento más grande, la tasa OOV de BPE se acercaría a 0%.

**Conclusión:** La tokenización sub‑palabra es esencial para modelos del lenguaje modernos (GPT, BERT, etc.) porque reduce drásticamente el vocabulario y elimina casi por completo el problema de palabras fuera de vocabulario.

### Recomendaciones para mejorar ambos modelos
- Aumentar los datos de entrenamiento (usar todas las oraciones de NLTK, no solo 5000).
- Probar con más merges BPE (ej. 5000 o 10000).
- Ajustar hiperparámetros: tamaño de embedding, capa oculta, learning rate.
- Usar arquitecturas más potentes (LSTM, Transformer).

---

## Modelo entrenado (sub‑palabra)

El modelo sub‑palabra (BPE) se ha subido a Google Drive. Puede descargarse desde el siguiente enlace:

🔗 [Descargar modelo_bpe_trigram.pt](https://drive.google.com/uc?export=download&id=TU_ID_DE_ARCHIVO)

> **Nota:** Reemplaza `TU_ID_DE_ARCHIVO` por el identificador real del archivo en Google Drive.

### Instrucciones de carga en Python

```python
import torch
from tu_modelo import TrigramModel   # asegúrate de tener la clase definida

modelo = TrigramModel(vocab_size_bpe, 100, 2, 50)
modelo.load_state_dict(torch.load("model_bpe_trigram.pt", map_location=torch.device('cpu')))
modelo.eval()
