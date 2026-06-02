# Práctica 4: Perplejidad en Modelos de Lenguaje

**Alumno:** Toporek Coca Eric — **314284987**

---

## Instrucciones de ejecución y uso

### Modelo web
https://huggingface.co/toporaku/complin_p4_neural_lm

### Requisitos previos

- **Python** ≥ 3.10
- Se recomienda usar un entorno virtual (`.venv`)

### Dependencias

Instalar las dependencias necesarias con `pip`:

```bash
pip install torch tokenizers nltk numpy rich jupytext
```

| Paquete | Uso |
|---------|-----|
| `torch` | Framework de redes neuronales (modelo trigrama Bengio) |
| `tokenizers` | Tokenizador BPE de Hugging Face |
| `nltk` | Corpus de texto (gutenberg, abc, genesis, etc.) |
| `numpy` | Manipulación de arreglos numéricos |
| `rich` | Salida con formato enriquecido en terminal |
| `jupytext` | Sincronización entre `.py` y `.ipynb` |

### Corpus de NLTK

La primera celda del notebook descarga automáticamente los corpus necesarios:

```
gutenberg, abc, genesis, inaugural, state_union, webtext, punkt_tab
```

### Ejecución

1. **Abrir el notebook** `p4_nerual_lm.ipynb` en Jupyter o VS Code
2. **Seleccionar el kernel** del entorno virtual (`.venv`)
3. **Ejecutar todas las celdas** en orden secuencial (`Run All`)

> **Nota:** El entrenamiento de los modelos puede tardar varios minutos dependiendo del hardware. Si se dispone de GPU/MPS, PyTorch la detectará automáticamente.

### Estructura del proyecto

```
P4/
├── p4_nerual_lm.ipynb   # Notebook principal (ejecutable)
├── p4_nerual_lm.py      # Código fuente pareado (jupytext, percent format)
├── README.md             # Este archivo
└── models/               # (generado al ejecutar) Modelos y artefactos
    ├── bpe_tokenizer.json
    ├── nn/
    └── export/
```

### Sincronización jupytext

El archivo `.py` está pareado con el `.ipynb` mediante jupytext. Para sincronizar cambios:

```bash
python -m jupytext --sync p4_nerual_lm.py
```

---

## Uso de LLMs: 5

Se utilizaron modelos de lenguaje grandes (LLMs) extensivamente durante el desarrollo de esta práctica. Las herramientas de IA asistieron en:

- Estructuración y redacción del código del notebook
- Implementación de la función de cálculo de perplejidad
- Diseño de la estrategia de generación de texto con sub-word tokens
- Integración del modelo base (word-level) y la sección de análisis comparativo
- Redacción y revisión de la documentación (README)

### Disclaimer:
Me averguenza haber usado tanto LLMs para hacer esta practica. Pero dada la demora, mi falta de equipo y mi rendimiento general en el curso tengo que tomar medidas desesperadas. :'v lo siento Diego. Adicionalmente, si alguien más está sin equipo y las siguientes prácticas requieren de uno, no tengo problemas en que me hagan matchmaking.
---

## ¿Qué es la Perplejidad?

En el contexto del Procesamiento de Lenguaje Natural, la **perplejidad** es la métrica intrínseca más utilizada para evaluar el desempeño de un modelo de lenguaje. De manera intuitiva, cuantifica qué tan confundido está un modelo al observar un conjunto de datos de prueba nunca antes visto.

Si un modelo asigna una alta probabilidad a secuencias de palabras reales (es decir, modela correctamente las regularidades estadísticas del idioma), su nivel de confusión será bajo y, por consiguiente, su perplejidad también lo será. En términos de teoría de la información, equivale al tamaño promedio del conjunto de palabras de donde el modelo está tratando de "adivinar" la siguiente palabra válida.

## ¿Cómo se calcula?

Para calcular la perplejidad de un modelo de lenguaje sobre una secuencia de prueba formada por las palabras $W = w_1, w_2, \dots, w_N$, necesitamos partir de la probabilidad $P(W)$ conjunta que el modelo le asigna a toda esa secuencia. Esta probabilidad se calcula utilizando la regla de la cadena de la probabilidad, la cual descompone $P(W)$ en el producto de las probabilidades condicionales de cada palabra dado su contexto previo (su historia):

$$P(W) = P(w_1, w_2, \dots, w_N) = \prod_{i=1}^N P(w_i | w_1, \dots, w_{i-1})$$

La perplejidad, denotada como $PP(W)$, se define matemáticamente como la inversa de la probabilidad de la secuencia normalizada por la raíz enésima (donde $N$ es el número total de palabras evaluadas):

$$PP(W) = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}}$$

Sustituyendo el cálculo de la probabilidad según la regla de la cadena, la fórmula completa queda de la siguiente manera:

$$PP(W) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i | w_1, \dots, w_{i-1})}}$$

**Cálculo mediante Entropía Cruzada (Cross-Entropy)**

En su implementación, multiplicar muchas probabilidades (valores muy pequeños entre 0 y 1) genera problemas graves de subdesbordamiento aritmético (*underflow*). Por este motivo, la perplejidad se calcula comúnmente en el espacio logarítmico, utilizando la métrica de entropía cruzada, $H(W)$:

$$H(W) = -\frac{1}{N} \sum_{i=1}^N \log P(w_i | w_1, \dots, w_{i-1})$$

Una vez obtenida la entropía cruzada media (típicamente empleando el logaritmo en base 2 o logaritmo natural), la perplejidad se calcula como la exponencial del resultado:

$$PP(W) = 2^{H(W)} \quad \text{o} \quad PP(W) = e^{H(W)}$$

## Relación entre la Perplejidad y la Calidad del Modelo

La relación entre la perplejidad y la calidad de un modelo predictivo es **inversamente proporcional**:

*   **Menor perplejidad:** Indica que el modelo logra predecir con mayor certidumbre y exactitud la siguiente palabra de la secuencia. Es un síntoma directo de un modelo de alta calidad, reflejando que su distribución interna de probabilidad se asemeja bastante a la distribución real que originó el corpus.
*   **Mayor perplejidad:** Sugiere que el modelo tiene gran incertidumbre y está asignando probabilidades demasiado difusas o francamente erróneas al vocabulario, lo que denota una pobre capacidad de modelado.

En la fase de entrenamiento, el objetivo de la función de pérdida suele ser minimizar esta entropía cruzada, lo cual reduce paralelamente la perplejidad.

## Ventajas y Limitaciones

El uso de la perplejidad ofrece ventajas y limitaciones:

### Ventajas
1.  **Evaluación Intrínseca Rápida:** No exige la preparación de tareas posteriores (como traducción automática o preguntas/respuestas) para medir el rendimiento. Permite validar rápidamente los efectos del ajuste de hiperparámetros.
2.  **Estandarización:** Es de conocimiento general en el ámbito científico, lo cual hace factible comparar el rendimiento puro (modelado estadístico) entre diferentes arquitecturas, desde clásicos N-gramas hasta modelos densos autoatencionales (Transformers).
3.  **Fundamento Matemático:** Cuenta con sólidas bases provenientes de la teoría de la información.

### Limitaciones
1.  **Dependencia Estricta del Vocabulario:** **Es matemáticamente inválido comparar las perplejidades de dos modelos que posean vocabularios diferentes** o estrategias de tokenización distintas (ej. palabras completas vs. sub-palabras como WordPiece). Un modelo que colapsa palabras poco frecuentes en un token `<UNK>` reduce el tamaño real de su espacio de búsqueda y obtiene una perplejidad bajísima de forma espuria.
2.  **No garantiza éxito Extrínseco:** Una mejora en la perplejidad no se traduce automáticamente de manera lineal en un mejor desempeño para tareas aplicadas del mundo real.
3.  **Sensibilidad de Dominio:** La métrica exige que los datos de prueba pertenezcan a la misma distribución de los de entrenamiento. Evaluar un modelo entrenado con literatura clásica usando texto de redes sociales resultará en una altísima perplejidad, sin que esto signifique necesariamente que el modelo es malo; simplemente refleja que la tarea está fuera de su dominio de aprendizaje (*out-of-distribution*).

## Referencias

1.  Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*. (3rd ed. draft). Capítulo 3: N-gram Language Models. Stanford University. Disponible en línea en: [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
2.  Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
3.  Shannon, C. E. (1948). A Mathematical Theory of Communication. *The Bell System Technical Journal*, 27(3), 379-423.
4.  Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of machine learning research*, 3(Feb), 1137-1155.
