# Práctica 5: Fine-tuning y Puesta en Producción de Modelos

**Alumno:** Omar Fernando Gramer Muñoz  
**Materia:** Lingüística Computacional

---

## Descripción

En esta práctica se realizó el fine-tuning de un modelo de lenguaje pre-entrenado
para la tarea de **Extractive Question Answering (Q&A)**, que consiste en responder
preguntas extrayendo la respuesta directamente de un contexto de texto proporcionado
por el usuario.

---

## Tarea de NLP

**Tipo de tarea:** Extractive Question Answering  
**Modelo base:** `distilbert-base-uncased`  
**Dataset:** SQuAD (Stanford Question Answering Dataset)  

El modelo recibe dos entradas:
- Un **contexto**: párrafo de texto en inglés
- Una **pregunta**: pregunta cuya respuesta se encuentra en el contexto ( tambien en inglés )

Y produce como salida el fragmento exacto del contexto que responde la pregunta.

---

## Proceso de entrenamiento

El fine-tuning se realizó en dos entrenamientos para comparar mejorias el rendimiento del modelo:

| | Primera ronda | Segunda ronda |
|---|---|---|
| Ejemplos de entrenamiento | 1,000 | 5,000 |
| Épocas | 3 | 6 |
| Mejor val. loss | 3.085 | **1.778** |
| Training loss final | 3.807 | **1.316** |

La segunda ronda mostró una mejora significativa, con el modelo respondiendo
correctamente y con alta confianza tras aumentar el tamaño del dataset. 
( **Nota**: Cabe mencionar que solo dejé los parámetros del segundo entrenamiento en el notebook )

---

## Resultados
A continuación presento algunas de las pruebas ejecutadas con el modelo:

| Pregunta | Respuesta | Confianza |
|---|---|---|
| Who designed the Eiffel Tower? | Gustave Eiffel | 90.61% |
| Who founded the University of Notre Dame? | Father Edward Sorin | 61.18% |
| When did World War II end? | May 8, 1945 | 23.23% |
| What is the capital of Australia? | Canberra | 97.49% |
| Who wrote the theory of relativity? | Albert Einstein | 99.21% |
| How many planets are in the solar system? | eight | 8.86% |
| What language is spoken in Brazil? | Portuguese | 96.31% |

El modelo responde con alta confianza preguntas de tipo **quién** y **cuál**.
Las preguntas de tipo **cuándo** y **cuántos** generan respuestas correctas
pero con menor confianza, debido a la ambigüedad cuando el contexto contiene
múltiples fechas o números.

---

## Modelo y aplicación

- 🤗 **Modelo en Hugging Face Hub:** [grameromarFC/distilbert-finetuned-squad](https://huggingface.co/grameromarFC/distilbert-finetuned-squad)  
- 🚀 **Aplicación desplegada:** [grameromarFC/qa-distilbert-squad](https://huggingface.co/spaces/grameromarFC/qa-distilbert-squad)

---

## Tecnologías utilizadas

- `transformers` — Modelo y pipeline de Hugging Face
- `datasets` — Carga y manipulación del dataset SQuAD
- `gradio` — Interfaz web interactiva
- `Hugging Face Spaces` — Plataforma de despliegue

---

## Emisiones de CO₂ (CodeCarbon)

El entrenamiento fue monitoreado con **CodeCarbon v3.2.7** para medir el impacto
ambiental del fine-tuning.

### Reporte de emisiones

| Métrica | Valor |
|---|---|
| Emisiones totales | 5.3821 gramos de CO₂ |
| Tasa de emisiones | 4.49 × 10⁻⁶ kg/s |
| Energía consumida | 0.0388 kWh |
| Duración del entrenamiento | 1,198 segundos (~20 minutos) |
| Equivalente en transporte | ~0.256 km en automóvil |

### Hardware utilizado

| Componente | Detalle |
|---|---|
| CPU | Intel Xeon @ 2.00GHz (2 núcleos) |
| GPU | Tesla T4 |
| RAM total | 12.67 GB |
| Utilización CPU | 62.34% |
| Utilización GPU | 94.28% |

### Contexto geográfico

El entrenamiento se ejecutó en **Google Colab**, con servidores ubicados en
**Oregon, Estados Unidos**. La región de Oregon es relevante porque una parte
significativa de su energía proviene de fuentes renovables como la hidroeléctrica,
lo que reduce el impacto ambiental respecto a otras regiones.

### Reflexión

Las 5.38 gramos de CO₂ generadas corresponden a un modelo pequeño (66M parámetros)
entrenado con 5,000 ejemplos durante 20 minutos. Modelos de mayor escala como
GPT-4 o LLaMA, entrenados desde cero sobre cientos de miles de millones de tokens,
pueden generar cientos de toneladas de CO₂. Esto subraya la importancia del
**fine-tuning** como alternativa más sostenible al entrenamiento desde cero,
y de herramientas como CodeCarbon para hacer visible el costo ambiental
del desarrollo de modelos de lenguaje.


## Reporte final

### ¿Qué tan bien se resolvió la tarea?

El modelo logró resolver correctamente 5 de 7 pruebas realizadas, con resultados
muy satisfactorios en preguntas de tipo **quién**, **cuál** y **qué idioma**,
alcanzando niveles de confianza superiores al 90% en varios casos.

Sin embargo, el modelo mostró limitaciones en preguntas de tipo **cuándo** y
**cuántos**, donde la confianza bajó considerablemente (8% - 23%). Esto se debe
a que cuando el contexto contiene múltiples fechas o números, el modelo tiene
dificultad para discriminar cuál es la respuesta correcta.

En general, considerando que se entrenó con apenas 5,000 ejemplos (menos del 6%
del dataset SQuAD completo), el rendimiento es razonablemente bueno y demuestra
el potencial del fine-tuning como técnica de transferencia de aprendizaje.

### ¿Qué tan útil fue la app?

La app es funcional y fácil de usar para casos donde el usuario dispone de un
texto de referencia y quiere extraer información específica de él. Casos de uso
prácticos incluyen resumir documentos, consultar artículos cortos o responder
preguntas sobre textos pequeños.

No obstante, la app tiene limitaciones claras:
- Solo funciona en **inglés**
- Solo puede responder preguntas cuya respuesta esté **explícitamente en el contexto**
- No es capaz de inferir, razonar ni generar información nueva

### Retos y dificultades

**Durante el fine-tuning:**
- El modelo inicial entrenado con 1,000 ejemplos respondía incorrectamente,
  devolviendo fechas o números independientemente del tipo de pregunta. Fue
  necesario aumentar el dataset a 5,000 ejemplos para obtener respuestas coherentes.
- Se observó overfitting a partir de la época 3: la pérdida de validación comenzó
  a subir mientras la de entrenamiento seguía bajando. Esto obligó a confiar en
  `load_best_model_at_end=True` para recuperar el mejor checkpoint.
- Hubo incompatibilidades entre versiones de la librería `transformers` que generaron
  errores con argumentos renombrados como `eval_strategy` y `tokenizer` en el `Trainer`.

**Durante el despliegue en producción:**
- La versión de Python (3.13) en Hugging Face Spaces era incompatible con versiones
  antiguas de `transformers` y `tokenizers`, lo que requirió varias iteraciones en
  el `requirements.txt` para encontrar versiones estables.
- La tarea `question-answering` no estaba disponible en la versión más reciente de
  `transformers` instalada por defecto en Spaces, lo que obligó a reimplementar el
  pipeline manualmente controlando `token_type_ids`, que DistilBERT no utiliza.
