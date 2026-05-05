# Práctica 4: Evaluación de modelos del lenguaje neuronales

**Integrantes:** Roberto Samuel Sánchez Rosas

Contiene el desarrollo de la Práctica 4 para la asignatura de Lingüística Computacional. El objetivo es entrenar, evaluar y comparar modelos del lenguaje neuronales (utilizando la arquitectura propuesta por Bengio et al., 2003), contrastando un enfoque tradicional basado en palabras completas frente a un modelo basado en sub-words mediante la compresión Byte-Pair Encoding (BPE).

## Contenido

- **Investigación (Perplejidad)**: Síntesis teórica sobre el cálculo y evaluación de la métrica de perplejidad en modelos del lenguaje.
- **Creación de Modelos del Lenguaje**:
  - **Modelo Base (Word-based)**: Entrenamiento con palabras completas, generando un vocabulario extenso para observar su comportamiento ante datos no vistos.
  - **Modelo Sub-word (BPE)**: Aplicación del algoritmo de compresión para limitar el vocabulario y mejorar la generalización morfológica.
- **Análisis Comparativo**: Contraste empírico de los modelos, incluyendo discusión de desempeño, ventajas, desventajas y recomendaciones de mejora.
- **Estrategia de Generación de Texto**: Implementación de una rutina de decodificación utilizando *Top-p (Nucleus) Sampling* para generar secuencias de palabras a partir de sub-unidades.

## Enlaces a los modelos

Los tensores de peso de las redes neuronales y el modelo de reglas BPE pueden ser descargados desde los siguientes enlaces:

- **Modelo Base (Palabras):** [Enlace drive](https://drive.google.com/file/d/1D1uLZHZL-gTM_yixAKDq52IExDIxO0CG/view?usp=sharing)
- **Modelo Sub-word (BPE):** [Enlace drive](https://drive.google.com/file/d/1sq_vTM4Cr_5jG51DS8IN8Wpd7bzTWK31/view?usp=drive_link)
- **Modelo de Reglas BPE (subword-nmt):** [Enlace drive](https://drive.google.com/file/d/1NoP0K2ufqWLBUrIxL0vN-C_TAPvJYuVL/view?usp=sharing)

### Código para cargar los modelos en memoria

Para evaluar los modelos sin necesidad de reentrenarlos, se debe instanciar la arquitectura y cargar los pesos descargados mediante el siguiente código:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeloNeuronalNgramas(nn.Module):
    def __init__(self, tam_vocabulario, dim_emb, tam_contexto, dim_oculta):
        super().__init__()
        self.embeddings = nn.Embedding(tam_vocabulario, dim_emb)
        self.linear1 = nn.Linear(tam_contexto * dim_emb, dim_oculta)
        self.linear2 = nn.Linear(dim_oculta, tam_vocabulario)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x_emb = self.embeddings(x)
        x_aplanado = x_emb.view(x.size(0), -1)
        activacion = torch.tanh(self.linear1(x_aplanado))
        activacion = self.dropout(activacion)
        prediccion = self.linear2(activacion)
        return F.log_softmax(prediccion, dim=1)

# Configuración del dispositivo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ejemplo de carga para el Modelo Base
# (Para el modelo BPE, cambiar tam_vocabulario a 9969)
modelo_evaluacion = ModeloNeuronalNgramas(tam_vocabulario=32631, dim_emb=200, tam_contexto=2, dim_oculta=100)
modelo_evaluacion.load_state_dict(torch.load('ruta/al/modelo_base.pt', map_location=dispositivo))
modelo_evaluacion.eval()

print("¡Modelo cargado exitosamente!")
```

## Perplejidad

### ¿Qué es la perplejidad?

La perplejidad (perplexity) es una medida para evaluar modelos de lenguaje basada en la probabilidad que estos asignan a un conjunto de prueba. Un modelo mejor asigna mayor probabilidad al texto, ya que predice con mayor precisión las palabras siguientes y por lo tanto, está menos "sorprendido" por ellas. Sin embargo, la probabilidad total de una secuencia depende de su longitud, disminuyendo a medida que el número de palabras aumenta. Para solucionar este problema, la perplejidad se define como la probabilidad inversa del conjunto de prueba, normalizada por el número de palabras o tokens, lo que permite obtener una medida por palabra y comparar resultados entre textos de distinta longitud.

### Fórmula matemática

Para una secuencia de prueba de longitud $N$, denotada como $W = w_1, w_2, \dots, w_N$, la perplejidad se define como la inversa de la probabilidad de la secuencia, normalizada por su longitud:

$$
\begin{align*}
\text{perplexity}(W) &= P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}}\\
&= \sqrt[N]{\frac{1}{P(w_1, w_2, \ldots, w_N)}}
\end{align*}
$$

Alternativamente se puede utilizar la regla de la cadena para expandir la probabilidad de la secuencia W:

$$
\text{perplexity}(W) = \sqrt[N]{ \prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1\ldots w_{i-1})}}
$$

### Relación con la calidad del modelo

La perplejidad está inversamente relacionada con la probabilidad que el modelo asigna a la secuencia de prueba. Debido a esta relación inversa, cuanto mayor es la probabilidad del texto, menor es la perplejidad, y por tanto mejor es el modelo. En consecuencia, minimizar la perplejidad es equivalente a maximizar la probabilidad del conjunto de prueba según el modelo de lenguaje.

### Ventajas y limitaciones

- **Ventajas**: Es una métrica cuantitativa bien definida que permite evaluar y comparar modelos de lenguaje de forma objetiva, ya que está directamente basada en las probabilidades que el modelo asigna a los datos de prueba. Esto permite medir mejoras en el modelo sin depender de tareas específicas como traducción o clasificación.

- **Limitaciones**: La perplejidad depende del conjunto de datos de evaluación y del modelo utilizado, por lo que no es una medida absoluta de "calidad lingüística". Además, su valor depende del esquema de tokenización y del vocabulario, lo que puede dificultar comparaciones entre modelos con representaciones distintas. Por último una baja perplejidad no garantiza que el texto generado sea coherente o adecuado en términos semánticos o de uso en aplicaciones reales.

### Fuentes

- Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models* (3rd ed.). Online manuscript released January 6, 2026. <https://web.stanford.edu/~jurafsky/slp3/>

## Análisis Comparativo

| Métrica | Modelo Base (Palabras) | Modelo Sub-word (BPE) |
| :--- | :--- | :--- |
| **Perplejidad (genesis)** | 120.97 | 1865.29 |
| **Tamaño vocabulario** | 32,631 | 9,969 |
| **OOV Rate** | 38.57% | 3.96% |

A primera vista, el **Modelo Base** exhibe una perplejidad significativamente menor (120.97) frente al Modelo Sub-word (1865.29). Sin embargo, esta diferencia está fuertemente influenciada por la alta tasa de tokens fuera de vocabulario (OOV Rate del 38.57%). En el corpus de evaluación (Genesis), el Modelo Base no puede representar una gran parte del texto y sustituye estos casos mediante tokens OOV, lo que distorsiona el cálculo de probabilidad y por tanto de la perplejidad.

El **Modelo Sub-word (BPE)** muestra una mayor capacidad de generalización al reducir significativamente el OOV Rate a 3.96% al representar las palabras mediante subunidades con un vocabulario más compacto (9,969 unidades). Aunque su perplejidad numérica es mayor, esto se debe a que trabaja con una segmentación más fina del texto, lo que hace que la predicción se distribuya sobre secuencias más largas de tokens. En consecuencia, la perplejidad no es directamente comparable entre ambos enfoques.

### Ventajas y desventajas de cada enfoque

- **Modelo Base (Word-based):**
  - **Ventajas:** Su arquitectura es más intuitiva pues cada token representa una palabra completa con significado semántico directo.
  - **Desventajas:** Presenta vocabularios muy grandes y un alto consumo de memoria en las matrices de embeddings. Además, es altamente vulnerable a palabras fuera de vocabulario (OOV), como errores ortográficos, variantes morfológicas o neologismos.

- **Modelo Sub-word (BPE):**
  - **Ventajas:** Reduce significativamente el problema de OOV al representar el lenguaje mediante subunidades frecuentes, manteniendo un vocabulario de tamaño fijo y más compacto.
  - **Desventajas:** Fragmenta las palabras en múltiples tokens, lo que incrementa la longitud de las secuencias y puede hacer la representación menos intuitiva.

### Recomendaciones para mejorar ambos modelos

Para el Modelo Base aplicar técnicas de preprocesamiento como la lematización (reducción de palabras a su forma base) para disminuir la variabilidad del vocabulario y reducir la aparición de palabras fuera de vocabulario (OOV). También puede considerarse la adopción de técnicas sub-word para mejorar la cobertura del lenguaje.

Para el Modelo Sub-word incrementar el número de iteraciones de entrenamiento pues como la tokenización BPE aumenta la longitud de las secuencias y la complejidad de las relaciones entre subunidades, el modelo suele requerir más épocas para converger y capturar mejor la estructura del lenguaje. Tambien podría mejorar la capacidad del modelo mediante arquitecturas más avanzadas que capturen dependencias de largo alcance, como redes neuronales recurrentes (LSTM) o modelos basados en atención como Transformers. Esto permite mejorar la predicción al modelar mejor el contexto completo de la secuencia. Además, ajustar el tamaño del vocabulario subword puede ayudar a equilibrar granularidad y eficiencia.

## Estructura de Archivos

- `Practica04LC.ipynb`: Notebook principal con la ejecución de los experimentos, la tokenización, el entrenamiento de la arquitectura de Bengio et al., y la generación de secuencias.
- `Practica04LC.py`: Script de Python con las definiciones modulares y funciones de la práctica.

## Requisitos y Dependencias

Para ejecutar los scripts y el notebook de esta práctica, es necesario contar con un entorno de Python 3.10+ y las siguientes dependencias:

```bash
pip install torch nltk numpy subword-nmt
```
