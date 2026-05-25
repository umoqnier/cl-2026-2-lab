# Práctica 5: Fine-tuning y puesta en producción de modelos

**Integrantes:** Roberto Samuel Sánchez Rosas

Contiene el desarrollo de la Práctica 5 para la asignatura de Lingüística Computacional. El objetivo es realizar el proceso completo de ingeniería de aprendizaje profundo de extremo a extremo (*End-to-End*): desde el entrenamiento adaptativo (*Fine-Tuning*) de un modelo del lenguaje preentrenado, hasta su auditoría ambiental y despliegue final como una aplicación web interactiva y pública en la nube.

## Contenido

- **Fine-Tuning de ALBERT**: Proceso de optimización de la arquitectura para la clasificación de secuencias a nivel de tokens.
- **Análisis de Rendimiento (Métricas de Calidad)**: Evaluación estadística del modelo sobre el corpus de validación utilizando métricas estandarizadas de procesamiento de lenguaje natural.
- **Auditoría de Sustentabilidad Ambiental**: Monitoreo de la huella de carbono y gasto energético derivado del cómputo intensivo mediante la librería *CodeCarbon*.
- **Puesta en Producción (Prototipo Web)**: Desarrollo de una interfaz de usuario interactiva desplegada en la infraestructura de Hugging Face Spaces.

## Enlaces al Proyecto

El modelo entrenado, el código fuente del servidor web y el prototipo interactivo pueden ser accedidos e inspeccionados en los siguientes enlaces:

- **Prototipo en Producción (Hugging Face Spaces)**: [https://huggingface.co/spaces/samuelsrosas/albert-ner-fciencias-lc]
- **Model Registry (Pesos y Configuración)**: Alojado directamente en la raíz del espacio de producción para garantizar una ejecución autónoma.

---

## Evaluación de la Tarea y Utilidad de la App

### Rendimiento del Modelo

El proceso de *Fine-Tuning* sobre la arquitectura **ALBERT-base-v2** se estructuró utilizando el dataset  **CoNLL-2003** enfocado en el Reconocimiento de Entidades Nombradas (NER). Tras un entrenamiento controlado de 3 épocas (750 pasos de optimización global) con un optimizador AdamW y un esquema de regularización por decaimiento de pesos, el modelo alcanzó una convergencia limpia y estable, reflejada en las siguientes métricas de validación:

| Métrica de Evaluación | Valor Obtenido | Significado e Interpretación Lingüística |
| :--- | :---: | :--- |
| **Precision** | **91.89%** | Alta fidelidad: de cada 100 entidades que el modelo etiqueta, aproximadamente 92 son correctas, minimizando los falsos positivos en las predicciones. |
| **Recall** | **92.52%** | Alta cobertura: el modelo logra recuperar e identificar de forma efectiva el 92.5% de todas las entidades reales presentes en el corpus. |
| **F1-Score Global** | **92.20%** | Balance óptimo: la media armónica confirma la robustez y el alto rendimiento del modelo ALBERT adaptado de forma individual. |
| **Accuracy (Exactitud)** | **98.33%** | Precisión global: reflejo del excelente manejo de los tokens fuera de entidad (`O`), que dominan estadísticamente el texto. |

La pérdida de validación exhibió un comportamiento de convergencia ideal a lo largo de las tres épocas, descendiendo de forma continua desde `0.098` en la primera época, pasando por `0.076` en la segunda, hasta alcanzar su punto mínimo y más estable de **`0.072`** en la época final. Al mismo tiempo, la pérdida de entrenamiento se mantuvo controlada en `0.152`. Este comportamiento balanceado descarta cualquier indicio de sobreentrenamiento, demostrando que el modelo alcanzó su punto óptimo de generalización estadística para la tarea NER.

### Utilidad del Prototipo Web

La aplicación web desarrollada en *Gradio* cumple con una función de abstracción tecnológica indispensable. En la práctica, el backend del clasificador computa probabilidades sobre un espacio matemático de **9 dimensiones** dictado por el esquema estándar de anotación de secuencias **BIO** (`B-PER`, `I-PER`, `B-LOC`, etc.).

Para un usuario final, inspeccionar etiquetas BIO en crudo resulta impráctico y ruidoso. El prototipo web implementado soluciona esto mediante una capa de presentación limpia: toma las salidas probabilísticas, unifica las secuencias continuas y renderiza un resaltador de texto dinámico que ilumina las palabras completas bajo 4 categorías amigables (`Persona`, `Organización`, `Ubicación` y `Misceláneo`). Esto convierte un modelo abstracto de Deep Learning en una herramienta de software interactiva, transparente y de utilidad inmediata para la extracción de conocimiento.

---

## Retos y Dificultades Tecnológicas

### 1. Durante la fase de Fine-Tuning (Modelado)

**Desalineación de Etiquetas por Sub-words:** El reto metodológico más complejo fue la pérdida de correspondencia biunívoca entre las palabras originales del dataset y los subtokens generados por el algoritmo *SentencePiece* de ALBERT. Al romper vocablos complejos o nombres no nativos del inglés (como fragmentar "Ana" en los subtokens "A" y "na"), los arreglos de entrada se desalineaban de los vectores de etiquetas originales. Se tuvo que implementar una rutina matemática de tokenización extendida para propagar la etiqueta de la palabra original únicamente al primer subtoken de la secuencia y rellenar los subtokens restantes con un índice especial (`-100`) ignorado nativamente por la función de pérdida *Cross-Entropy*.

### 2. Durante la Puesta en Producción (Ingeniería de Software)

**Fragmentación Visual en el Frontend:** Inicialmente, al levantar el pipeline en bruto dentro de la interfaz, el tokenizador SentencePiece exponía sus divisiones internas al usuario, mostrando palabras cortadas acompañadas del caracter especial `_`. Esto rompía la usabilidad de la interfaz. La dificultad se superó incorporando la estrategia de agregación `aggregation_strategy="simple"` directamente en el pipeline de Hugging Face, delegando al backend la fusión indexada de caracteres antes de enviarlos al componente visual `gr.HighlightedText`.

**Aislamiento del Entorno Local y Dependencias Ocultas:** Al gestionar el entorno local mediante la herramienta `uv`, surgió un conflicto de alcance con los entornos virtuales globales del repositorio. Además, el modelo ALBERT requería librerías de soporte que no se explicitan en entornos BERT tradicionales (como `protobuf` y `sentencepiece`). Se aisló correctamente el entorno de la práctica mediante la bandera `--active` en `uv`.

---

## Sustentabilidad

En alineación con las buenas prácticas de la computación verde y la transparencia en IA, se auditó el ciclo completo de desarrollo de este notebook. El gasto energético de la GPU Tesla T4 provista por la infraestructura en la nube y los núcleos de procesamiento de la CPU fueron monitoreados de forma transparente. El reporte completo de consumo se encuentra exportado en el directorio `./output_metrics/emissions.csv`, sirviendo como métrica base para contrastar el costo ecológico contra la ganancia estadística en el F1-Score.

## Estructura de Archivos

- `Practica05LC.ipynb`: Notebook principal con la carga y preparación del corpus CoNLL-2003, monitoreo con CodeCarbon, Fine-Tuning de ALBERT y evaluación final.
- `Practica05LC.py`: Script de Python con las definiciones modulares y funciones de la práctica.
- `app.py`: Script de Python que define la arquitectura del servidor web, las funciones de mapeo de etiquetas y la interfaz gráfica interactiva en Gradio.
- `/output_metrics/`: Carpeta que resguarda el archivo de auditoría ambiental de CodeCarbon.

## Requisitos y Reproducción

Para reproducir los experimentos, evaluar el modelo o revisar las métricas de sustentabilidad ambiental, basta con ejecutar secuencialmente las celdas del archivo principal `Practica05LC.ipynb` dentro de un entorno con soporte para aceleración por hardware (GPU Tesla T4 recomendada).

Las dependencias de software requeridas se instalan automáticamente en la primera celda del notebook mediante el gestor de paquetes:

```bash
!pip install transformers datasets evaluate codecarbon accelerate seqeval -q
```
