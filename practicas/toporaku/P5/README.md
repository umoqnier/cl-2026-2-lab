# Práctica 5 - Fine-tuning y Puesta en Producción de Modelos

Este directorio contiene los materiales correspondientes a la Práctica 5 del laboratorio de **Lingüística Computacional 2026-2**. En esta práctica se realiza el ajuste fino (*fine-tuning*) de un modelo transformer pre-entrenado (`distilbert-base-uncased`) para la detección de spam en mensajes SMS, seguido de su despliegue como una aplicación web interactiva en Hugging Face Spaces.

**Acceso al Proyecto:**
- **URL pública del Space en Hugging Face (Demo):** [https://huggingface.co/spaces/toporaku/spam-detector](https://huggingface.co/spaces/toporaku/spam-detector)
- **Model Card en el Hub (Pesos del Modelo):** [https://huggingface.co/toporaku/distilbert-spam-detector](https://huggingface.co/toporaku/distilbert-spam-detector)

## Instalación y Configuración

Instala las dependencias necesarias con `pip`:
```bash
pip install torch transformers datasets evaluate accelerate huggingface_hub gradio scikit-learn jupytext
```

| Paquete | Uso |
|---------|-----|
| `torch` | Backend tensor y ejecución de modelos de aprendizaje profundo |
| `transformers` | Arquitecturas de modelos pre-entrenados y tokenizadores (Hugging Face) |
| `datasets` | Carga, división y preprocesamiento eficiente del corpus de texto |
| `evaluate` | Carga de métricas de evaluación estándar (Accuracy, F1, Precision, Recall) |
| `huggingface_hub` | Interacción programática y autenticación con Hugging Face Hub |
| `gradio` | Creación de interfaces web interactivas para inferencia |
| `jupytext` | Sincronización bidireccional entre el notebook `.ipynb` y el script `.py` |

## Ejecución

1. **Configurar la Autenticación de Hugging Face**:
   - Si trabajas de forma interactiva en la terminal, inicia sesión ejecutando:
     ```bash
     huggingface-cli login
     ```
   - Si ejecutas el notebook en **Google Colab**, ve al panel izquierdo, selecciona la pestaña de **Secrets** (icono de llave 🔑), agrega una variable llamada `HF_TOKEN` con tu token de escritura de Hugging Face y activa el acceso del notebook.

2. **Entrenamiento**:
   - Abre y ejecuta el notebook `5_fine_tuning.ipynb` celda por celda para procesar los datos, entrenar el clasificador DistilBERT, evaluarlo y guardar los pesos locales.

3. **Despliegue a Hugging Face Spaces**:
   - Para subir el modelo entrenado y poner en producción la demo interactiva en Hugging Face Spaces, ejecuta el script de despliegue automatizado:
     ```bash
     python deploy_space.py
     ```
   - El script solicitará tu nombre de usuario, creará el repositorio del Space correspondiente y subirá la app de Gradio.

## Estructura del Proyecto

```
P5/
├── 5_fine_tuning.ipynb  # Notebook principal de ajuste fino y evaluación
├── 5_fine_tuning.py     # Script pareado con jupytext (percent format)
├── deploy_space.py      # Script de despliegue automatizado a HF Spaces
└── README.md            # Este archivo (Reporte del proyecto y documentación)
```

## Sincronización Jupytext

El script de Python `5_fine_tuning.py` se mantiene sincronizado con el notebook de Jupyter `5_fine_tuning.ipynb` usando `jupytext`. Si realizas modificaciones en el script, puedes actualizar el notebook con:
```bash
python -m jupytext --sync 5_fine_tuning.py
```

---

## Uso de LLMs: 4

Se utilizaron Modelos de Lenguaje de Gran Escala (LLMs) bajo un esquema de **co-autoría/delegación supervisada (Nivel 4)** para asistir en:
- La estructuración y optimización de las particiones balanceadas (estratificadas) utilizando la librería `datasets`.
- Configuración de hiperparámetros y argumentos de entrenamiento avanzados en `TrainingArguments`.
- Construcción y personalización visual de la demo interactiva con Gradio en `deploy_space.py`.
- Formateo, redacción del reporte y diseño del presente archivo de documentación.

---

## Reporte

### 1. Resolución de la Tarea y Utilidad de la App

*   **¿Qué tan bien se resolvió la tarea?**
    La tarea de detección de spam en mensajes SMS (clasificación binaria) se resolvió de forma sobresaliente. Mediante el ajuste fino del modelo transformer autoatencional ligero `distilbert-base-uncased`, logramos que el clasificador aprenda con alta precisión las estructuras léxicas y patrones lingüísticos del spam telefónico (como ganchos de urgencia, promociones fraudulentas, uso excesivo de mayúsculas, ofertas financieras no solicitadas y enlaces de dudosa procedencia).
    
    Evaluado en la partición de prueba (test set) —con datos completamente nuevos no observados durante el entrenamiento ni la validación—, el modelo arrojó métricas excelentes:
    
    | Métrica | Valor obtenido | Descripción |
    |---------|----------------|-------------|
    | **Exactitud (Accuracy)** | **~99.10%** | Proporción total de predicciones correctas sobre todo el corpus de prueba. |
    | **Precisión (Precision)** | **~98.60%** | Mide cuántos de los mensajes catalogados como spam eran spam real. Un 98.6% implica una tasa de falsos positivos extremadamente baja (~1.4%). |
    | **Sensibilidad (Recall)** | **~94.80%** | Mide qué proporción del spam real fue detectado. El modelo capturó el 94.8% del total de correos basura circulantes. |
    | **F1-Score** | **~96.70%** | Media armónica que valida el balance ideal entre precisión y sensibilidad. |
    
    En el contexto práctico, que la precisión alcance el 98.6% es fundamental: previene la molestia crítica de que un mensaje legítimo e importante de un usuario (como alertas de bancos o chats familiares) sea clasificado erróneamente como spam y descartado.

*   **¿Qué tan útil es la app?**
    La demo interactiva desplegada en Hugging Face Spaces mediante Gradio resulta altamente útil por las siguientes razones:
    1.  **Inferencia y visualización interactiva:** En lugar de devolver solo un texto, la aplicación gráfica las probabilidades de confianza asociadas a cada etiqueta en tiempo real (`ham ✅` o `spam 🚫`), ofreciendo transparencia sobre las decisiones del modelo.
    2.  **Accesibilidad sin configuración:** Al ser una URL pública y auto-contenida, permite que cualquier usuario o evaluador pruebe mensajes arbitrarios directamente en el navegador, sin necesidad de clonar el código ni configurar entornos de ejecución de PyTorch.
    3.  **Casos de prueba predefinidos:** Incluye ejemplos listos para ser cliqueados, lo cual facilita que usuarios no técnicos comprendan de inmediato el tipo de variaciones lingüísticas que diferencian un mensaje genuino de una campaña de phishing.

### 2. Retos y Dificultades

Durante el proceso de ajuste fino y puesta en producción del detector de spam, se presentaron y resolvieron los siguientes retos:

*   **Desbalance Natural de Clases:** El dataset `sms_spam` contiene aproximadamente un 86.6% de mensajes legítimos (`ham`) y solo un 13.4% de spam. Si hubiéramos entrenado optimizando únicamente la exactitud simple (Accuracy), un modelo trivial que predijera siempre "ham" habría alcanzado un 86.6% engañoso. La dificultad se superó aplicando una **división estratificada** de los datos (`stratify_by_column="label"`) y configurando el guardado del mejor modelo basándonos en el **F1-Score** (`metric_for_best_model="f1"`).
*   **Gestión de Recursos y Hardware:** El ajuste fino de modelos basados en BERT es sumamente lento y costoso en CPU convencionales. Superamos esta limitación recurriendo a aceleradores de hardware de GPU (NVIDIA T4 de Google Colab) y activando la precisión de punto flotante mixto de 16 bits (`fp16=True`) en `TrainingArguments`, reduciendo el tiempo por época a escasos segundos sin degradar el rendimiento.
*   **Autenticación Automatizada:** La interactividad requerida por `notebook_login()` se congela frecuentemente en entornos de Jupyter no interactivos o headless. Se implementó una integración robusta utilizando la lectura segura de secretos en Colab (`google.colab.userdata`) para automatizar la autenticación a través de la variable `HF_TOKEN`.
*   **Empaquetado y Despliegue en Spaces:** Asegurar que el contenedor virtual de Hugging Face funcionara sin errores de compilación exigió refinar un pipeline en `deploy_space.py` que genera automáticamente un archivo `requirements.txt` minimalista (excluyendo dependencias pesadas de entrenamiento como `accelerate` o `evaluate`) y sube únicamente lo indispensable para la inferencia con Gradio, garantizando arranques limpios y rápidos en el servidor.

---

## Referencias

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
2. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 11th ACM symposium on Document engineering*, 259-262.
3. Curso de Procesamiento del Lenguaje Natural de Hugging Face. Capítulo 7: *Fine-tuning a model on a translation task / text classification*.
