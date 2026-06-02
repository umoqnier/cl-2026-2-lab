# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="49cbfc80-2f6b-4dfe-a24f-58b334d3a696"
# # **Práctica 5: Fine-tunning y puesta en producción de modelos**
#
# ### **Nombre:** Omar Fernando Gramer Muñoz
# ### **Materia:** Lingüística Computacional
# ### **Matrícula:** 419003698

# %% [markdown] id="8b70a472-7e61-4287-bc45-83c66983b97b"
# ## **Objetivo**
# Realizar fine-tuning de un modelo pre-entrenado (`distilbert-base-uncased`)
# para la tarea de **Extractive Question Answering** usando el dataset **SQuAD**,
# y desplegarlo como una aplicación web con Gradio en Hugging Face Spaces.

# %% colab={"base_uri": "https://localhost:8080/"} id="410fece0-2c47-4d6d-a2f9-f7c29a7162a6" outputId="93a57f5e-4a93-47a1-9c18-a91817cf9abb"
# Instalamos las librerías necesarias para el proyecto
# transformers: modelos pre-entrenados de Hugging Face
# datasets: para cargar y manipular datasets
# evaluate: para evaluar el rendimiento del modelo
# python-dotenv: para cargar variables de entorno de forma segura
# gradio: para construir la interfaz web de la app

# !pip install transformers datasets evaluate python-dotenv gradio -q

# %% colab={"base_uri": "https://localhost:8080/"} id="2d5c8b44-6f87-43fa-be95-5285321b8f91" outputId="16661ed7-746c-4be3-a0a0-831628137118"
# Verificamos que las librerías se instalaron correctamente
import transformers
import datasets
import evaluate
import gradio

print(f"transformers: {transformers.__version__}")
print(f"datasets:     {datasets.__version__}")
print(f"evaluate:     {evaluate.__version__}")
print(f"gradio:       {gradio.__version__}")

# %% colab={"base_uri": "https://localhost:8080/"} id="e59d1491-114b-4c00-afae-00f732cc301b" outputId="892967b6-320f-4cb6-815a-afa27fea8962"
# Importaciones
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np
import gradio as gr

print("✅ Librerías importadas correctamente")

# %% [markdown] id="215c0777-cb2d-4bf7-9363-3048d6831651"
# ## **Autenticación en Hugging Face**
#
# Cargamos el token de acceso desde el archivo `.env` para autenticarnos
# en el Hub de Hugging Face de forma segura.

# %% id="7b411fe4-80c1-46ba-9d06-63cbbdcde697"
from dotenv import load_dotenv
import os
from huggingface_hub import login

# Cargamos las variables de entorno del archivo .env
load_dotenv()

# Obtenemos el token
token = os.getenv("HF_TOKEN")

# Nos autenticamos en Hugging Face
login(token=token)

# %% [markdown] id="a223b4b2-3e67-4679-8265-a30121132830"
# ## **Carga del dataset**
#
# Usaremos **SQuAD** (Stanford Question Answering Dataset), uno de los datasets
# más populares para la tarea de Extractive Q&A.
#
# Contiene pares de (contexto, pregunta, respuesta) donde la respuesta
# es siempre un fragmento extraído directamente del contexto.
#
# Para que el fine-tuning sea viable en una computadora local, usaremos
# solo un subconjunto pequeño del dataset.

# %% colab={"base_uri": "https://localhost:8080/"} id="2149621c-995b-4118-a2f8-83949cbf8984" outputId="2cc09fc3-4731-4c69-c510-716e8d78f3f0"
from datasets import load_dataset

# Cargamos solo una fracción pequeña del dataset para que sea manejable
# train: 5000 ejemplos, validation: 500 ejemplos
dataset = load_dataset("squad")

dataset["train"] = dataset["train"].select(range(5000))
dataset["validation"] = dataset["validation"].select(range(500))

print(dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="13854bc9-fe97-44df-ba81-02a4f73f3a23" outputId="cd11ad33-6125-4be6-d091-bce51a7f1e2b"
# Exploramos un ejemplo para entender la estructura del dataset
ejemplo = dataset["train"][0]

print("Contexto:")
print(ejemplo["context"])
print("\nPregunta:")
print(ejemplo["question"])
print("\nRespuesta:")
print(ejemplo["answers"])

# %% [markdown] id="e1a41e8a-365b-463d-9e83-648fe88f1ead"
# ## **Tokenizer y modelo base**
#
# Usaremos `distilbert-base-uncased` como modelo base. Es una versión
# más ligera de BERT (40% menos parámetros) que mantiene el 97% de su
# rendimiento, lo que lo hace ideal para fine-tuning en recursos limitados.
#
# El tokenizer convierte el texto en tokens numéricos que el modelo puede procesar.

# %% colab={"base_uri": "https://localhost:8080/", "height": 327, "referenced_widgets": ["b6f115567dbf416487c2d5f15136d5a4", "f97e0625db66406d963bc809d17e2ebd", "615e8dedd8a4494e8b109f83237937de", "4f8550894a22400d887d52fac952f154", "3df564812b2e42b1b67c42858fdfa6fe", "1f73917573454292aa497388a8fd3568", "b39cd75c694942b38443c086faaa7298", "851b3981a5254d6da83f8ff9f1292784", "561405f2a8c24b6f960a10f1be80285b", "e9b6b486da664b3cb9076a5129e7c74f", "481369154c474077920b399d4789f5aa"]} id="4517e609-b08f-473d-9bd7-6aca3bfb7e71" outputId="f0be22c4-8503-46de-bac7-c38f9d78b9a4"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

checkpoint = "distilbert-base-uncased"

# Cargamos el tokenizer del modelo base
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Cargamos el modelo pre-entrenado para la tarea de Q&A
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

print(f"Modelo cargado: {checkpoint}")
print(f"Número de parámetros: {model.num_parameters():,}")

# %% [markdown] id="6f8f639d-e4bd-4476-a9df-6163ebfa6f2e"
# ## **Preprocesamiento de los datos**
#
# Antes de entrenar necesitamos transformar los datos al formato que el modelo espera.
# Esto incluye:
#
# - Tokenizar el par (pregunta, contexto)
# - Manejar secuencias largas con sliding window (stride)
# - Calcular las posiciones de inicio y fin de la respuesta en los tokens

# %% id="db26d5b9-e606-4cb1-9467-e4549e204903"
# Parámetros de tokenización
MAX_LENGTH = 384   # Longitud máxima de la secuencia
STRIDE = 128       # Solapamiento entre ventanas para contextos largos

def preprocess(examples):
    # Eliminamos espacios extra de las preguntas
    questions = [q.strip() for q in examples["question"]]

    # Tokenizamos el par (pregunta, contexto)
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",   # Solo truncamos el contexto, no la pregunta
        stride=STRIDE,
        return_overflowing_tokens=True,  # Permite sliding window
        return_offsets_mapping=True,     # Mapeo de tokens a caracteres
        padding="max_length",
    )

    # Mapeamos cada feature al ejemplo original
    sample_map = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx]

        # Índices de inicio y fin de la respuesta en caracteres
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Encontramos los límites del contexto en los tokens
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Si la respuesta no está en el contexto de esta ventana
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Encontramos el token de inicio
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # Encontramos el token de fin
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


# %% colab={"base_uri": "https://localhost:8080/", "height": 255, "referenced_widgets": ["ce00d11d9b2c45cbb4638dc062764ca9", "7807ce4aef504c4ca7c2fdf2cf55ad2a", "7abf9bf0596144eda7b6c4ac238dd90c", "24ed88f0e13045869d5a01f0e4ccdeb1", "3b7c8b482df24f6b8d9a50dc40e4e0a7", "28c7e52f6ec14e8fb99ced42235cd52e", "51ab739c390a49d694b5f39767b0022c", "b30d1fb0061c4c22a5600c31c431d190", "a9f255ec4b97465693db4882e9317865", "8401e4c66515469f95a1441eacfc1ab3", "581d9b31b05642e196d00dbc545f087b", "784d2559d4ed40e3917fa988243c14d0", "f65bf00c989148d9a3f0d9e2040ebaf4", "a286ad3b6ec645799799da919b76d54a", "d75775de50b74e33a63298236e79d527", "5567601b0fed425f9185603bfb0425e8", "ce43f7bdd78048c492fa0c82bb6b3b39", "5363ee6083ea42d08f3e90cb569dc97d", "081bd98c159e47cfba21ec45de047e21", "10589a89edfe4745a61dadfe39b7673c", "ee38dbbb2f43487cb9110f34e24a1920", "d9e6e7f1c0e3441780b53fe51b1cbf8f"]} id="9583268a-4928-46d9-9b85-5fa26ac99545" outputId="53becb11-2042-49ac-acab-e4185eea81d7"
# Aplicamos el preprocesamiento a todo el dataset
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print(tokenized_dataset)

# %% [markdown] id="66d3e725-e932-48f8-88e2-5f0781534261"
# ## **Entrenamiento del modelo usando Fine-tuning**
#
# El fine-tuning es una técnica de transferencia de aprendizaje (*transfer learning*)
# que consiste en tomar un modelo pre-entrenado en una tarea general y adaptarlo a una
# tarea específica con un conjunto de datos más pequeño. En lugar de entrenar desde cero,
# aprovechamos el conocimiento que el modelo ya adquirió, lo que reduce significativamente
# el tiempo y los recursos necesarios.
#
# En nuestro caso, partimos de `distilbert-base-uncased`, pre-entrenado sobre grandes
# volúmenes de texto en inglés, y lo adaptamos para la tarea de Extractive Question
# Answering usando el dataset SQuAD.
#
# Usaremos la clase `Trainer` de Hugging Face que simplifica enormemente
# el ciclo de entrenamiento. Solo necesitamos definir:
#
# - Los hiperparámetros de entrenamiento
# - El modelo, datos y tokenizer

# %% id="2e07bfc4-9d52-4639-a098-d852885df551"
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-finetuned-squad",  # Carpeta donde se guardan los checkpoints
    eval_strategy="epoch",               # Evaluamos al final de cada época
    save_strategy="epoch",                     # Guardamos al final de cada época
    learning_rate=2e-5,                        # Tasa de aprendizaje
    num_train_epochs=6,                        # Número de épocas
    per_device_train_batch_size=16,            # Ejemplos por batch en entrenamiento
    per_device_eval_batch_size=16,             # Ejemplos por batch en evaluación
    weight_decay=0.01,                         # Regularización
    load_best_model_at_end=True,               # Cargamos el mejor modelo al finalizar
    push_to_hub=True,                          # Subimos el modelo al Hub al finalizar
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 310, "referenced_widgets": ["da83a5de3b0945f1bf499537e6ac3a22", "692c50908e5443cba2c93e8c9fd7d47b", "7f1f716c46514e908d6485a9a652527a", "bf8c7cd09f6144a3a6846b94e1a8256b", "ad30e3ec91944cd7931531d2eac0c291", "bf109a9184f344769b8ca739f79cef3b", "348f56dbc3c54d198420b5d849621cd0", "a61451b65d0943848d9a048f95aa149c", "eb9b3d3f1b6c47f5b90e2a9bde9c6174", "bb947229c04e4442a5ae0a2f2d4fe3a7", "f576d5eaf8114d52ba7180793f901c87"]} id="jLqK0kDqugR1" outputId="b6a9b67e-52a4-4f78-f38f-b5dfe71a8140"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Recargamos el modelo base desde cero para reentrenar limpiamente
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

print("Modelo base recargado correctamente")

# %% colab={"base_uri": "https://localhost:8080/", "height": 528, "referenced_widgets": ["cad6aa7518a24ac39d7b3f558500e587", "de8b01baeb644c959b37a551ee1ae30b", "56e217d9681349aebdd9a46d5f95bdf9", "b3f694ba75474b22a611c8933c4896ba", "e58f61a88c674ed08df811226a637c3a", "03ce3417e05646e882657a52ee69a7bd", "0898341835954e10877abcdc2d4fe73d", "9e2b0a6f52a5430989a0b536bd3e3f4f", "d57daf5fc787485ab5a53ec8f9f5d7a5", "5d56532cdf4647fa9ba154b750126b33", "3b6a0d7d15f947f892532a2c6a162f78", "a246763b2ba44d9386adf5b565bbf2b8", "44eb29116bfe4e859c3e0e33b4730126", "e9fffde1b5904dcb98a8c1a0d75819ff", "4e6142aa60ea40dba22109db4920f95d", "e4ca143427a74c02a1bd7f464eaa1889", "2977fef51ad749b89bc616afbdc1cd00", "63c3a95eeb7046768368a95a4f3f9d44", "e145d63625544cad9cecf85fd34d507f", "b5895b5edc7a4c28ba75db3d282adc03", "334afbfb40884920891ed90a50bb2c48", "1c532aba5af443528768598dcb6fa11c", "bf5efc85629a4b17a49fe413800cf2db", "9faf98ee75894a3f8e09eb075a4ee586", "c0da7662e2dd4025b8836b045386ae02", "5507681123f5421591e63ef763d2f6dc", "1a56333999144abd8098d8e41524defb", "89cdbfe33a414538b031183dba59a45e", "b028813ee87844178fc3713de349f2c2", "d61860c8249448f7a23a32968a8c9551", "6ff17a55b49346f5b0b2000b6f3b3e06", "2891c76308e141ac974dc728dbb151ed", "ac18168de6ae4a018decee0577bde74c", "ab0b51eb243249599977bacbc031c383", "be5985be0a8e453bbfaa7597025a7ece", "2e17dc586ac242629cf2bce3d5274ffa", "6878af2d044d4d788a3c960bcde5971f", "af142defc37c40f38e2dcd0674c6a96e", "4801ba4d739e4f199ae711aecfbb1723", "c0e537905e6344e8aadfbe92fb362940", "8f65e46530d046c3b325d113c941471f", "04de113ebd25406dbe884ddcc9c4d8ff", "73f1c79193a34c7c8b2b372d74928a6c", "7bd74d288ca84a30acf3ebcffa1cb47f", "5b5cf09d9eec4dc6900c50bd56886145", "e827633057b44f6abfdc92670f5ef632", "a0069aac13c8456dac283e202681e4fc", "c69df46f506d4033942af8407fc06346", "20ad9e2cd88d446eb3d5410290dc796a", "0c17b2c9377743998a6887b337842e51", "0abf2dd0ef2a4704b1549fdd64fa87ff", "b06b63ee1bb343ac9aaf0301341b6149", "5e8b78c389994f9184bb5804639d25c6", "aaea6b7a9172490f9782dfa8fb35b473", "1208985df1474f25a3b02478fba9defb", "4860602f5e1647f19e418463c4c27dae", "97e0e15c598641dfb8ac9ca31f3e0f70", "ce95604d69164c66a56e82a011a5a4b5", "55b60f4ce11747118d07f6358ad234ed", "fcff9f81de2245c880c06368c09fe299", "0433e226af5e43c9a86d0cdf27d9b5be", "326d52560b534590adb5762d6045867c", "6d624a96aa2c4d36a96cde35ee764c33", "003a436851c64f6d891780eb7ed5694a", "d5f2014fc7fb4ab38720ed3af3ed47b7", "0bdd6c98deea44d28cf83ce72629d586"]} id="ffa643bf-fb89-4045-9620-6b85e45670aa" outputId="70df752a-9949-42ab-c17d-8cfae0405831"
from transformers import Trainer
# Entrenamiento del modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
)

print("Trainer listo. Iniciando fine-tuning...")
trainer.train()

# %% [markdown] id="a49b6b34-bdc5-4ece-bbbb-3ababc8bed63"
# ## **Probando el modelo**
#
# Verificamos que el modelo fine-tuneado responde correctamente antes de desplegarlo como aplicación web.

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["8434b976e5bd468b808c8955c1dd6573", "0fa13f98908e46bca8e18552e346d446", "70cdae45d36f4909be90fa749e115add", "51b528f9f01a439aab31362732a4985d", "a98c7a635bde4027aa4dc469e48ad4f3", "6a6f110faaf84b019008ea3963fd4a07", "9768d83a932e469ba06ee93a1c16a7e0", "420730399324445d89c5ab90ae7d9ca9", "2ccb3db261e84cebb68ade7e9e86b12b", "ae78dcf0ba634ea1b90e66504a31f2d0", "634ff7724d0747c293abbda19ffa5940"]} id="14e63cb9-58bb-4cbb-bc86-c15c39cde591" outputId="57a70083-5954-4452-cca1-3edb19b6f811"
from transformers import pipeline

# Recargamos el modelo actualizado desde el Hub
qa_pipeline = pipeline(
    "question-answering",
    model="grameromarFC/distilbert-finetuned-squad"
)


# %% colab={"base_uri": "https://localhost:8080/"} id="3xkdWfdtp6o9" outputId="d7285a1c-9b42-46c2-b2f5-459d7b2b3955"
# Función reutilizable para probar el modelo
def probar(pregunta, contexto):
    resultado = qa_pipeline(question=pregunta, context=contexto)
    print(f"Pregunta:  {pregunta}")
    print(f"Respuesta: {resultado['answer']}")
    print(f"Confianza: {resultado['score']:.4f}")
    print("-" * 60)

# Agrega tus propios ejemplos aquí
probar(
    pregunta="...",
    contexto="..."
)

# %% colab={"base_uri": "https://localhost:8080/"} id="0484cfb5-0ccb-4b15-9f77-c908d58f956b" outputId="bfc5bb75-6eec-4567-e9f6-771da71ba8af"
# Pruebas adicionales del modelo

probar(
    pregunta="When did World War II end?",
    contexto="""
    World War II was a global conflict that lasted from 1939 to 1945. It involved
    the majority of the world's nations and was the deadliest conflict in human history.
    The war ended in Europe on May 8, 1945, known as Victory in Europe Day, and in
    the Pacific on September 2, 1945, when Japan formally surrendered aboard the USS Missouri.
    """
)

probar(
    pregunta="What is the capital of Australia?",
    contexto="""
    Australia is a country and continent surrounded by the Indian and Pacific oceans.
    Its capital is Canberra, which was purpose-built to serve as the nation's capital
    after a dispute between Sydney and Melbourne over which city should hold that role.
    Sydney is the largest city in Australia, while Melbourne is the second largest.
    """
)

probar(
    pregunta="Who wrote the theory of relativity?",
    contexto="""
    The theory of relativity was developed by Albert Einstein in the early 20th century.
    It consists of two theories: special relativity, published in 1905, and general
    relativity, published in 1915. Einstein received the Nobel Prize in Physics in 1921,
    though it was awarded for his discovery of the photoelectric effect, not relativity.
    """
)

probar(
    pregunta="How many planets are in the solar system?",
    contexto="""
    The solar system consists of the Sun and the objects that orbit it. There are eight
    recognized planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
    Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union.
    The four inner planets are rocky, while the four outer planets are gas or ice giants.
    """
)

probar(
    pregunta="What language is spoken in Brazil?",
    contexto="""
    Brazil is the largest country in South America and the fifth largest in the world.
    The official language of Brazil is Portuguese, which was introduced during the colonial
    period when Portugal claimed the territory in 1500. Brazil is the only Portuguese-speaking
    country in South America. Its capital is Brasília and its largest city is São Paulo.
    """
)

# %% [markdown] id="ZtxpYjQr4zOo"
# ## **Análisis de resultados**
#
# El modelo respondió correctamente en 5 de 7 pruebas, con alta confianza en la mayoría
# de los casos. Las dos respuestas marcadas como parcialmente correctas o inseguras
# revelan patrones interesantes sobre las limitaciones del modelo:
#
# - **Preguntas "quién" y "cuál"**: el modelo responde con alta confianza (0.6 - 0.99).
#   Esto se debe a que SQuAD contiene muchos ejemplos de este tipo, por lo que el modelo
#   aprendió bien a identificar entidades como personas, lugares e idiomas.
#
# - **Preguntas "cuándo" y "cuántos"**: el modelo responde correctamente pero con menor
#   confianza (0.08 - 0.23). Cuando el contexto contiene múltiples fechas o números,
#   el modelo tiene dificultad para discriminar cuál es la respuesta correcta. Por ejemplo,
#   en la pregunta sobre el fin de la Segunda Guerra Mundial, el contexto menciona dos fechas
#   (May 8 y September 2) lo que genera ambigüedad.
#
# ### Limitaciones identificadas
#
# - El modelo fue entrenado con únicamente 5,000 ejemplos, una fracción pequeña del
#   dataset SQuAD completo (87,000 ejemplos). Esto limita su capacidad de generalización.
#
# - Se observó **overfitting** a partir de la época 3: la pérdida de validación comenzó
#   a aumentar mientras la pérdida de entrenamiento seguía bajando. El mejor modelo
#   correspondió a la época 2 (val. loss: 1.778).
#
# - El modelo solo funciona en **inglés** y únicamente puede extraer respuestas que
#   estén explícitamente presentes en el contexto proporcionado. No es capaz de
#   inferir ni generar información nueva.
#

# %% [markdown] id="ZgOwWdC46IXH"
# ## **Despliegue de la aplicación con Gradio**
#
# Gradio es una librería de Python que permite crear interfaces web interactivas
# para modelos de machine learning con muy pocas líneas de código. Desplegaremos
# nuestra app en **Hugging Face Spaces**, una plataforma gratuita que nos dará
# una URL pública para compartir el modelo.
#
# La aplicación desplegada está disponible en:
# 🚀 [grameromarFC/qa-distilbert-squad](https://huggingface.co/spaces/grameromarFC/qa-distilbert-squad)
#
# A continuación se muestra también una versión local de la interfaz ejecutada
# directamente desde el notebook, para ilustrar el funcionamiento de la app
# antes de su despliegue.

# %% colab={"base_uri": "https://localhost:8080/", "height": 657, "referenced_widgets": ["c3d9ed174ade4dc68a2ce41750af9bfc", "01e3fa35aafd44b897d3cbfb72585f3d", "c61309a902164c97beca04c37d3cc76d", "098a75f6e0fe4bdda8892bb1effa0e1b", "6f9327f62e034c1db9eba551cf422dca", "e0df0489ee7e44029d1c24734d4031e9", "8292fcde463c4c1facf421a8db01087a", "0876cb6e5c47483d934709000934a542", "4feaf1b91cea4969a44316ece13777a4", "a9aad3b6b34d4dc9a2c922c150e711d0", "64c6e2e4d849438fa23310737d5197ed"]} id="lUaub3u06OYE" outputId="9eb7b576-2166-447d-8478-7b4c9ebdcfa3"
import gradio as gr
from transformers import pipeline

# Cargamos el modelo desde el Hub
qa_pipeline = pipeline(
    "question-answering",
    model="grameromarFC/distilbert-finetuned-squad"
)

# Función que conecta el modelo con la interfaz
def responder(pregunta, contexto):
    if not pregunta or not contexto:
        return "Por favor ingresa una pregunta y un contexto."
    resultado = qa_pipeline(question=pregunta, context=contexto)
    respuesta = resultado["answer"]
    confianza = f"{resultado['score']:.2%}"
    return f"**Respuesta:** {respuesta}\n\n**Confianza:** {confianza}"

# Construimos la interfaz
demo = gr.Interface(
    fn=responder,
    inputs=[
        gr.Textbox(label="Pregunta", placeholder="Escribe tu pregunta aquí..."),
        gr.Textbox(label="Contexto", placeholder="Pega el texto donde buscar la respuesta...", lines=8),
    ],
    outputs=gr.Markdown(label="Resultado"),
    title="Extractive Question Answering",
    description="Ingresa un contexto y una pregunta. El modelo extraerá la respuesta directamente del texto.",
    examples=[
        ["Who designed the Eiffel Tower?", "The Eiffel Tower is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889."],
        ["What language is spoken in Brazil?", "Brazil is the largest country in South America. The official language of Brazil is Portuguese, introduced during the colonial period."],
    ]
)

demo.launch()

# %% [markdown] id="W3dwLT5qQBpO"
# ## **Punto Extra: Medición de emisiones de CO₂ con CodeCarbon**
#
# El entrenamiento de modelos de lenguaje consume grandes cantidades de energía eléctrica,
# lo que genera emisiones de CO₂ y contribuye al cambio climático. Cuantificar este impacto
# es una práctica cada vez más importante en el campo de la inteligencia artificial responsable.
#
# **CodeCarbon** es una librería de Python que mide en tiempo real el consumo energético
# de tu código y lo convierte en emisiones equivalentes de CO₂, tomando en cuenta factores
# como la ubicación geográfica del hardware y la fuente de energía utilizada.
#
# En esta sección integraremos CodeCarbon directamente en el ciclo de entrenamiento
# para reportar el costo ambiental del fine-tuning de nuestro modelo, el cual ha sido entrenado usando Google collab.

# %% colab={"base_uri": "https://localhost:8080/"} id="JVoRr0WlPlwD" outputId="e8a330f9-9a92-41cb-9b1a-54455fa2040b"
# !pip install codecarbon -q

# %% colab={"base_uri": "https://localhost:8080/", "height": 597, "referenced_widgets": ["7f3f694f6ae6446e8df11562015cd06e", "854200915efc47c5a5888beb308aef89", "7043ff1aa011442c9cd3800abe5a4715", "4cc78b897aff474f9097fbc3e195383b", "440cee13c47b453b9e7818b23bc510f6", "ef5de70c56794d849c6d8d636b4a89c3", "7ffbba411f3c46b180dbcc3f5e731f91", "93118042345d413290fcd7a938ee66dd", "dbfb94661ce04e3bba8a1bac62a4cd29", "1522da67e3a949e0898feaadfb6fd7fd", "496e6b11e23c4f51803adaf524d416ca", "6da0395a7e464f4282b9c3c1a38b760d", "8574f15d9d12493b9cf79a7c4925000a", "96b517bffadf404f920248f0d2efae16", "81ecfd5beb2b4c3d95d41e4fef81d241", "46a95d342172439b8bbbc289a1b0b9eb", "3f038b224cdf4684adc6fad69265e746", "674a22d8e511438daf6a95327de911bf", "230c1c122f7a4d10953c5e131f740cf8", "f4cb50547f244405bee84246f3da63a2", "18e62ffcf0214d7282c9c7432290c673", "94f2f761b6e047c686eeb51047c4a466", "e0304b72dd3c4c88994a97086ca20054", "a494279ff9e34ca0b46215c492be0063", "7f1bef26f9bb4f1387a0644fc50e8111", "d701c48ec2c54b9092adfed8a71f5769", "a1a700cb64864246859c428eb9ac9340", "b52166ac9853427c963681a77829a39a", "9f3da5d2195141c3813903888ea6a21c", "116f8630e88c47179524d462ab30fcb6", "127ca9dbc02742c191134696ad66b8a7", "aabde3fd3db74fd2909fe61044a7f694", "44f87cb5a32e49e79d5ed67a9cc02e23", "00eb713cacfb4d0bace1ea15f0fdbddb", "7dc22f4373f54f1eb60de9247aedc56c", "102eb212463b4233a97d0132cb0c8210", "314f23c0d46c4eb9a6f218a6a32f6e8e", "01bc724980ba4b54bc3d57b0bd9436c8", "357f881e30d545398dd458cb33ad6af7", "c1a4c29e4f7043d99010b2a948c7b732", "1ecc688c696248b2adb53ba04936f814", "dd157d7a66f7434e9101b4128ba46376", "096ff1b26eee4959aeb76d2e1e8f5827", "ce4c3204101643d29beed4da97363ac8", "ca0a124a9f28488784a5a262159ecd9e", "5a2037a8c95a44cbac274011b03c118f", "b08114e855cb4c4a8ddb966d62fe3d16", "86c07c7ac2a04b70b156bd5e42f07846", "546396d543dc4cd59e04b7317c9b58f9", "9539137749944694805770633a02e8c5", "6c323b1b5d784070a7447b81ef4d2a6a", "e6bd784f49f2455881dc1da9daf21d34", "702a668ab1584d44a0b90c397b1de218", "902f2b9da12940d8bf75f14bb0f6addc", "57bd4c7016c044848f95b81156001346", "1c6541357ddb41038dcd01889b49c12b", "c6c67dbbafe5487ba6abae17ef83f5ed", "f2c6db47245d4b1f95ab84ede15d0b4c", "8a6af31762cb41fb9e89daa2dc30f3b5", "7bb41cd5de794d0490a4c06c9d8586f1", "022b68c9df2c47c0abb4540d1fa44f0a", "ac03d60d32c3463487bcd79840dbcd7a", "596ef10a9d28442c817d56085f19173a", "bc1e3fd7f38a4411a0e259299e7861b9", "2ef2a87f2aee4da8a1ac8c7337cabfbc", "c131ff47fbed428093c866fb0cc0dee9"]} id="TQlF1Bp7Pxmy" outputId="79235d2d-8acf-4c5f-f27c-7f06f639433f"
from transformers import Trainer
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="distilbert-finetuned-squad",
    output_dir=".",           # Guarda el reporte en la carpeta actual
    log_level="error",        # Solo muestra errores, no logs intermedios
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
)

# Iniciamos el tracker antes de entrenar
tracker.start()
trainer.train()

# Detenemos el tracker y obtenemos las emisiones
emissions = tracker.stop()

print(f"\n{'='*50}")
print(f"  Emisiones de CO₂: {emissions * 1000:.4f} gramos")
print(f"  Equivalente a:    {emissions * 1000 / 21:.6f} km en automóvil")
print(f"{'='*50}")

# %% [markdown] id="JGxpMAIrUqfG"
# ## **Resultados de emisiones de CO₂**
#
# | Métrica | Valor |
# |---|---|
# | Emisiones totales | 5.3821 gramos de CO₂ |
# | Equivalente en transporte | 0.2563 km en automóvil |
# | Ejemplos de entrenamiento | 5,000 |
# | Épocas | 6 |
#
# El fine-tuning de `distilbert-base-uncased` generó **5.38 gramos de CO₂**,
# equivalente a recorrer aproximadamente 256 metros en automóvil.
#
# Si bien esta cifra parece pequeña, es importante contextualizarla:
#
# - Entrenamos con apenas 5,000 ejemplos de un dataset que contiene 87,000
# - Usamos un modelo "pequeño" de 66 millones de parámetros
# - El entrenamiento duró aproximadamente 20 minutos
#
# Modelos más grandes como GPT-4 o BERT-large entrenados desde cero
# sobre datasets completos pueden generar cientos de toneladas de CO₂.
# Esto refuerza la importancia del fine-tuning como alternativa más
# sostenible frente al entrenamiento desde cero.
