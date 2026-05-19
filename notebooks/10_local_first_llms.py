# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10. Local first LLMs

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/10_local_first_llms.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
# ![](https://realpython.com/cdn-cgi/image/width=960,format=auto/https://files.realpython.com/media/How-to-Integrate-Local-LLMs-With-Ollama-and-Python_Watermarked.835ee5f2672d.jpg)
#
# > sauce - https://realpython.com/ollama-python/

# %% [markdown]
# ## Objetivos

# %% [markdown]
# - Instalar LLMs locales
# - Explorar las capacidades de LLMs locales
#     - Integrarlos con un editor de código para asistencia
# - Fundamentos de RAG e implementación con modelos locales

# %% [markdown]
# ## ollama

# %% [markdown]
# ![](https://ollama.com/public/ollama.png)

# %% [markdown]
# Ollama es una app chiquita que permite obtener y correr LLMs de forma sencilla. Podemos visitar la [documentación](https://github.com/ollama/ollama/blob/main/README.md#quickstart) para saber más características. Hay una lista de modelos disponibles y filtros para identificar sus capacidades.

# %% [markdown]
# ### Modelos de Embeddings

# %% [markdown]
# ![](https://ollama.com/public/blog/what-are-embeddings.svg)

# %% [markdown]
# Estos modelos estan especialmente diseñados para generar vectores de embeddings (o seá arreglos de números que capturan características semánticas y de otros tipos de la entrada). Estos modelos son fundamentales para crear aplicaciones como *RAGs*.
#
# > source: https://ollama.com/blog/embedding-models

# %% [markdown]
# ### Modelos con soporte para *tools*

# %% [markdown]
# ![](https://wallpapercave.com/wp/wp2195747.jpg)

# %% [markdown]
# Llamar *tools* le da la posibilidad a los LLMs de realizar tareas complejas o interactuar con entornos fuera del contexto local del LLM:
#
# Ejemplos pueden ser los siguientes:
#
# - Uso de funciones pre-exitentes o APIs
# - Navegación en la web
# - Utilizar un interprete de código
#
# > source: https://ollama.com/blog/tool-support

# %% [markdown]
# ### Comandos útiles

# %%
# !ollama serve &

# %%
# !ollama --help

# %%
# !ollama list

# %%
# !ollama pull <modelo>

# %% [markdown]
# ### Modelos a utiliza:
#
# - `qwen3-embedding:0.6b`   
# - `phi4-mini:latest`
# - `gemma3:270m`
# - `qwen2.5-coder:0.5b` 

# %% [markdown]
# ### Probando en terminal

# %% [markdown]
# - Usando la CLI
#
# ```
# $ ollama run <model>
# ```
#
# - Usando la API
#
# ```bash
# curl http://localhost:11434/api/chat -d '{
#   "model": "gemma3:270m",
#   "messages": [
#     { "role": "user", "content": "why is the sky blue?" }
#   ]
# }'
# ```
#
# - Paquete para `python`

# %%
# !pip install ollama

# %%
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3:270m', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

# %%
print(response['message']['content'])

# %%
stream = chat(
    model="gemma3:270m",
    messages=[
  {
    'role': 'user',
    'content': 'What are key elements of quantum computing?',
  },
], stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# %% [markdown]
# ### Integrando llms a nuestro flujo de trabajo local

# %% [markdown]
# Vamos a integrar los modelos que obtenemos con `ollama` con VsCode via el plug-in llamado [Continue](https://docs.continue.dev/). Alternativamente se puede realizar la integración con multiples editores como [zed](https://zed.dev/).

# %% [markdown]
# #### Demo en vivo (esperemos que salga bien)
#
# - https://docs.continue.dev/guides/ollama-guide
# - https://docs.continue.dev/customize/deep-dives/autocomplete

# %% [markdown]
# ### Configuraciones
#
# ```json
# # ~/.continue/config.json
# models:
#   - name: Autodetect
#     provider: ollama
#     model: AUTODETECT
#     roles:
#       - chat
#       - edit
#       - apply
#       - rerank
#       - autocomplete
# ```

# %% [markdown]
# ## Question answering (*Q&A*) y la técnica de Retrieval-augmented generation (*RAG*)

# %% [markdown]
# Una tarea que han resuelto los LLMs es la generación de respuestas a preguntas del usuario. Sistemas especializados en *Q&A* de hecho han mostrado ser tan buenos o mejores que personas antes de los LLMs. Por ejemplo, [Watson](https://www.youtube.com/watch?v=P18EdAKuC1U) que ganó el juego Jeopardy en 2011 superando humanos en preguntas como:
#
# > Soy la comida por la que le hacen burla a los chilangos fuera de la CDMX^[1]
#
# [1]: Guajolota

# %% [markdown]
# Los sistemas de *Q&A* estan diseñados para completar información de acuerdo a las necesidades de las personas. Ya que mucha información está disponible en forma de texto (como en internet, libros o nuestros emails), estos sistemas están intimamente ligados a los motores de búsqueda. En realidad, esta distinción es cada vez más difusa ya que los motores de búsqueda actuales incorporan LLMs para proponer respuestas.
#
# <center><img src="https://static.boredpanda.com/blog/wp-content/uploads/2024/05/google-ai-overviews-2-66581fd5866f6__700.jpg" width=700></center>

# %% [markdown]
# En general los sistemas de *Q&A* se han enfocado en un subtipo de preguntas: **factoides**. Este tipo de preguntas pueden ser respondidas con simples hecho expresados en respuestas cortas o medianas. Por ejemplo:
#
# - ¿Dónde está el Museo de Antropología e Historia de la CDMX?
# - ¿Cómo poner el `@` en un teclado en inglés?
# - ¿Cómo instalar archlinux sin morir en el intento?^[2]
#
# [2]: Visita la wiki: https://wiki.archlinux.org/

# %% [markdown]
# ### ❓ Con los temas vistos ¿Cómo resolverían esta tarea?

# %% [markdown]
# Una opción es hacer un *fine-tunning* a un modelo pre-entrenado con un *dataset* de question-answering y despues crear prompts con la pregunta y la respuesta en blanco:
#
# > Q: ¿Dónde se encuentra la biblioteca del IIMAS? R: ____

# %% [markdown]
# ### Problemas de LLMs tirando factos

# %% [markdown]
# Los LLMs tienen varias deficiencias a la hora de responder a las preguntas que les hacemos
#
# - **Alucinaciones:** Los modelos tienden a alucinar, esto es que crean respuestas que parecen convincentes y bien formadas pero que no son reales en absoluto. Es dificil saber cuando un modelo está alucinando.
# - **Carencia de datos privados:** Los modelos han sido entrenados con grandes cantidades de datos pero no todos los datos posibles. Si queremos que respondan cosas acerca de nuestros correos o registros dentales probablemente no obtendremos respuestas satisfactorias.
# - **Datos estáticos:** Los modelos tienen problemas en responder preguntas acerca de eventos cuya información esta cambiando rápidamente. Los LLMs se entrenan con datos hasta alguna fecha.

# %% [markdown]
# ![](https://i.pinimg.com/originals/82/c1/22/82c122be87204cf8baa442aa27e68a84.gif)

# %% [markdown]
# ### Acerca de los *RAGs*

# %% [markdown]
# Por las razones antes enumeradas, una estrategía para que los LLMs realicen *Q&A* efectivamente es la de *retrieval-augmented generation (RAG)*. *RAG* utiliza técnicas de *Information Retrieval (IR)* para obtener documentos que serán reelevantes para responder a la pregunta del usuario. Despues se utiliza un LLM para generar una respuesta con base en los documentos obtenidos.
#
# Basar las respuestas en los documentos obtenidos resuelve varios de los problemas mencionados anteriormente. En primer lugar, ayuda a que la respuesta esté basada en hecho obtenidos de documentos previamente curados. Además, el sistema puede otorgar al usuario el contexto o documentos que tomó en cuenta para generar la respuesta (un ejemplo: [perplexity AI](https://www.perplexity.ai/search/las-bicicletas-de-pinon-fijo-s-Qr0l4YnETi.Q_d4yrxVyVA)). Esta característica brinda confianza y mayor explicabilidad. Por último, esta técnica permite agregar al sistema información personal o confidencial como registros médicos, legales o notas (aunque mucho ojo con dar sus datos a grandes empresas).

# %% [markdown]
# ### Arquitectura de un *RAG*

# %% [markdown]
# La idea principal es que dada una **pregunta** del usuario y tomando en cuenta un conjunto de **documentos reelevantes** previamente **obtenidos** se **generar** una respuesta. Podemos entonces dividirlo en dos fases:
#
# 1. *Retrieval:* Obtenemos los documentos reelevantes de alguna colección
# 2. *Generation:* Se genera una respuesta con base en estos documentos reelvantes

# %% [markdown]
# <center><img src="https://nextcloud.tepezil.net/apps/files_sharing/publicpreview/qsnw9Q8q4TWLQ88?file=/&fileId=74486&x=2560&y=1440&a=true&etag=4560b4a64d98ac7e1e16b563ce2db91e" width=700></center>
# > Tomada de Speech and Language Processing, (Jurafsky et al 2025)

# %% [markdown]
# Visto de otro modo, la tarea de *Q&A* puede modelarse como predicción de texto de forma auto-regresiva condicionada a un prompt con características particulares.
#
# ```
# Q: ¿Quien escribió el libro 'Bovedas de acero'? A:
# ```
#
# $$
# p(x_1,...,x_n) = \displaystyle\prod_{i=1}^{n} p(\texttt{[Q:]};q;\texttt{[A:]};x_{<i})
# $$
#
# Podemos hacer esto gracias a que los LLMs codifican una enorme cantidad de información en los parámetros gracias al acceso a muchísimos datos de entrenamiento. Sin embargo, si bien esté prompt servirá para responder preguntas *factoides*, aún tendriamos los problemas de alucinaciones, falta de evidencia en la respuesta y limitaciones con datos no disponibles de forma pública.
#
# Los *RAGs* lidian con este problema condicionando la respuesta con documentos reelevantes y algún prompts como: "Con base en los siguientes documentos, contesta la siguiente pregunta:"
#
# Supongase que tenemos una query $q$ y un conjunto de documentos reelevantes a la query $R(q)$ el prompt se vería como se muestra a continuación:
#
# ```c
# doc 1
# doc 2
# ...
# doc n
#
# Con base en los textos anteriores, responde esta pregunta: Q: "¿Quien escribió el libro 'Bovedas de acero'" A:
# ```
#
# $$
# p(x_1,...,x_n) = \displaystyle\prod_{i=1}^{n} p(x_i|R(q);prompt;\texttt{[Q:]};q;\texttt{[A:]};x_{<i})
# $$
#
# Se pueden combinar enfoques clásicos como $tfidf$ o $BM25$ con representaciones densas (AKA *embeddings*) de los documentos para la obtención y ordenamiento de los documentos reelevantes. Una parte importante es el *prompt engineering*; decidir como marcar la pregunta o los documentos o si agregar tokens especiales como `[SEP]` puede mejorar o empeorar nuestros resultados.
#
# - [Curso gratuito de Prompt Engineering, DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

# %% [markdown]
# ## Creando un RAG con `langchain`

# %%
import os
from rich import print as rprint

# Necesario que langchain vea esto
os.environ["OLLAMA_HOST"] = "127.0.0.1"
os.environ["OLLAMA_PORT"] = "11434"

# %%
rprint(os.environ["OLLAMA_HOST"])
rprint(os.environ["OLLAMA_PORT"])

# %% [markdown]
# ### Dependencias

# %% [markdown]
# ```
# "langchain-text-splitters>=0.3.8",
# "langchain-community>=0.3.21",
# "langgraph>=0.4.3",
# "langchain-ollama>=0.3.2",
# "langchain-chroma>=0.2.2",
# ```

# %% [markdown]
# ### Cargando un modelo

# %%
from langchain_ollama.chat_models import ChatOllama

MODEL = "qwen3.5:0.8b"

llm = ChatOllama(model=MODEL, reasoning=False)

# %% [markdown]
# ### Cargando embeddings

# %%
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# %%
import bs4
from rich import print as rprint
from rich.rule import Rule


# %% [markdown]
# ### Creación de un motor de búsquedas semánticas

# %% [markdown]
# LangChain utiliza abtracciones para integrar la carga y recuperación de información en bases de datos vectoriales, y otras fuentes, en un flujo con LLMs. Estas abstracciones son importantes para aplicaciones que requieren extraer datos y hacer "razonamiento" sobre los mismos como parte de la inferencia (como es el caso del RAG). Estas abstracciones son:

# %% [markdown]
# #### Documentos y cargadores

# %% [markdown]
# Los documentos representan unidades de texto con metadata que puede carpturar información sobre donde viene el documento, su relación con otros documentos y más.

# %%
from langchain_core.documents import Document

documents: list[Document] = [
    Document(
        page_content="Simply put, bikepacking is a mix of all-terrain cycling and backpacking.",
        metadata={"source": "bikepacking-doc"},
    ),
    Document(
        page_content="Bikepacking involves carrying the essential gear—and not much more—on an off-road-capable bike for an overnight or multi-day ride",
        metadata={"source": "bikepacking-doc"},
    ),
]

# %% [markdown]
# Sin embargo, es más común utilizar [*doc loaders*](https://python.langchain.com/docs/concepts/document_loaders/) para obtener [integraciones](https://python.langchain.com/docs/integrations/document_loaders/) varias con fuentes de datos.

# %%
from langchain_community.document_loaders import WebBaseLoader

# Cargador basado en extracción de datos de la web
loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://www.davidsbatista.net/blog/2017/11/13/Conditional_Random_Fields/",
        "https://www.davidsbatista.net/blog/2017/11/12/Maximum_Entropy_Markov_Model/"
        ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header", "post")
        )
    ),
)
docs = loader.load()

# %%
rprint(f"Total documents {len(docs)}")
for i, doc in enumerate(docs, start=1):
    rprint(Rule(f"Doc {i}"))
    rprint(doc.page_content[:300])
    rprint(doc.metadata)

# %% [markdown]
# #### Separadores de texto (*text splitters*)

# %% [markdown]
# Típicamente en las aplicaciones de recuperación de información o *question answering* una página de un documento puede ser demasiado grande para una representación. Lo que buscamos es obtener partes del documento para contestar preguntas basadas en la *query* de entrada y separar los documentos va a prevenir que porciones reelevantes del texto no sean opacadas por texto alrededor.
#
# Definiremos la cantidad de caracteres que tendrá cada *chunk* y los caracteres de *overlap*, que ayuda a mitigar que perdamos información reelevante al separar el documento. `RecursiveCharacterTextSplitter` separará recursivamente utilizando el salto de línea hasta obtener *chunks* del tamaño deseado.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# %%
rprint(f"Splits {len(all_splits)}")
for i, split in enumerate(all_splits[80:90], start=1):
    rprint(Rule(f"split #{i}"))
    rprint(split)

# %% [markdown]
# #### Embeddings

# %% [markdown]
# La forma habitual de hacer búsquedas en textos no estructurados es por medio de vectores. Recordemos que las representaciones vectoriales son útiles para hacer búsquedas basandonos en una medida de similitud (ej: *cosine similarity*).

# %%
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
rprint(f"Generated vectors of length {len(vector_1)}")
rprint("V1", vector_1[:10])
rprint("V2", vector_2[:10])

# %% [markdown]
# #### Creando la base de datos de embeggings

# %% [markdown]
# Los objetos `VectorStore` de langchain exponen métodos para agregar texto o `Documents` al almacenamiento y realizar *queries* con base en multiples medidas de similitud. Se inicializan con modelos de embeddings (en nuestro caso [Nomic Embed Text](https://ollama.com/library/nomic-embed-text)) que determinará como es que el texto será transformado a una representación vectorial.

# %%
from langchain_chroma import Chroma

vector_store = Chroma(
    # Nombramos nuestra colección
    collection_name="my_collection",
    # embeddings lo definimos más arriba
    embedding_function=embeddings,
    # Where to save data locally
    persist_directory="./my_chroma_langchain_db",
)

# %% [markdown]
# Una vez creada la base de datos vectorial podemos indexar los documentos y dada una *query* de entrada obtener documentos reelevantes.

# %%
# %%time
ids = vector_store.add_documents(documents=all_splits)

# %%
rprint(ids[:10])

# %%
results = await vector_store.asimilarity_search("Conditional Random Fields", k=3)

rprint(results)

# %%
embedding = embeddings.embed_query("What are the difference beetween HMM and MEMM?")

results = vector_store.similarity_search_by_vector(embedding)
rprint(f"Results={len(results)}")
for i, result in enumerate(results):
    rprint(Rule(f"Result #{i}"))
    rprint(result.page_content)

# %% [markdown]
# #### Generación

# %% [markdown]
# La lógica de la app consistirá en los siguientes pasos:
#
# 1. Tomar la pregunta del usuario
# 2. Obtener documentos reelevantes a la pregunta en cuestión
#     - La obtención de los documentos estará superditada por el *retriever* que definamos
# 3. Pasar los documentos obtenidos y la pregunta inicial al modelo
# 4. Generar una respuesta
#     - Para la generación usaremos el modelo obtenido de `ollama`

# %% [markdown]
# #### Creando el prompt

# %%
from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the
question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the
answer concise
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# %%
example_messages = prompt.invoke(
    {"context": "[blue]<My retrieved and absolutely relevant documents>[/]", "question": "[green]<The question in question>[/]"}
).to_messages()

rprint(example_messages[0].content)

# %%
# 1. Pregunta del user
question = input(f"panzaGPT [{MODEL}]>> ")

# 2. Obtener documentos reelevantes
retrieved_docs = vector_store.similarity_search(question)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 3. Pasarlos a junto con la pregunta al modelo
prompt_result = prompt.invoke({"question": question, "context": docs_content})

# 4. Generar una respuesta
answer = llm.invoke(prompt_result)

# %%
rprint(answer)

# %%
rprint(answer.content)

# %%
ans_struct = eval(answer.content)
rprint(ans_struct["thoughts"]["text"])

# %% [markdown]
# ### Obtenedores (*Retrievers*) y LangGraph

# %% [markdown]
# #### Retrievers

# %% [markdown]
# Los objetos que heredan de los [*Runnables*](https://python.langchain.com/api_reference/core/index.html#langchain-core-runnables) implementan un conjunto de métodos sincronos y asíncronos. Uno de estos objetos son los [*Retrievers*](https://python.langchain.com/api_reference/core/index.html#langchain-core-retrievers). Podemos obtener *retrievers* de los *vector_stores* o construir los propios.

# %% [markdown]
# #### LangGraph

# %% [markdown]
# LangGraph es un orquestador que nos permitirá administrar los pasos de obtención y generación. En general utilizar `langgraph` permite escalar nuestras apps y construir agentes de forma "sencilla". Algunas características interesantes son las siguientes:
#
# - Definir la lógica de la app una vez habilitar soporte para multiples modelos, llamadas async y batches
# - Perminte agregar fácilmente características como [persistencia](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints), [aprobación human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
#
# Para usar `langgraph` precisamos tres ingredientes:
#
# 1. Una forma de modelar los estados
# 2. Nodos por los que pasará
# 3. Un flujo de control

# %%
from typing import TypedDict

class State(TypedDict):
    """Define the states of the app"""
    question: str
    context: list[Document]
    answer: str


# %%
from langgraph.graph import START, StateGraph

# Define app steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# %%
# Define control flow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# Add a new node
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# %%
response = graph.invoke({"question": "What is a CRF?"})
rprint(eval(response["answer"])["thoughts"]["text"])

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ### Multiples formas de invocar nuestro grafo (gráfica (?))

# %%
for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
):
    rprint(f"{step}\n\n----------------\n")

# %%
for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="messages"
):
    print(message.content, end="")

# %% [markdown]
# ## Dando posibilidades conversacionales al RAG

# %% [markdown]
# Hoy día es muy popular interactuar con estos sistemas de Q&A a traves de una interfaz de chat conversacional. Esto es permitir que el usuario tenga una conversación de ida y vuelta con nuestro sistema. Esto implica que el sistema debe tener "memoria" para acceder a las preguntas y respuestas pasadas y cierta lógica para incorporar el historial para generar nuevas respuestas.
#
# Una forma modelar la interface conversacional es a traves de [mensajes](https://python.langchain.com/docs/concepts/messages/) con ciertos roles (user, IA, system), contenido y metadata. En particular los estados de nuestro RAG serán representado como secuencias de mensajes con las siguientes particularidades:
#
# 1. Entrada del usuario modelada como `HumanMessage`
# 2. La query que haremos al *vector store* como `AIMessage`
# 3. Los documentos reelevantes como `ToolMessage`
# 4. La respuesta final como `AIMessage`
#
# Este modelo de estadod viene integrado en LangGraph

# %%
from langgraph.graph import MessagesState, StateGraph

graph_msg_builder = StateGraph(MessagesState)

# %% [markdown]
# ### Tool calling

# %% [markdown]
# Permitir que se realizen llamadas a *tools* en la etapa de *retrieval* posibilitará que el modelo genera la query.

# %% [markdown]
# #### ❓ ¿Porqué sería reelevante hacer esto?

# %% [markdown]
# En una conversación puede que la query del usuario deba contextualizarse basandonos en el historial. Por ejemplo:
#
# ```
# User: ¿Qué es el mole?
#
# IA: El model es un platillo mexicano, a base de chocolate, pimienta, pan (bolillo), tortilla tostada, chiles, tomate, cebolla, clavo (especie), comino, nuez, almendra (no en todos los tipos de mole), laurel.
#
# User: ¿Cuál es la forma mas sencilla de prepararlo?
# ```
#
# En este caso, el modelo debería generar una query del esitlo: "formas sencillas de preparar mole". Habilitar llamadas a *tools* permite esta generación.

# %%
# Convirtiendo nuestro retrieve en una tool
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Retrieve information related to a user query"""
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# %% [markdown]
# Más sobre crear *tools* en la [docu](https://python.langchain.com/docs/how_to/custom_tools/). Acá explican porqué `response_format="content_and_artifact"`

# %% [markdown]
# El grafo consistirá en tres nodos:
#
# 1. Un nodo que procesa la entrada del usuario y genera una query para el *retriever* o responde directamente
# 2. Otro nodo para el *retriever tool* que ejecutará la obtención de los documentos reelevantes
# 3. El último nodo que genera la respuesta final utilizando el contexto del *retriever*
#
# Los elementos del grafo se definen acontinuación

# %%
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

# 1. Generamos un AIMessage que podría incluir la llamada a una tool
def query_or_respond(state: MessagesState):
    """Genera una tool call para retrieval o responde directo
    """
    llm_with_tools = llm.bind_tools([retrieve_tool])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# %%
# 2. Ejecutamos el paso de *retrieval*
tools = ToolNode([retrieve_tool])


# %%
# 3. Generamos la respuesta utilizando el contenido obtenido
def generate(state: MessagesState):
    """Genera una respuesta"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if not message.type == "tool":
            break
        recent_tool_messages.append(message)
    # Obtenemos los mensajes de tools en orden inverso
    tool_messages = recent_tool_messages[::-1]
    # Creando un prompt con los mensajes
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    convertation = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message)] + convertation

    response = llm.invoke(prompt)
    return {"messages": [response]}


# %% [markdown]
# #### Construyendo el grafo

# %% [markdown]
# Concentraremos los elementos definidos anteriormente en un solo objeto `graph`. Conectaremos los pasos en una secuencia y permitiremos que el primer paso `query_or_respond` realice un *short-circuit* y responda directamente en caso de no necesitar llamas a las *tools*. Esto permire que nuestro RAG brinde una experiencia conversacional más "natural", por ejemplo: respondiendo a saludos del usuario donde, en principio, no se requeriría ir a la base de datos vectorial.

# %%
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_msg_builder.add_node(query_or_respond)
graph_msg_builder.add_node(tools)
graph_msg_builder.add_node(generate)

graph_msg_builder.set_entry_point("query_or_respond")
graph_msg_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)

graph_msg_builder.add_edge("tools", "generate")
graph_msg_builder.add_edge("generate", END)

graph_tools = graph_msg_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph_tools.get_graph().draw_mermaid_png()))

# %% [markdown]
# ### Pruebas del RAG

# %%
input_message = "Hello"

for step in graph_tools.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# Vemos que no ejecutó el paso de *retriever*

# %%
input_message = "What is a CRF model?"

for step in graph_tools.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
