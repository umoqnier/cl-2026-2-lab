# -*- coding: utf-8 -*-
# %% [markdown]
# ### Práctica 2: Propiedades estadísticas del lenguaje y Diversidad
# #### 1. Verificación empírica de la Ley de Zipf
# Verificar si la ley de Zipf se cumple en los siguientes casos:
#
# 1.   En un lenguaje artificial creado por ustedes.
#
#
# *   Creen un script que genere un texto aleatorio seleccionando caracteres al azar de un alfabeto definido
#
#
#        *   Nota: Asegúrense de incluir el carácter de "espacio" en su alfabeto para que el texto se divida en "palabras" de longitudes variables.
#
#
# *   Obtengan las frecuencias de las palabras generadas para este texto artificial
#
#
# 2.   Alguna lengua de bajos recursos digitales (low-resourced language)
#
#
# *  Busca un corpus de libre acceso en alguna lengua de bajos recursos digitales
# *   Obten las frecuencias de sus palabras
#
# En ambos casos realiza lo siguiente:
#
#
# *   Estima el parámetro $\alpha$
#  que mejor se ajuste a la curva
# *   Generen las gráficas de rango vs. frecuencia (en escala y logarítmica).
#
#
#     *   Incluye la recta aproximada por $\alpha$
# *   ¿Se aproxima a la ley de Zipf? Justifiquen su respuesta comparándolo con el comportamiento del corpus visto en clase.
#
# [!TIP] Puedes utilizar los corpus del paquete py-elotl
#
#
#
#
#

# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter
#Aquí colocaré todas las librerías que use para el ejercicio número uno

# %%
#Creamos un minicorpus haciendo uso de random seed para poder replicar el como nos lo fabrica el código para la entrega
random.seed(42)
vocabulario=string.ascii_lowercase + " "#Definimos el vocabulario con ayuda de las letras conocidas a través de string

pesos = [1] * 26 + [5]#Asignamos pesos (probabilidades) a cada letra y al espacio
palabras = random.choices(vocabulario,weights=pesos,k=100000)#Formamos las palabras con random choices y formamos 100000 elementos entre espacios y palabras
corpus="".join(palabras)#Formamos el corpus con dichos 100000 elementos junto a los espacios seleccionados
lista=corpus.split()#Generamos una lista con el corpus generado
df=pd.DataFrame(lista,columns=["palabra"])#Llamamos df a la tabla de las palabras contenidas en el corpus
frecuencias=df['palabra'].value_counts().reset_index()
frecuencias.columns = ['Palabra', 'Frecuencia']
frecuencias['rank']=frecuencias['Palabra'].rank(ascending=False,method='first')
print(frecuencias.head(10))

# %%
print(corpus)

# %%
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title("Rango vs. Frecuencia")
plt.plot(frecuencias['rank'],frecuencias['Frecuencia'],color='blue')
plt.grid()
plt.subplot(1,2,2)
plt.title("Rango vs. Frecuencia (Log)")
plt.loglog(frecuencias['rank'],frecuencias['Frecuencia'],color='red')
plt.grid()
plt.show()

# %%
# !pip install datasets

# %% [markdown]
# ### Usando un corpus de elotl o de texto real
#
#

# %%
# !pip install elotl  
#Instalamos la paquetería de elotl

# %%
import elotl.corpus
print("Name\t\tDescription")
list_of_corpus = elotl.corpus.list_of_corpus()
for row in list_of_corpus:
    print(row)#Usamos la función proporciondad por el mismo elotl para ver el contenido de los corpus posibles

# %%
tsunkua = elotl.corpus.load('tsunkua')#Tomamos el del Otomí para este ejemplo
print(tsunkua)

# %%
from datasets import Dataset
#Vamos a separar cada una de las dos listas que nos ofrece el corpus al ser una lista de listas esto podría traer problemas
spanish = [entry[0] for entry in tsunkua if isinstance(entry, list) and len(entry) > 0]#Extraemos solo las del español
otomi =[entry[1] for entry in tsunkua if isinstance(entry,list)and len(entry)>0]#En esta solo extraemos las palabras en otomí
todo=spanish+otomi#Ahora hacemos una suma de todas las palabras del corpus
# Creamos el Dataset de Hugging Face donde cada fila tiene una cadena de texto en la columna 'text'
datatsunkua = Dataset.from_dict({"text": todo})
print(datatsunkua)

# %%
corpus_2 = datatsunkua.take(9000)#Tomamos 9000 elementos de los totales de nuestro dataset asignandolo a corpus
 #A partir de aquí usaremos las funciones vistas en clase

# %%
import re

def normalize_corpus(example):
    example["text"] = re.sub(r"[\W]", " ", example["text"])
    example["text"] = example["text"].lower()
    return example


# %%
from datasets.iterable_dataset import IterableDataset


def count_words(corpus: IterableDataset) -> Counter:
    word_counts = Counter()
    normalized_corpus = corpus.map(normalize_corpus)
    for row in normalized_corpus:
        text = row["text"]
        word_counts.update(text.split())
    return word_counts


# %%
words = count_words(corpus_2)#Con la conversión hecha podemos mapear las palabras del corpus

# %%
words.most_common(10)#Verificamos las 10 palabras más comunes

# %%
import pandas as pd


def counter_to_pandas(counter: Counter) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(counter, orient="index").reset_index()
    df.columns = ["word", "count"]
    df.sort_values("count", ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# %%
corpus_freqs = counter_to_pandas(words)

# %%
#Hacemois el plot como en clase de ambos en este corpus
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
corpus_freqs["count"].plot(loglog=True, legend=False)
plt.title("Ley de Zipf (log-log)")
plt.xlabel("Rango logaritmo")
plt.ylabel("Frecuencia logaritmo")
plt.grid()
plt.subplot(1,2,2)
corpus_freqs["count"].plot(legend=False,color='red', marker="o")
plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.grid()
plt.show()

# %% [markdown]
# ### ANÁLISIS
#
# A diferencia del lenguaje artificial (aleatorio) vemos un comportamiento casi similar al que vimos en clase con el corpus de elotl siguiendo un comportamiento similar al de cualquier corpus, mientras que el artifical no sigue dicha reglas con palabras formadas aleatoriamente y probabilidades también aleatorias donde todas las palabras tienen la probabilidad de aparecer por lo menos una vez y los espacios cada 5 de ellas
