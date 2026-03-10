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
#     display_name: P1 (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 1 - Lingüística Computacional
#

# %% [markdown]
# # Fonética

# %%
from collections import defaultdict

import pandas as pd
import requests

# %%
IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"

# %%
response = requests.get(IPA_URL.format(lang="es_MX"))

# %%
ipa_list = response.text.split("\n")

# %%
ipa_list[0].split("\t")


# %%
def download_ipa_corpus(iso_lang: str) -> str:
    """
    Descarga el archivo del diccionario IPA para el idioma dado.
    """
    print(f"Descargando {iso_lang}...", end=" ")
    response = requests.get(IPA_URL.format(lang=iso_lang))
    print(f"status={response.status_code}")
    
    if response.status_code != 200:
        print(f"Error al descargar el corpus para {iso_lang}")
        return ""
    
    return response.text


# %%
def parse_response(response: str) -> dict:
    """
    Convierte el texto crudo del diccionario IPA en un diccionario de Python.
    Formato esperado por línea: palabra[TAB]ipa
    """
    ipa_list = response.rstrip().split("\n")
    result = {}
    
    for item in ipa_list:
        if item == "":
            continue
        
        item_list = item.split("\t")
        
        if len(item_list) == 2:
            word, ipa = item_list
            result[word] = ipa
    
    return result


# %%
es_data = parse_response(download_ipa_corpus("es_MX"))


# %%
def get_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """
    Busca una palabra en el diccionario y devuelve sus transcripciones IPA.
    Si no existe, devuelve lista vacía.
    """
    return dataset.get(word.lower(), "").split(", ") if dataset.get(word.lower(), "") else []


# %%
get_ipa_transcriptions("mayonesa", es_data)


# %%
def levenshtein(s1: str, s2: str) -> int:
    """
    Calcula la distancia de Levenshtein entre dos cadenas.
    """
    m = len(s1)
    n = len(s2)

    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                costo = 0
            else:
                costo = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + costo
            )

    return dp[m][n]


# %%
def encontrar_palabra_mas_cercana(palabra: str, dataset: dict) -> str:
    """
    Encuentra la palabra más cercana en el diccionario usando distancia de Levenshtein.
    """

    palabra_mas_cercana = None
    distancia_minima = float("inf")

    for palabra_dic in dataset.keys():
        distancia = levenshtein(palabra.lower(), palabra_dic.lower())

        if distancia < distancia_minima:
            distancia_minima = distancia
            palabra_mas_cercana = palabra_dic

    return palabra_mas_cercana


# %%
def obtener_ipa_aproximado(palabra: str, dataset: dict) -> dict:
    """
    Busca la transcripción IPA de una palabra.
    Si no existe en el diccionario, aproxima usando la palabra más cercana.
    """

    resultado_exacto = get_ipa_transcriptions(palabra, dataset)

    if resultado_exacto:
        return {
            "encontrada": True,
            "palabra_ingresada": palabra,
            "palabra_coincidente": palabra.lower(),
            "transcripciones": resultado_exacto
        }

    palabra_cercana = encontrar_palabra_mas_cercana(palabra, dataset)
    transcripcion_aprox = get_ipa_transcriptions(palabra_cercana, dataset)

    return {
        "encontrada": False,
        "palabra_ingresada": palabra,
        "palabra_coincidente": palabra_cercana,
        "transcripciones": transcripcion_aprox
    }


# %%
from pprint import pprint

# %%
print(obtener_ipa_aproximado("tomate", es_data))
print(obtener_ipa_aproximado("cassa", es_data))
print(obtener_ipa_aproximado("cucharacha", es_data))

# %% [markdown]
# # Morfología

# %%
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import requests as r

# %%
# Lenguas disponibles en el corpus
LANGS = {
    "ces": "Czech",
    "eng": "English",
    "fra": "French",
    "hun": "Hungarian",
    "spa": "Spanish",
    "ita": "Italian",
    "lat": "Latin",
    "rus": "Russian",
}

# Categorías morfológicas del corpus
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Root",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound",
}


# %%
def get_track_files(lang: str, track: str = "word") -> list[str]:
    """
    Genera los nombres de archivo del corpus SIGMORPHON
    según el idioma y el track.

    Parameters
    ----------
    lang : str
        Código de idioma (ej. "spa", "eng", "hun").

    track : str
        Tipo de datos del shared task (por defecto "word").

    Returns
    -------
    list[str]
        Lista de archivos del corpus para ese idioma.
    """

    return [
        f"{lang}.{track}.test.gold",
        f"{lang}.{track}.dev",
    ]


# %%
def get_raw_corpus(files: list) -> list:
    """
    Descarga y concatena archivos TSV del corpus SIGMORPHON.

    Parameters
    ----------
    files : list
        Lista de nombres de archivos a descargar.

    Returns
    -------
    list
        Lista de líneas del corpus descargado.
    """

    result = []

    for file in files:
        print(f"Downloading {file}.tsv", end=" ")
        response = r.get(
            f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv"
        )
        print(f"status={response.status_code}")

        lines = response.text.split("\n")
        result.extend(lines[:-1])

    return result


# %%
def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """
    Convierte el corpus en un DataFrame para análisis.
    """

    data_list = []

    for line in corpus_list:

        try:
            word, tagged_data, category = line.split("\t")

        except ValueError:
            word, tagged_data = line.split("\t")
            category = "NOT_FOUND"

        morphemes = tagged_data.split()

        data_list.append(
            {
                "words": word,
                "morph": morphemes,
                "category": category,
                "lang": lang,
            }
        )

    df = pd.DataFrame(data_list)

    df["word_len"] = df["words"].apply(lambda word: len(word))
    df["morph_count"] = df["morph"].apply(lambda m: len(m))

    return df


# %%
# Lenguas que analizaremos
langs = ["spa", "rus", "hun"]

# %%
dfs = []

for lang in langs:
    print(f"\nProcesando {lang} ({LANGS[lang]})")

    files = get_track_files(lang)
    raw = get_raw_corpus(files)

    df_lang = raw_corpus_to_dataframe(raw, lang=lang)
    dfs.append(df_lang)

df_all = pd.concat(dfs)

df_all.head()

# %%
ratio = df_all.groupby("lang")["morph_count"].mean()

print("Ratio morfemas/palabra:")
print(ratio)

# %%
inflection = (df_all["category"] == "100").groupby(df_all["lang"]).mean()

print("Porcentaje de flexión (100):")
print(inflection)

# %%
derivation = (df_all["category"] == "010").groupby(df_all["lang"]).mean()

print("Porcentaje de derivación (010):")
print(derivation)
