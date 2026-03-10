# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### FONÉTICA
# #### 1. Con base en el sistema de búsqueda visto en la práctica 1, dónde se recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario.
# ##### *¿Cómo devolver o aproximar su transcripción fonológica?
# A mi se me ocurrió una forma, la de analizar y relacionar ciertos tipos de sílabas o letras conocidas y colocar como definición su fonema si es que existe, así podríamos acercarnos a el sonido fonético real, creo que es una solución rápida y un poco simple pero que podría apoyarnos en la obtención de palabras con sílabas pocos comunes o nuevas que no se encuentren en el corpus o API seleccionado.
#
# Esta idea nació tras experimentar por un par de días con ciertas palabras que descubrí no están dentro del API que usamos de ejemplo en clase, palabras como luis o francisco palabras comunes de nombres propios para nosotros, sin embargo conteniendo palabras como ñoño o luís (si, con acento).
#
# Por ejemplo, si queremos que tenga palabras no existentes en el corpus podríamos indicar que siempre de la forma fonética de las letras para que cualquier palabra pueda ser escrita en abse a las reglas de la fonética y pronunciación de nuestro idioma para cada vocal y consonante
# ##### *Reutiliza el sistema de búsqueda visto en clase y mejóralo con esta funcionalidad.
# Los ejemplos utilizados serán:
# Luis
# Francisco
# frijol
#
# Además de colocar como comentarios dentro de los códigos los cambios propuestos.

# %%
import http
from collections import defaultdict

import pandas as pd
import requests as r

from rich import print as rprint
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text #Estás son las librerias y funciones necesarias ya definidas que iremos usando

# %%
IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"
#Exportamos el ipa que vimos en clase

# %%
#Empezamos con las definiciones como en clase
response = r.get(IPA_URL.format(lang="es_MX"))
ipa_list = response.text[:1000].split("\n")
ipa_list[-1].split("\t")


# %%
def get_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict

    Given a word this function return the IPA transcriptions

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code

    Returns
    -------
    list[str]:
        List with posible transcriptions if any,
        else an empty list
    """
    return dataset.get(word.lower(), "").split(", ")
    


# %%
def download_ipa_corpus(iso_lang: str) -> str:
    """Get ipa-dict file from Github

    Parameters:
    -----------
    iso_lang:
        Language as iso code

    Results:
    --------
    dict:
        Dictionary with words as keys and phonetic representation
        as values for a given lang code
    """
    print(f"Downloading {iso_lang}", end="::")
    response = r.get(IPA_URL.format(lang=iso_lang))
    status_code = response.status_code
    print(f"status={status_code}")
    if status_code != http.HTTPStatus.OK:
        print(f"ERROR on {iso_lang} :(")
        return ""
    return response.text


# %%
def parse_response(response: str) -> dict:
    """Parse text response from ipa-dict to python dict

    Each row have the format:
    [WORD][TAB]/[IPA]/(, /[IPA]/)?

    Parameters
    ----------
    response: str
        ipa-dict raw text

    Returns
    -------
    dict:
        A dictionary with the word as key and the phonetic
        representations as value
    """
    ipa_list = response.rstrip().split("\n")
    result = {}
    for item in ipa_list:
        if item == '':
            continue
        item_list = item.split("\t")
        result[item_list[0]] = item_list[1]
    return result
    


# %%
lang_codes = {    
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",#En mi caso decidi reducirlo a español tanto europeo como mexicano
}
iso_lang_codes = list(lang_codes.keys())

def get_corpora() -> dict:
    """Download corpora from ipa-dict github

    Given a list of iso lang codes download available datasets.

    Returns
    -------
    dict
        Lang codes as keys and dictionary with words-transcriptions
        as values
    """
    return {
        code: parse_response(download_ipa_corpus(code))
         for code in iso_lang_codes
        }

corpora = get_corpora()


# %%
def get_formated_string(code: str, name: str):
    return f"[b]{name}[/b]\n[yellow]{code}"


# %%
rprint(
    Panel(Text("Representación fonética de palabras", style="bold", justify="center"))
)
rendable_langs = [
    Panel(get_formated_string(code, lang), expand=True)
    for code, lang in lang_codes.items()
]
rprint(Columns(rendable_langs))

lang = input("lang>> ")
rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")
while lang:
    dataset = corpora[lang]
    query = input(f"  [{lang}]word>> ")
    results = get_ipa_transcriptions(query, dataset)
    print(query, " | ", ", ".join(results))
    while query:
        query = input(f"  [{lang}]word>> ")
        if query:
            results = get_ipa_transcriptions(query, dataset)
            if not results:
                reglas = {'i':'í','v': 'b', 'z': 's', 'ce': 'se', 'ci': 'si', 'y': 'i', 'h': ''}
                aprox=query.lower()
                for letra, sonido in reglas.items():
                    aprox=aprox.replace(letra, sonido)
                results = [f"/{aprox}/"]
            rprint(query, " | ", ", ".join(results))
    lang = input("lang>> ")
    rprint(f"Selected language: [yellow]{lang_codes[lang]}[/]") if lang else rprint(
        "Adios 👋🏼"
    )
