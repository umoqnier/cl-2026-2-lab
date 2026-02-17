# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="76b3a996-772a-4be8-a8eb-f1e9ae67d03e" editable=true slideshow={"slide_type": "slide"}
# # 1. Niveles Lingüísticos

# + [markdown] editable=true slideshow={"slide_type": ""}
# <a target="_blank" href="https://colab.research.google.com/github/umoqnier/cl-2026-2-lab/blob/main/notebooks/1_niveles_linguisticos.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# + [markdown] id="615a09ab-2b52-440a-a4dc-fd8982c3c0e7" editable=true slideshow={"slide_type": "subslide"}
# ## Objetivos

# + [markdown] id="f034458c-9cb0-4966-a203-3145074c3fca" editable=true slideshow={"slide_type": ""}
# - Trabajar tareas a diferentes niveles lingüísticos (Fonético, Morfólogico, Sintáctico)
# - Manipular y recuper información de datasets disponibles en Github para resolver tareas de NLP
# - Comparar enfoques basados en reglas y estadísticos para el análisis morfológico

# + [markdown] id="3c169487-91d2-4afb-a12a-849c26a5be86" editable=true slideshow={"slide_type": "subslide"}
# ## Fonética y Fonología

# + [markdown] id="d0647e1e-a8c5-418f-81c7-31d2e86c88a4" editable=true slideshow={"slide_type": ""}
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/IPA_chart_2020.svg/660px-IPA_chart_2020.svg.png" width=500></center

# + editable=true slideshow={"slide_type": "subslide"}
from IPython.display import YouTubeVideo

# + editable=true slideshow={"slide_type": ""}
YouTubeVideo("DcNMCB-Gsn8", width=960, height=615)

# + editable=true slideshow={"slide_type": "subslide"}
YouTubeVideo("74nnLh0Vdcc", width=960, height=615)

# + editable=true slideshow={"slide_type": "subslide"}
import http
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests as r
from nltk.corpus import cess_esp
from rich import print as rprint
from rich.columns import Columns
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

# + [markdown] id="aa915e8e-038e-4de6-8956-f6221b1d8490" editable=true slideshow={"slide_type": "subslide"}
# ### International Phonetic Alphabet (IPA)

# + [markdown] id="09b4f076-b23b-46a8-9101-e37d79d374c8" editable=true slideshow={"slide_type": ""}
# - Las lenguas naturales tienen muchos sonidos diferentes por lo que necesitamos una forma de describirlos independientemente de las lenguas
# - IPA es una representación escrita de los [sonidos](https://www.ipachart.com/) del [habla](http://ipa-reader.xyz/)

# + [markdown] id="19eee353-6fd4-474a-86ca-8382ad51bf0f" editable=true slideshow={"slide_type": "subslide"}
# ### Dataset: [IPA-dict](https://github.com/open-dict-data/ipa-dict) de open-dict

# + [markdown] id="18f45a54-5f64-408e-98f3-fc31114dc84a" editable=true slideshow={"slide_type": "fragment"}
# - Diccionario de palabras para varios idiomas con su representación fonética
# - Representación simple, una palabra por renglon con el formato:
#
# ```
# [PALABRA][TAB][IPA]
#
# Ejemplos
# mariguana	/maɾiɣwana/
# zyuganov's   /ˈzjuɡɑnɑvz/, /ˈzuɡɑnɑvz/
# ```

# + [markdown] id="7cb52e47-d493-4b30-a991-ba5c4458d047" editable=true slideshow={"slide_type": "fragment"}
# - [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
# - URL: `https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/<iso-lang>`

# + [markdown] id="4152e020-fbc0-4ec5-8d51-ccd2e8a089fc" editable=true slideshow={"slide_type": "subslide"}
# #### Explorando el corpus 🗺️

# + id="25b595d7-7201-42bd-abb3-3acf9731d219" editable=true slideshow={"slide_type": "fragment"}
IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"


# + colab={"base_uri": "https://localhost:8080/"} id="3f45ba75-bbd3-4f13-8abf-b822fbf90dda" outputId="fa1dcad8-0851-4dad-eedd-49f1d91db7cb" editable=true slideshow={"slide_type": "fragment"}
# ¿Como empezamos?

# + [markdown] id="c671dbe4-1f99-443a-afb9-3f92951bef35" editable=true slideshow={"slide_type": "subslide"}
# #### Obtención y manipulación

# + id="1fdc23af-9a0b-470d-a5f1-e5bddfa0b53e" editable=true slideshow={"slide_type": "fragment"}
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
# + editable=true slideshow={"slide_type": "subslide"}



# + id="0a83a2a2-8e0e-4881-98f3-b9251a6be778" editable=true slideshow={"slide_type": "subslide"}
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
# + editable=true slideshow={"slide_type": "subslide"}



# + id="b834aaba-0716-41b5-935b-7f4a61e9da03" editable=true slideshow={"slide_type": "subslide"}
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

# + colab={"base_uri": "https://localhost:8080/"} id="d89e45e2-5010-4701-84fd-e62b910233e7" outputId="0839146b-ed76-4c0a-c7cf-078ac7278791" editable=true slideshow={"slide_type": "subslide"}



# + [markdown] id="37f69f04-ad55-4ca2-8bb1-d52fa58c4051" editable=true slideshow={"slide_type": "subslide"}
# #### Obtengamos datasets

# + colab={"base_uri": "https://localhost:8080/"} id="0828cb63-19b9-4cac-8df2-7dc8282fc4c3" outputId="763fe008-d810-4822-b1fe-826b784b988a" editable=true slideshow={"slide_type": "fragment"}
# Get datasets
dataset_es_mx = parse_response(download_ipa_corpus("es_MX"))
dataset_ja = parse_response(download_ipa_corpus("ja"))
dataset_en_us = parse_response(download_ipa_corpus("en_US"))
dataset_fr = parse_response(download_ipa_corpus("fr_FR"))

# + colab={"base_uri": "https://localhost:8080/"} id="694ef1c1-871e-407e-9ce5-5e177804f72f" outputId="f221b632-e77c-4649-e194-dbb49e2a644b" editable=true slideshow={"slide_type": "subslide"}
# Simple query
get_ipa_transcriptions("beautiful", dataset_en_us)

# + colab={"base_uri": "https://localhost:8080/"} id="04a5fdbb-3acc-4e34-9685-61a591f2b598" outputId="13068823-f467-4526-d9d9-32b3f5bcf340" editable=true slideshow={"slide_type": "fragment"}
# Examples
print(f"dog -> {get_ipa_transcriptions('dog', dataset_en_us)} 🐶")
print(f"mariguana -> {get_ipa_transcriptions('mariguana', dataset_es_mx)} 🪴")
print(f"猫 - > {get_ipa_transcriptions('猫', dataset_ja)} 🐈")
print(f"croissant -> {get_ipa_transcriptions('croissant', dataset_fr)} 🥐")

# + colab={"base_uri": "https://localhost:8080/"} id="9001ea35-855f-499b-a5b9-3c70a0ba7397" outputId="91960028-7806-4544-99ba-af5ea517c68e" editable=true slideshow={"slide_type": "fragment"}
# Diferentes formas de pronunciar
print(f"[es_MX] hotel | {dataset_es_mx['hotel']}")
print(f"[en_US] hotel | {dataset_en_us['hotel']}")

# + colab={"base_uri": "https://localhost:8080/"} id="9fc9c19c-f6f4-4e35-9155-71aacaef1a05" outputId="16852ef9-832b-42ca-9390-717c42104153" editable=true slideshow={"slide_type": "fragment"}
print(f"[ja] ホテル | {dataset_ja['ホテル']}")
print(f"[fr] hôtel | {dataset_fr['hôtel']}")

# + [markdown] id="41ca5bf8-93b1-4b10-9596-02bce9caccb8" editable=true slideshow={"slide_type": "subslide"}
# #### Obtener la distribución de frecuencias de los símbolos fonológicos para español

# + id="d616b9ef-396a-479b-adfd-1e15eab3fe37" editable=true slideshow={"slide_type": ""}



# + id="d3868dcd-eb1b-4e4a-9f52-ad7e2be72971"
def get_phone_symbols_freq(dataset: dict):
    freqs = defaultdict(int)
    ipas = [_.strip("/") for _ in dataset.values()]
    unique_ipas = set(ipas)
    for ipa in unique_ipas:
        for char in ipa:
            freqs[char] += 1
    return freqs


# + id="ijDBSOM5UB5i"
freqs_es = get_phone_symbols_freq(dataset_es_mx)
# Sorted by freq number (d[1]) descendent (reverse=True)
distribution_es = dict(sorted(freqs_es.items(), key=lambda d: d[1], reverse=True))
df_es = pd.DataFrame.from_dict(distribution_es, orient='index')
# -

df_es.head()

# + [markdown] id="aeb6c269-ad75-4e49-9090-247dc9a60231" editable=true slideshow={"slide_type": "slide"}
# #### 🧙🏼‍♂️ Ejercicio: Encontrar homófonos (palabras con el mismo sonido pero distinta ortografía) para el español
#
# - Ejemplos: Casa-Caza, Vaya-Valla

# + colab={"base_uri": "https://localhost:8080/"} id="-UXEnSv6700t" outputId="0e02caa7-93da-4e37-c304-f1b96073f44d" editable=true slideshow={"slide_type": "fragment"}
# Tu código bonito aquí ✨
# Hit: Use Counter please

# + [markdown] id="a6e06a95-ceb6-49c0-bcbb-ff456976e510" editable=true slideshow={"slide_type": "subslide"}
# #### Obteniendo todos los datos

# + id="9d92e8bd-53c9-4f2a-b926-b2de8ac19357" editable=true slideshow={"slide_type": "subslide"}
lang_codes = {
    "ar": "Arabic (Modern Standard)",
    "de": "German",
    "en_UK": "English (Received Pronunciation)",
    "en_US": "English (General American)",
    "eo": "Esperanto",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "fa": "Persian",
    "fi": "Finnish",
    "fr_FR": "French (France)",
    "fr_QC": "French (Québec)",
    "is": "Icelandic",
    "ja": "Japanese",
    "jam": "Jamaican Creole",
    "km": "Khmer",
    "ko": "Korean",
    "ma": "Malay (Malaysian and Indonesian)",
    "nb": "Norwegian Bokmål",
    "nl": "Dutch",
    "or": "Odia",
    "ro": "Romanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tts": "Isan",
    "vi_C": "Vietnamese (Central)",
    "vi_N": "Vietnamese (Northern)",
    "vi_S": "Vietnamese (Southern)",
    "yue": "Cantonese",
    "zh_hans": "Mandarin (Simplified)",
    "zh_hant": "Mandarin (Traditional)"
}
iso_lang_codes = list(lang_codes.keys())


# + id="aaf29cd0-be3c-4821-a608-71275da4852e" editable=true slideshow={"slide_type": "subslide"}
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


# + colab={"base_uri": "https://localhost:8080/"} id="f04bc065-584b-4373-bc7f-9e328793bbe4" outputId="9195f012-6ba8-4c12-b0ed-1ed2f419a266" editable=true slideshow={"slide_type": "fragment"}
corpora = get_corpora()


# + [markdown] id="41cc36cd-7919-4372-ada2-d9f52432b56c" editable=true slideshow={"slide_type": "subslide"}
# #### Sistema de búsqueda (naïve)

# + id="24424a3c-70df-416f-8923-6bcd2a435007" editable=true slideshow={"slide_type": "subslide"}
def get_formated_string(code: str, name: str):
    return f"[b]{name}[/b]\n[yellow]{code}"


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="bd09c65f-c559-4571-98a8-20c0d2c0308e" outputId="14a4e15e-43b2-4f83-8081-f5562967c8d1" editable=true slideshow={"slide_type": "fragment"}
rprint(Panel(Text("Representación fonética de palabras", style="bold", justify="center")))
rendable_langs = [Panel(get_formated_string(code, lang), expand=True) for code, lang in lang_codes.items()]
rprint(Columns(rendable_langs))

lang = input("lang>> ")
rprint(f"Selected language: {lang_codes[lang]}") if lang else rprint("Adios 👋🏼")
while lang:
    dataset = corpora[lang]
    query = input(f"  [{lang}]word>> ")
    results = get_ipa_transcriptions(query, dataset)
    rprint(query, " | ", ", ".join(results))
    while query:
        query = input(f"  [{lang}]word>> ")
        if query:
            results = get_ipa_transcriptions(query, dataset)
            rprint(query, " | ", ", ".join(results))
    lang = input("lang>> ")
    rprint(f"Selected language: [yellow]{lang_codes[lang]}[/]") if lang else rprint("Adios 👋🏼")

# + [markdown] id="dc8b18ff-9d70-49a6-98c1-7feb7d0c268a" editable=true slideshow={"slide_type": "slide"}
# #### 👩‍🔬 Ejercicio: *[Ortographic Depth](https://en.wikipedia.org/wiki/Orthographic_depth)*
#
# Algunas lenguas se escriben como suenan y otras no. Calcula el promedio de los *ratios* (`len(chars) / len(phones)`) para todas las lenguas en el corpus. La salida debe verse más o menos así:
#
# ```
# Φ Spanish (Spain) = ??
# Φ Spanish (Mexico) = ??
# Φ English (General American) = ??
# ```
#
# Dónde `??` deberá mostrar el ratio (`Φ`) calculado
# + editable=true slideshow={"slide_type": "fragment"}
# Tu código bonito aquí ✨


# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ##### ¿Pregunta?: ¿Qué lengua usarías si quieres reducir costos en un LLM?

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo: Fabricando rimas

# + id="a4116005-8fb4-454d-a2ab-e2f083eda000" editable=true slideshow={"slide_type": "fragment"}
def get_rhyming_patterns(sentence: str, dataset: dict) -> dict[str, list]:
    words = sentence.split()
    word_ipa_map = {}
    for word in words:
        ipa_transcriptions = get_ipa_transcriptions(word, dataset)
        # Remove "/" char from transcriptions
        word_ipa_map.update({word: [_.strip("/") for _ in ipa_transcriptions]})

    rhyming_patterns = defaultdict(list)
    for word, ipas in word_ipa_map.items():
        for ipa in ipas:
            # Getting last 2 elements of the ipa representation
            pattern = ipa[-2:]
            rhyming_patterns[pattern].append(word)
    return rhyming_patterns


# + editable=true slideshow={"slide_type": "fragment"}
def display_rhyming_patterns(patterns: dict[str, list]) -> None:
    for pattern, words in patterns.items():
        if len(set(words)) > 1:
            print(f"{pattern}:: {', '.join(words)}")


# + [markdown] id="f4daacb3-544b-4b3a-ab77-23b2fc4dd07e" editable=true slideshow={"slide_type": "subslide"}
# #### Testing

# + [markdown] id="3HoQEf8i8qTo" editable=true slideshow={"slide_type": "fragment"}
# ```
# ɣo:: juego, fuego
# on:: con, corazón
# ʎa:: brilla, orilla
# ```

# + colab={"base_uri": "https://localhost:8080/"} id="057bb91d-5bef-47b4-ba82-42559f457c2b" outputId="0435b8cf-4395-438b-a78f-0f872b9cb287" editable=true slideshow={"slide_type": "fragment"}
sentence = "If you drop the ball it will fall on the doll"

dataset = data.get("en_US")
rhyming_words = get_rhyming_patterns(sentence, dataset)
display_rhyming_patterns(rhyming_words)

# + [markdown] id="86f07e0f-bfe4-4a14-be9d-a7b0b7661260" editable=true slideshow={"slide_type": "subslide"}
# #### Material extra (fonética)

# + colab={"base_uri": "https://localhost:8080/"} id="64dc5e71-449b-4fdf-a3f4-d0f1776c5bbf" outputId="8b70a454-65b1-447d-aee3-73b1933b134b" editable=true slideshow={"slide_type": ""}
# apt-get install -y espeak
# !sudo pacman -S espeak-ng

# + id="c87a1b9d-848c-488f-b2e4-1782f07bc557" editable=true slideshow={"slide_type": ""}
# !espeak --help

# + [markdown] id="9fc31a40-1d6e-4c56-b07e-74a0c47a89c4" editable=true slideshow={"slide_type": "slide"}
# ## Morfología

# + [markdown] id="GJ10fzsXvFSS" editable=true slideshow={"slide_type": ""}
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Flexi%C3%B3nGato-svg.svg/800px-Flexi%C3%B3nGato-svg.svg.png" width=200></center>
#
# > De <a href="//commons.wikimedia.org/wiki/User:KES47" class="mw-redirect" title="User:KES47">KES47</a> - <a href="//commons.wikimedia.org/wiki/File:Flexi%C3%B3nGato.png" title="File:FlexiónGato.png">File:FlexiónGato.png</a> y <a href="//commons.wikimedia.org/wiki/File:Nuvola_apps_package_toys_svg.svg" title="File:Nuvola apps package toys svg.svg">File:Nuvola apps package toys svg.svg</a>, <a href="http://www.gnu.org/licenses/lgpl.html" title="GNU Lesser General Public License">LGPL</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=27305101">Enlace</a>

# + [markdown] id="9677e8f1-aa3e-4f9c-8c7c-1422ea9ca913" editable=true slideshow={"slide_type": "subslide"}
# El análisis morfológico es la determinación de las partes que componen la palabra y su representación lingüística, es una especie de etiquetado
#
# Los elementos morfológicos son analizados para:
#
# - Determinar la función morfológica de las palabras
# - Hacer filtrado y pre-procesamiento de text

# + [markdown] id="d962a952-6fa8-4410-82b6-8e7c6ba2f7e4" editable=true slideshow={"slide_type": "subslide"}
# ### Análisis morfológico basado en reglas

# + [markdown] id="cf70678d-f1d1-401e-aa23-da90a2ea7eaa" editable=true slideshow={"slide_type": "fragment"}
# Recordemos que podemos hacer un analizador morfológico haciendo uso de un transductor que vaya leyendo y haciendo transformaciones en una cadena. Formalmente:
#
# * $Q = \{q_0, \ldots, q_T\}$ conjunto finito de estados.
# * $\Sigma$ es un alfabeto de entrada.
# * $q_0 \in Q$ es el estado inicial.
# * $F \subseteq Q$ es el conjunto de estados finales.
#
# Un transductor es una 6-tupla $T = (Q, \Sigma, \Delta, q_0, F, \sigma)$ tal que
#
# * $\Delta$ es un alfabeto de salida teminal
# * $\Sigma$ es un alfabeto de entrada no terminal
# * $\sigma: Q \times \Sigma \times \Delta \longrightarrow Q$ función de transducción

# + [markdown] id="cdf136a8-6a9d-4434-b533-191553db242b" editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo: Parsing con expresiones regulares

# + [markdown] id="605d64c5-e102-4972-b5f6-92e0c08b495b" editable=true slideshow={"slide_type": "fragment"}
# Con fines de prácticidad vamos a _imitar_ el comportamiento de un transductor utilizando el modulo de python `re`

# + [markdown] id="cb409f23-1456-40bd-8fdd-997a862ad190" editable=true slideshow={"slide_type": "fragment"}
# La estructura del sustantivo en español es:
#
# ` BASE+AFIJOS (marcas flexivas)   --> Base+DIM+GEN+NUM`

# + id="1e2959df-96a4-40f5-a932-108abad269be" editable=true slideshow={"slide_type": "subslide"}
palabras = [
    'niño',
    'niños',
    'niñas',
    'niñitos',
    'gato',
    'gatos',
    'gatitos',
    'perritos',
    'paloma',
    'palomita',
    'palomas',
    'flores',
    'flor',
    'florecita',
    'lápiz',
    'lápices',
    # 'chiquitititititos',
    #'curriculum', # curricula
    #'campus', # campi
]


# + id="66599e2e-d67b-49e2-9f80-0a20e756ca19" editable=true slideshow={"slide_type": "subslide"}
def morph_parser_rules(words: list[str]) -> list[str]:
    """Aplica reglas morfológicas a una lista de palabras para realizar
    un análisis morfológico.

    Parameters:
    ----------
    words : list of str
        Lista de palabras a las que se les aplicarán las reglas morfológicas.

    Returns:
    -------
    list of str
        Una lista de palabras después de aplicar las reglas morfológicas.
    """

    # Lista para guardar las palabras parseadas
    morph_parsing = []

    # Reglas que capturan ciertos morfemas
    # {ecita, itos, as, os}
    for w in words:
        # ecit -> DIM
        R0 = re.sub(r"([^ ]+)ecit([a|o|as|os])", r"\1-DIM\2", w)
        # it -> DIM
        R1 = re.sub(r"([^ ]+)it([a|o|as|os])", r"\1-DIM\2", R0)
        # a(s) -> FEM
        R2 = re.sub(r"([^ ]+)a(s)", r"\1-FEM\2", R1)
        # a -> FEM
        R3 = re.sub(r"([^ ]+)a\b", r"\1-FEM", R2)
        # o(s) -> MSC
        R4 = re.sub(r"([^ ]+)o(s)", r"\1-MSC\2", R3)
        # o .> MSC
        R5 = re.sub(r"([^ ]+)o\b", r"\1-MSC", R4)
        # es -> PL
        R6 = re.sub(r"([^ ]+)es\b", r"\1-PL", R5)
        # s -> PL
        R7 = re.sub(r"([^ ]+)s\b", r"\1-PL", R6)
        # Sustituye la c por z cuando es necesario
        parse = re.sub(r"c-", r"z-", R7)

        # Guarda los parseos
        morph_parsing.append(parse)
    return morph_parsing


# + editable=true slideshow={"slide_type": "subslide"}
def prettify_tags(word: str) -> str:
    tags = {
        "DIM": "[b yellow]DIM[/]",
        "FEM": "[b green]FEM[/]",
        "MSC": "[b magenta]MSC[/]",
        "PL": "[b blue]PL[/]",
    }
    for tag, pretty_tag in tags.items():
        word = word.replace(tag, pretty_tag)
    return word


# + colab={"base_uri": "https://localhost:8080/"} id="85c9648c-0c09-4d19-b52b-f020943caf5a" outputId="6804fc5f-edb7-423c-9bf3-96ec7fccc605" editable=true slideshow={"slide_type": "subslide"}
morph_parsing = morph_parser_rules(palabras)
for palabra, parseo in zip(palabras, morph_parsing):
    rprint(palabra, "-->", prettify_tags(parseo))

# + [markdown] id="3a104dae-e815-4271-9ff2-96802d50df9e" editable=true slideshow={"slide_type": "fragment"}
# #### Preguntas 🤔
# - ¿Qué pasa con las reglas en lenguas donde son más comunes los prefijos y no los sufijos?
# - ¿Cómo podríamos identificar características de las lenguas?

# + [markdown] id="8b3d6dac-6b02-45db-b352-a549a25fdabe" editable=true slideshow={"slide_type": "subslide"}
# #### Herramientas para hacer sistemas de análisis morfológico basados en reglas

# + [markdown] id="2ff2d85d-64c1-4272-9d6d-cf3599094588" editable=true slideshow={"slide_type": ""}
# - [Apertium](https://en.wikipedia.org/wiki/Apertium)
# - [Foma](https://github.com/mhulden/foma/tree/master)
# - [Helsinki Finite-State Technology](https://hfst.github.io/)
# - Ejemplo [proyecto](https://github.com/apertium/apertium-yua) de analizador morfológico de Maya Yucateco
# - Ejemplo normalizador ortográfico del [Náhuatl](https://github.com/ElotlMX/py-elotl/tree/master)
#
#
# También se pueden utilizar diferentes métodos de aprendizaje de máquina para realizar análisis/generación morfológica. En los últimos años ha habido un shared task de [morphological reinflection](https://github.com/sigmorphon/2023InflectionST) para poner a competir diferentes métodos

# + [markdown] id="d5b878ce-f60a-4069-b618-fcb0d4e77256" editable=true slideshow={"slide_type": "subslide"}
# ### Segmentación morfológica

# + [markdown] id="bcdc6126-3d69-4f90-9cfe-9dd6845525d5" editable=true slideshow={"slide_type": "fragment"}
# #### Corpus: [SIGMORPHON 2022 Shared Task on Morpheme Segmentation](https://github.com/sigmorphon/2022SegmentationST/tree/main)

# + [markdown] id="78102f5e-2bcc-41cd-981f-514d786cb9be" editable=true slideshow={"slide_type": "fragment"}
# - Shared task donde se buscaba convertir las palabras en una secuencia de morfemas
# - Dividido en dos partes:
#     - Segmentación a nivel de palabras (nos enfocaremos en esta)
#     - Segmentación a nivel oraciones

# + [markdown] id="c6b0c352-e6ac-49db-9b3a-74da7db6ddad" editable=true slideshow={"slide_type": "subslide"}
# #### Track: words
#
# | word class | Description                      | English example (input ==> output)     |
# |------------|----------------------------------|----------------------------------------|
# | 100        | Inflection only                  | played ==> play @@ed                   |
# | 010        | Derivation only                  | player ==> play @@er                   |
# | 101        | Inflection and Compound          | wheelbands ==> wheel @@band @@s        |
# | 000        | Root words                       | progress ==> progress                  |
# | 011        | Derivation and Compound          | tankbuster ==> tank @@bust @@er        |
# | 110        | Inflection and Derivation        | urbanizes ==> urban @@ize @@s          |
# | 001        | Compound only                    | hotpot ==> hot @@pot                   |
# | 111        | Inflection, Derivation, Compound | trackworkers ==> track @@work @@er @@s

# + [markdown] id="84e2235d-f919-4c5d-832e-93551ba9ac4b" editable=true slideshow={"slide_type": "subslide"}
# #### Explorando el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="1a59cbf7-de9d-4618-8229-5cd369fa1c07" outputId="740f54b5-52b8-4665-863a-0969ec99fd86" editable=true slideshow={"slide_type": "fragment"}
response = r.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")
response.text[:100]

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="17faa0d5-2d5f-4035-bf65-c0e5b256aa0f" outputId="29c6287e-391d-4346-f843-874766bb1a9c" editable=true slideshow={"slide_type": "fragment"}
raw_data = response.text.split("\n")
raw_data[-2]

# + colab={"base_uri": "https://localhost:8080/"} id="a1814873-5c65-4f34-9c59-01ccdf130089" outputId="80ca7d49-9053-4b41-a383-e1fc40749cb5" editable=true slideshow={"slide_type": "fragment"}
element = raw_data[2].split("\t")
element

# + colab={"base_uri": "https://localhost:8080/"} id="99c1dfb3-a1ad-4a78-b26b-8a560894f058" outputId="da0bb22f-5ea6-4e65-e8fe-a8acfd829fc2" editable=true slideshow={"slide_type": "fragment"}
element[1].split()

# + id="bb87e719-41c2-4e22-b9bf-5303902be165" editable=true slideshow={"slide_type": "subslide"}
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
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Root",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound"
}


# + id="721ea2fa-24d8-4035-b0a5-87c86f821c6d" editable=true slideshow={"slide_type": "subslide"}
def get_track_files(lang: str, track: str = "word") -> list[str]:
    """Genera una lista de nombres de archivo del shared task

    Con base en el idioma y el track obtiene el nombre de los archivos
    para con información reelevante para hacer análisis estadístico.
    Esto es archivos .test y .dev

    Parameters:
    ----------
    lang : str
        Idioma para el cual se generarán los nombres de archivo.
    track : str, optional
        Track del shared task de donde vienen los datos (por defecto es "word").

    Returns:
    -------
    list[str]
        Una lista de nombres de archivo generados para el idioma y la pista especificados.
    """
    return [
        f"{lang}.{track}.test.gold",
        f"{lang}.{track}.dev",
    ]

# + editable=true slideshow={"slide_type": ""}


# + id="f583e168-1f5d-4426-9789-5fac8b2b221c" editable=true slideshow={"slide_type": "subslide"}
def get_raw_corpus(files: list) -> list:
    """Descarga y concatena los datos de los archivos tsv desde una URL base.

    Parameters:
    ----------
    files : list
        Lista de nombres de archivos (sin extensión) que se descargarán
        y concatenarán.

    Returns:
    -------
    list
        Una lista que contiene los contenidos descargados y concatenados
        de los archivos tsv.
    """
    result = []
    for file in files:
        print(f"Downloading {file}.tsv", end=" ")
        response = r.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv")
        print(f"status={response.status_code}")
        lines = response.text.split("\n")
        result.extend(lines[:-1])
    return result

# + editable=true slideshow={"slide_type": ""}


# + id="54de3b3d-be08-437d-b4ee-ff55d4fee2a9" editable=true slideshow={"slide_type": "subslide"}
def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """Convierte una lista de datos de corpus en un DataFrame

    Parameters:
    ----------
    corpus_list : list
        Lista de líneas del corpus a convertir en DataFrame.
    lang : str
        Idioma al que pertenecen los datos del corpus.

    Returns:
    -------
    pd.DataFrame
        Un DataFrame de pandas que contiene los datos del corpus procesados.
    """
    data_list = []
    for line in corpus_list:
        try:
            word, tagged_data, category = line.split("\t")
        except ValueError:
            # Caso donde no existe la categoria
            word, tagged_data = line.split("\t")
            category = "NOT_FOUND"
        morphemes = tagged_data.split()
        data_list.append(
            {"words": word, "morph": morphemes, "category": category, "lang": lang}
        )
    df = pd.DataFrame(data_list)
    df["word_len"] = df["words"].apply(lambda x: len(x))
    df["morph_count"] = df["morph"].apply(lambda x: len(x))
    return df

# + editable=true slideshow={"slide_type": ""}


# + colab={"base_uri": "https://localhost:8080/"} id="fe645a77-f8ca-4bf1-b11e-2274b78ba6d1" outputId="d7e50464-31a2-4979-ca32-1ecdb7c07e5c" editable=true slideshow={"slide_type": "subslide"}
files = get_track_files("spa")
raw_spa = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_spa, lang="spa")

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="774a3b39-3a37-43d5-afb4-5d98154afe9e" outputId="c99841e0-ea76-4343-9cca-0390ee68910c" editable=true slideshow={"slide_type": "fragment"}
df.head(20)

# + [markdown] id="0ffef737-bd71-43ed-be6d-a357819ab7c8" editable=true slideshow={"slide_type": "subslide"}
# #### Análisis cuantitativo para el Español

# + colab={"base_uri": "https://localhost:8080/", "height": 384} id="e02c54b1-8b19-4a37-8a64-37749beb0418" outputId="979c36e3-eace-4040-f56a-c749ad5e0fb3" editable=true slideshow={"slide_type": "fragment"}
print("Total unique words:", len(df["words"].unique()))
df["category"].value_counts().head(30)

# + colab={"base_uri": "https://localhost:8080/"} id="39965a92-4719-4043-919b-3b99dca0b8f9" outputId="bac4b6a4-efa9-40ce-ac86-51939ec0d6bd" editable=true slideshow={"slide_type": "fragment"}
df["word_len"].mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="baf02d64-9bc1-4135-9f05-e523728ba269" outputId="d9774960-1726-44e7-cb31-d7061716f52f" editable=true slideshow={"slide_type": "fragment"}
plt.hist(df['word_len'], bins=10, edgecolor='black')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.title('Word Length Distribution')
plt.show()


# + id="c902ecf4-889a-4082-b4b8-0cb89ba9b16c" editable=true slideshow={"slide_type": "subslide"}
def plot_histogram(df, kind, lang):
    """Genera un histograma de frecuencia para una columna específica
    en un DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos para generar el histograma.
    kind : str
        Nombre de la columna para la cual se generará el histograma.
    lang : str
        Idioma asociado a los datos.

    Returns:
    -------
    None
        Esta función muestra el histograma usando matplotlib.
    """
    counts = df[kind].value_counts().head(30)
    plt.bar(counts.index, counts.values)
    plt.xlabel(kind)
    plt.ylabel('Frequency')
    plt.title(f'{kind} Frequency Graph for {lang}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 487} id="1f36d2ce-cca3-49f0-b989-69dae98f9646" outputId="b3648c42-99eb-452c-fb85-d73218829014" editable=true slideshow={"slide_type": "subslide"}
plot_histogram(df, "category", "spa")


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo 🥑: Adivina, adivinador, probabilidad del tipo de morfemas
#
# - No todos los afijos son iguales. Algunos sirven para conjugar (Flexivos) y otros para crear conceptos nuevos (Derivativos).
# - Aprovechando que el dataset de SIGMORPHON nos dice si una palabra es de tipo `Inflection` (100) o `Derivation` (010) calculemos qué tan probable es que un morfema sea flexivo o derivativo.
#

# + editable=true slideshow={"slide_type": "subslide"}
def analyze_morpheme_types(df: pd.DataFrame) -> pd.DataFrame:
    inflection_counts = defaultdict(int)
    derivation_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for _, row in df.iterrows():
        morphemes = row['morph'] 
        category = row['category']

        # Con base en SIGMORPHON:
        # Pos 0: Inflection, Pos 1: Derivation
        is_inflection = category[0] == '1'
        is_derivation = category[1] == '1'

        for morph in morphemes:
            clean_morph = morph.replace("@@", "")

            # Ignoramos morfemas muy cortos o vacíos
            if len(clean_morph) < 2:
                continue

            total_counts[clean_morph] += 1
            if is_inflection:
                inflection_counts[clean_morph] += 1
            if is_derivation:
                derivation_counts[clean_morph] += 1

    results = []
    for morph, total in total_counts.items():
        if total > 5:
            p_inflection = inflection_counts[morph] / total
            p_derivation = derivation_counts[morph] / total
            results.append({
                'morfema': morph,
                'total_freq': total,
                'prob_flexion': round(p_inflection, 3),
                'prob_derivacion': round(p_derivation, 3)
            })

    return pd.DataFrame(results).sort_values(by='total_freq', ascending=False)


# + editable=true slideshow={"slide_type": "subslide"}
morph_stats = analyze_morpheme_types(df)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Los morfemas más "Gramaticales" (Flexivos)
#
# Deberíamos ver terminaciones verbales y plurales.

# + editable=true slideshow={"slide_type": "fragment"}
rprint("Top Morfemas Flexivos:")
rprint(morph_stats.sort_values(by=['prob_flexion', "total_freq"], ascending=False).head(10))

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Los morfemas más "Léxicos" (Derivativos)
#
# Deberíamos ver sufijos que cambian sustantivos a adjetivos, etc.

# + editable=true slideshow={"slide_type": "fragment"}
rprint("Top Morfemas Derivativos:")
rprint(morph_stats.sort_values(by=['prob_derivacion', "total_freq"], ascending=False).head(10))

# + [markdown] id="408683e9-5d9b-44a6-bcb5-d2ad29bf7903" editable=true slideshow={"slide_type": "subslide"}
# #### Morfosintaxis

# + [markdown] id="10233e8c-b359-4ad7-ab6f-8d6d0c600a39" editable=true slideshow={"slide_type": "fragment"}
# - Etiquetas que hacen explícita la funcion gramatical de las palabras en una oración
# - Determina la función de la palabra dentro la oración (por ello se le llama Partes del Discurso)
# - Se le conoce tambien como **Análisis morfosintáctico**: es el puente entre la estructura de las palabras y la sintaxis
# - Permiten el desarrollo de herramientas de NLP más avanzadas
# - El etiquetado es una tarea que se puede abordar con técnicas secuenciales, por ejemplo, HMMs, CRFs, Redes neuronales

# + [markdown] id="8db57418-b7bc-4dc1-972d-aef484e9ea48" editable=true slideshow={"slide_type": ""}
# <center><img src="https://byteiota.com/wp-content/uploads/2021/01/POS-Tagging.jpg" height=500 width=500></center

# + [markdown] id="628dd2cd-c0b4-4b12-aa08-b74e5a81579c" editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo
#
# > El gato negro rie malvadamente
#
# - El - DET
# - gato - NOUN
# - negro - ADJ
# - ríe - VER
#
# <center><img src="https://i.pinimg.com/originals/0e/f1/30/0ef130b255ea704625b2ad473701dee5.gif"></center

# + [markdown] id="522e2222-af00-4315-82ce-11534116f0b8" editable=true slideshow={"slide_type": "subslide"}
# ### Etiquetado POS usando Conditional Random Fields (CRFs)

# + [markdown] id="2276eafd-a3e4-48a8-b97a-359503a7d66f" editable=true slideshow={"slide_type": "fragment"}
# - Modelo de gráficas **no dirigido**. Generaliza los *HMM*
#     - Adiós a la *Markov assuption*
#     - Podemos tener cualquier dependencia que queramos entre nodos
#     - Nos enfocaremos en un tipo en concreto: *LinearChain-CRFs* ¡¿Por?!
#
# <center><img width=300 src="https://i.kym-cdn.com/entries/icons/original/000/032/676/Unlimited_Power_Banner.jpg"></center>
#

# + [markdown] id="c5a1bff5-1f06-416b-9244-c4eab4dd989a" editable=true slideshow={"slide_type": "subslide"}
# - Modela la probabilidad **condicional** $P(Y|X)$
#     - Modelo discriminativo
#     - Probabilidad de un estado oculto dada **toda** la secuecia de entrada
# ![homer](https://media.tenor.com/ul0qAKNUm2kAAAAd/hiding-meme.gif)

# + [markdown] id="74beab61-39ec-43bf-8ca8-44cbb9d62149" editable=true slideshow={"slide_type": "subslide"}
# - Captura mayor **número de dependencias** entre las palabras y captura más características
#     - Estas se definen en las *feature functions* 🙀
# - El entrenamiento se realiza aplicando gradiente decendente y optimización con algoritmos como [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
#
#
# <center><img src="https://iameo.github.io/images/gradient-descent-400.gif"></center>
#

# + [markdown] id="4653baf1-edc2-4813-be95-643a1b0f60f7" editable=true slideshow={"slide_type": "subslide"}
# $P(\overrightarrow{y}|\overrightarrow{x}) = \frac{1}{Z} \displaystyle\prod_{i=1}^N exp\{w^T ⋅ \phi(y_{i-1}, y_i, \overrightarrow{x}, i)\}$
#
# Donde:
# - $\overrightarrow{y}$ = Etiquetas POS
# - $\overrightarrow{x}$ = Palabras en una oración
# - $w^T$ = Vector de pesos a aprender
# - $\phi$ = Vector de *Features*
#     - Calculado con base en un conjunto de *feature functions*
# - $i$ = la posición actual en la oración
# - $Z$ = factor de normalización

# + [markdown] id="5c13a366-4bc3-425c-8db4-bb2986dc2e8f" editable=true slideshow={"slide_type": "subslide"}
# ![](https://aman.ai/primers/ai/assets/conditional-random-fields/Conditional_Random_Fields.png)
#
# Tomado de http://www.davidsbatista.net/blog/2017/11/13/Conditional_Random_Fields/

# + [markdown] id="5ad8c1dc-4c5b-41f8-95c2-40be89f8f07f" editable=true slideshow={"slide_type": "subslide"}
# #### Feature functions
#
# $\phi(y_{i-1}, y_i, \overrightarrow{x}, i)$
#
# - Parte fundamental de los CRFs
# - Cuatro argumentos:
#     - Todos los datos observables $\overrightarrow{x}$ (conectar $x$ con cualquier $y$)
#     - El estado oculto anterior $y_{i-1}$
#     - El estado oculto actual $y_i$
#     - El index del timestamp $i$
#         - Cada feature list puede tener diferentes formas

# + [markdown] id="c1cfd87f-d4d2-4501-bd47-5a7dc843db2b" editable=true slideshow={"slide_type": "subslide"}
# - Aquí es donde está la flexibilidad del modelo
# - Tantas *features* como queramos, las que consideremos que pueden ayudar a que el modelo tenga un mejor desempeño
#     - Íntimamente ligadas a la lengua. Para mejor desempeño se debe hacer un análisis de sus características.
# - Ejemplo:
#
# ```python
# [
#     "word.lower()",
#     "EOS",
#     "BOS",
#     "pre-word",
#     "nxt-word",
#     "word-position",
#     ...
# ]
# ```

# + [markdown] id="1c50c45f-9e37-42e0-9cfe-1a78429660ed" editable=true slideshow={"slide_type": "subslide"}
# ### Implementación de CRFs

# + colab={"base_uri": "https://localhost:8080/"} id="ca05605b-7776-4fc8-9e6a-7541a28d659f" outputId="bf87417f-828e-499d-98e6-50f438c9dc9a" editable=true slideshow={"slide_type": "fragment"}
# !uv add nltk scikit-learn sklearn-crfsuite

# + [markdown] id="b2e0fb29-beee-4a0a-8010-0f62db206981" editable=true slideshow={"slide_type": "subslide"}
# #### Obteniendo otro corpus más

# + colab={"base_uri": "https://localhost:8080/"} id="5651f7bc-21de-4379-93b8-07499f6df74a" outputId="86fd3488-9802-4fca-b9fd-241091a3b405" editable=true slideshow={"slide_type": "fragment"}
# Descargando el corpus cess_esp: https://www.nltk.org/book/ch02.html#tab-corpora
nltk.download('cess_esp')

# + id="1ab197a8-2ea3-4d85-acf8-745d6f99be83" editable=true slideshow={"slide_type": "fragment"}
# Cargando oraciones
corpora = cess_esp.tagged_sents()

# + colab={"base_uri": "https://localhost:8080/"} id="We80idF3qUIb" outputId="57544344-2f95-44f4-e8b7-486c7b6631b3" editable=true slideshow={"slide_type": "fragment"}
rprint(corpora[1])


# + id="45d7a052-6e9e-47ca-8fc2-13ff05963b53" editable=true slideshow={"slide_type": "subslide"}
def get_tags_map() -> dict:
    tags_raw = r.get(
        "https://gist.githubusercontent.com/vitojph/39c52c709a9aff2d1d24588aba7f8155/raw/af2d83bc4c2a7e2e6dbb01bd0a10a23a3a21a551/universal_tagset-ES.map"
    ).text.split("\n")
    tags_map = {line.split("\t")[0].lower(): line.split("\t")[1] for line in tags_raw}
    return tags_map


def map_tag(tag: str, tags_map=get_tags_map()) -> str:
    return tags_map.get(tag.lower(), "N/F")


def parse_tags(corpora: list[list[tuple]]) -> list[list[tuple]]:
    result = []
    for sentence in corpora:
        result.append([(word, map_tag(tag)) for word, tag in sentence])
    return result


# + id="c07cc1fb-8b6d-4a3d-a637-93c15bcbbfc6" editable=true slideshow={"slide_type": "subslide"}
corpora = parse_tags(corpora)

# + colab={"base_uri": "https://localhost:8080/"} id="d2347079-1504-458b-9743-5bc4dfa7d1e6" outputId="b4a40eb1-b9ff-423c-914f-b315097e7ee6" editable=true slideshow={"slide_type": ""}
rprint(corpora[0])


# + [markdown] id="d7864024-2275-4cf6-93f4-16c6b0f54451" editable=true slideshow={"slide_type": "subslide"}
# #### Feature lists

# + id="2b6ab1f8-71fc-4862-ac2a-35fcb3ca5b2a" editable=true slideshow={"slide_type": ""}
def word_to_features(sent, i):
    word = sent[i][0]
    features = {
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "prefix_1": word[:1],
        "prefix_2": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word_len": len(word),
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update(
            {
                "prev_word.lower()": prev_word.lower(),
                "prev_word.istitle()": prev_word.istitle(),
            }
        )
    else:
        # Beginning of sentence
        features["BOS"] = True
    return features


# Extract features and labels
def sent_to_features(sent) -> list:
    return [word_to_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent) -> list:
    return [label for token, label in sent]


# + colab={"base_uri": "https://localhost:8080/"} id="59da4b3c-7d1d-4725-936c-0afdbf74137f" outputId="8c9c0967-e2d2-4681-849b-dc4a9796bfff" editable=true slideshow={"slide_type": "subslide"}
# ¿Cuantas oraciones tenemos disponibles?
len(corpora)

# + id="7266b709-0287-4f48-a62f-12ca9d7aaffb" editable=true slideshow={"slide_type": "fragment"}
# Preparando datos para el CRF
X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in corpora]
y = [[pos for _, pos in sent] for sent in corpora]

# + colab={"base_uri": "https://localhost:8080/"} id="f7f68f41-8ad8-42c4-b20d-626561047433" outputId="884befed-36dd-465f-ff62-ee8fee034187" editable=true slideshow={"slide_type": "fragment"}
# Exploración de data estructurada
rprint(X[0])

# + id="e6bdf707-c7fc-408b-9d81-65528e068cbb" editable=true slideshow={"slide_type": "subslide"}
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# + id="49503189-368e-4d83-abd2-b06433bae3e8" editable=true slideshow={"slide_type": "fragment"}
assert len(X_train) + len(X_test) == len(corpora), "Something wrong with my split :("
assert len(y_train) + len(y_test) == len(corpora), "Something wrong with my split :("

# + colab={"base_uri": "https://localhost:8080/"} id="cba343a2-482f-4c67-a842-704ab5fc6f3e" outputId="3355fa0f-d2f3-4c66-de0b-595c8b754495" editable=true slideshow={"slide_type": "subslide"}
# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True,
)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

# + colab={"base_uri": "https://localhost:8080/"} id="983f2a29-b455-4ca3-8115-eb4962e25481" outputId="d59aff27-781e-4b97-f262-7f39845c7e88" editable=true slideshow={"slide_type": "subslide"}
y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

# Evaluate the model
report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat)
rprint(report)

# + [markdown] id="1266180c-54ca-433c-bf77-e7052df67291" editable=true slideshow={"slide_type": "slide"}
# # Tarea 1: Exploración de Niveles del lenguaje 🔭

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### FECHA DE ENTREGA: 2 de Marzo 2026 at 11:59pm

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### Fonética
#
# 1. Con base en el sistema de búsqueda visto en clase que recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario.
#     - ¿Cómo devolver o **aproximar** su transcripción fonológica?
#     - Reutiliza el sistema de búsqueda visto en clase y mejóralo con esta funcionalidad.
#     - Muestra al menos tres ejemplos

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### Morfología
#
# 2. Elige tres lenguas del corpus que pertenezcan a familias lingüísticas distintas
#    - Ejemplo: `spa` (Romance), `eng` (Germánica), `hun` (Urálica)
#    - Para cada una de las tres lenguas calcula y compara:
#        -  **Ratio morfemas / palabra**: El promedio de morfemas que componen las palabras
#         -  **Indicé de Flexión / Derivación**: Del total de morfemas, ¿Qué porcentaje son etiquetas de flexión (`100`) y cuáles de derivación (`010`)?
# 3. Visualización
#     - Genera una figura con **subplots** para comparar las lenguas lado a lado.
#     - *Plot 1*: Distribución de la longitud de los morfemas
#     - *Plot 2*: Distribución de las categorías (flexión, derivación, raíz, etc.)
# 4. Con base en esta información, responde la pregunta: *¿Cuál de las tres lenguas se comporta más como una lengua aglutinante y cuál como una lengua aislante?*
#     - Justifica tu respuesta usando tus métricas y figuras

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### EXTRA:
#
# - Genera la [matriz de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) para el etiquetador CRFs visto en clase
# - Observando las etiquetas donde el modelo falló responde las preguntas:
#     - ¿Por qué crees que se confundió?
#     - ¿Es un problema de ambigüedad léxica (la palabra tiene múltiples etiquetas)?
#     - ¿Qué *features* añadirías para solucionarlo?
