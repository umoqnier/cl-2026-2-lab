# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cl-2026-2-lab
#     language: python
#     name: python3
# ---

# ## Sección I: Fonética. 
#
# ### ¿Qué es la fonética?
# La fonética estudia la producción y la percepción de los sonidos o fonos, a esta rama de la lingüística le interesan cómo suenan las vocales y consonantes del español, además se preocupa por las cualidades fisiológicas, por lo que tiene una estrecha relación con ciencias como la anatomía, la fisiología y la acústica

# ### Ejercicio 1:
#
#
# 1. Con base en el sistema de búsqueda visto en la [práctica 1](https://github.com/umoqnier/cl-2026-2-lab/blob/main/notebooks/1_niveles_linguisticos_I.ipynb), dónde se recibe una palabra ortográfica y devuelve sus transcripciones fonológicas, proponga una solución para los casos en que la palabra buscada no se encuentra en el lexicón/diccionario.
#     - ¿Cómo devolver o **aproximar** su transcripción fonológica?
#     - Reutiliza el sistema de búsqueda visto en clase y mejóralo con esta funcionalidad.
#     - Muestra al menos tres ejemplos
#     
#
# ### Solucion:
#
# Mi solucion consiste en volver el sistema mas robusto. El sistema original solo puede devolver ersultados si la palabra coincide caracter por caracter con el diccionario. Para mi solucion utilice la libreria difflib, de modo que si una palabra no es encontrada tal cual en el dataset de es_MX entonces se recorre las llaves del diccionario calculando la similitud de caracteres, se selecciona la palabra con mayor puntaje de similitud (con un umbral de 0.6 para reducir casos absurdos) y se devuelve la representacion IPA de la palabra seleccionada.

# +
import requests as r
import http
import difflib  # para calcular similitudes entre cadenas

def download_ipa_corpus(iso_lang: str) -> str:
    IPA_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt"
    response = r.get(IPA_URL.format(lang=iso_lang))
    if response.status_code != http.HTTPStatus.OK:
        print(f"ERROR on {iso_lang} :(")
        return ""
    return response.text

def parse_response(response: str) -> dict:
    ipa_list = response.rstrip().split("\n")
    result = {}
    for item in ipa_list:
        if item == '': continue
        item_list = item.split("\t")
        result[item_list[0]] = item_list[1]
    return result

def get_ipa_transcriptions_improved(word: str, dataset: dict) -> list[str]:
    """
    Busca una palabra en el dataset. Si no existe, busca la palabra 
    ortográficamente más cercana y devuelve su transcripción.
    """
    word = word.lower()
    transcription = dataset.get(word)
    
    if not transcription:
        matches = difflib.get_close_matches(word, dataset.keys(), n=1, cutoff=0.6)
        if matches:
            closest_word = matches[0]
            transcription = dataset.get(closest_word)
            print(f"-> '{word}' no encontrada. Aproximando mediante: '{closest_word}'")
        else:
            return ["No se encontró una aproximación cercana"]
            
    return transcription.split(", ")

#  EJECUCIÓN Y EJEMPLOS
es_mx_data = parse_response(download_ipa_corpus("es_MX"))

# Ejemplos de prueba (Palabras con errores o variantes)
test_words = ["pagaritoz", "bankueta", "pelotta"]

print("|---> Resultados de Búsqueda Fonética <---|")
for w in test_words:
    res = get_ipa_transcriptions_improved(w, es_mx_data)
    print(f"Original: {w} | IPA: {res}\n")
# -

# ## Sección II: Morfología
#
# ### ¿Qué es la Morfología?
# La morfología​ es la rama de la lingüística que estudia la estructura interna de las palabras para definir y clasificar sus unidades: las variantes de las palabras y la formación de nuevas palabras. Analiza la estructura de las palabras y partes de palabras, tales como tema, palabras raíz, prefijos y sufijos.

# ### Ejercicio 2:
#
# 2. Elige tres lenguas del corpus que pertenezcan a familias lingüísticas distintas
#    - Ejemplo: `spa` (Romance), `eng` (Germánica), `hun` (Urálica)
#    - Para cada una de las tres lenguas calcula y compara:
#        -  **Ratio morfemas / palabra**: El promedio de morfemas que componen las palabras
#         -  **Indicé de Flexión / Derivación**: Del total de morfemas, ¿Qué porcentaje son etiquetas de flexión (`100`) y cuáles de derivación (`010`)?

# Las lenguas que seleccione son: 
# -   Portugues , de la familia romance.
# -   Ruso, de la familia eslava.
# -   Turco, de la familia túrquicia.

# +
import pandas as pd
import requests as r

# 1. Definición de lenguas y descarga de datos
# Familias: Romance (por), Eslava (rus), Túrquica (tur)
langs_to_process = ["por", "rus", "tur"]

def get_raw_corpus(lang: str) -> list:
    url = f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{lang}.word.test.gold.tsv"
    response = r.get(url)
    return response.text.split("\n")[:-1]

# 2. Función para procesar y calcular métricas individualmente
def calculate_morph_metrics(lang_code: str):
    raw_data = get_raw_corpus(lang_code)
    
    data_list = []
    for line in raw_data:
        parts = line.split("\t")
        if len(parts) >= 3:
            word, segments, category = parts[0], parts[1], parts[2]
            morphemes = segments.split()
            data_list.append({
                "word": word,
                "morph_count": len(morphemes),
                "category": category
            })
    
    df = pd.DataFrame(data_list)
    
    # Cálculo de métricas
    ratio_morfemas = df["morph_count"].mean()
    
    # Porcentajes basados en etiquetas SIGMORPHON (100=Flexión, 010=Derivación)
    #
    total = len(df)
    perc_flexion = (len(df[df["category"] == "100"]) / total) * 100
    perc_derivacion = (len(df[df["category"] == "010"]) / total) * 100
    
    return {
        "Ratio": round(ratio_morfemas, 3),
        "Flexión (%)": round(perc_flexion, 2),
        "Derivación (%)": round(perc_derivacion, 2)
    }

# 3. Ejecución y presentación de resultados
resultados = {lang: calculate_morph_metrics(lang) for lang in langs_to_process}

print(f"{'Lengua':<12} | {'Ratio M/P':<10} | {'Flexión %':<10} | {'Derivación %':<12}")
print("-" * 55)
for lang, m in resultados.items():
    print(f"{lang:<12} | {m['Ratio']:<10} | {m['Flexión (%)']:<10} | {m['Derivación (%)']:<12}")


# +
def calculate_metrics(df, name):
    ratio = df["morph_count"].mean()
    # Contamos categorías basadas en las etiquetas SIGMORPHON
    flexion = len(df[df["category"] == "100"]) / len(df) * 100
    derivacion = len(df[df["category"] == "010"]) / len(df) * 100
    
    print(f"--- Métricas para {name} ---")
    print(f"Ratio Morfemas/Palabra: {ratio:.2f}")
    print(f"Índice de Flexión (100): {flexion:.2f}%")
    print(f"Índice de Derivación (010): {derivacion:.2f}%\n")

calculate_metrics(df_por, "Portugués")
calculate_metrics(df_rus, "Ruso")
calculate_metrics(df_tur, "Turco")
