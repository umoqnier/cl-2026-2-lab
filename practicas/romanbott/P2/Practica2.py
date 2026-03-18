# ---
# jupyter:
#   jupytext:
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
# # Práctica 2: Propiedades estadísticas del lenguaje y diversidad

# %% [markdown]
# ## Verificación empírica de la ley de Zipf

# %% [markdown]
# Crearemos un lenguaje artificial de la siguiente forma:
# + Definimos un conjunto de "vocales"
# + El complemento serán las "consonantes"
# + Generamos caracteres aleatorios de la siguiente forma:
#   + Con probabilidad $.1$ se genera un "espacio en blanco" lo que determina el fin de la palabra
#   + En caso que no se genere un espacio en blanco, con probabilidad $.5$ tomamos una muestra aleatoria del conjunto de vocales, y en otro caso del conjunto de consonantes.

# %%
from random import Random

# %%
from string import ascii_lowercase

# %%
rng = Random()

# %%
VOC = "gjhwo"
CONS = [x for x in ascii_lowercase if x not in VOC]

def generate_word():
    """
    Generates a pseudo-random word by sampling from vowels and consonants.

    The function constructs a word by repeatedly appending a random character 
    from either a vowel (VOC) or consonant (CONS) collection. The process 
    continues as long as a random roll is greater than 0.1, ensuring the 
    result is at least one character long.

    Returns:
        str: A randomly generated string of characters.
    """
    word = []
    while len(word) == 0:
        while rng.random() > .1:
            if rng.random() > .5:
                word.append(rng.choice(VOC))
            else:
                word.append(rng.choice(CONS))
    
    return "".join(word)



# %%
for _ in range(5):
    print(generate_word())

# %% [markdown]
# Obtenemos un "corpus" sintético generando millones de palabras aleatorias.
# Como únicamente nos interesa analizar las frecuencias de las palabras obtenidas, los resultados los recopilamos con un `Counter`.
#
# Utilizaremos `multiprocessing` para usar varios hilos simultáneamente.

# %%
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def worker_task(iterations):
    local_counter = Counter()
    for _ in range(iterations):
        local_counter[generate_word()] += 1
    return local_counter

def run_parallel(total_iterations, num_processes):
    chunk_size = total_iterations // num_processes
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(worker_task, [chunk_size] * num_processes)
    
    final_counter = Counter()
    for res in results:
        final_counter.update(res)
    return final_counter



# %%
NUM_TOKENS = 10**6 # Puede ser 10**7 aunque no cambia mucho el resultado
CORES = max(multiprocessing.cpu_count() - 2, 2)
synthetic_corpus_freq = run_parallel(NUM_TOKENS, CORES)

# %% [markdown]
# Utilizaremos `linalg.lstsq` de `numpy` para ajustar una función lineal en el plano $\log - \log$ a las frecuencias obtenidas del corpus sintético.

# %%
import numpy as np

def prepare_data(counter_obj, top_k):
    data = counter_obj.most_common(top_k)
    
    ranks = np.arange(1, len(data) + 1)
    counts = np.array([item[1] for item in data])
    
    x = np.log(ranks).reshape(-1, 1)
    y = np.log(counts)
    
    return x, y

def fit_linear(counter_obj, top_k):
    x, y = prepare_data(counter_obj, top_k)
    
    X_mat = np.column_stack((x, np.ones(len(x))))

    coeffs, residuals, rank, s = np.linalg.lstsq(X_mat, y, rcond=None)
    
    slope, intercept = coeffs
    return slope, intercept


# %% [markdown]
# Dado que los datos son escasos en la parte inicial de la gráfica, tomamos únicamente los primeros 500 tipos para la regresión:

# %%
m_synth, b_synth = fit_linear(synthetic_corpus_freq, 500)

# %% [markdown]
# Utilizaremos la siguiente función para graficar la distribución de las palabras generadas en escala log-log.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.set_theme()


# %%
def plot_zipf(counter: Counter, m: float, b: float, 
              top_k: int | None = None, 
              title: str = "Análisis de la Ley de Zipf", 
              ax: plt.Axes | None = None):
    """
    Plots the frequency distribution (Zipf's Law) and its linear fit on a log-log scale.

    Args:
        counter (Counter): Object containing the frequencies of the elements.
        m (float): Slope of the linear model (coefficient).
        b (float): Intercept of the linear model.
        top_k (int, optional): Number of most frequent elements to plot.
        title (str): Title of the plot.
        ax (plt.Axes, optional): Matplotlib axis to plot on. 
                                 If None, a new figure is created.

    Returns:
        plt.Axes: The axis object with the generated plot.
    """
    
    data = counter.most_common(top_k) if top_k else counter.most_common()
    df = pd.DataFrame(data, columns=["Palabra", "Frecuencia"])
    df["Rank"] = np.arange(1, len(df) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        data=df,
        x="Rank",
        y="Frecuencia",
        ax=ax,
        marker='o',
        markersize=3,
        markeredgewidth=0,
    )

    # Plot Regression Line
    # The linear fit was log(y) = m * log(x) + b
    # Therefore: y = exp(b) * x^m
    x_range = df["Rank"].values
    y_fit = np.exp(b) * (x_range ** m)
    
    ax.plot(x_range, y_fit, color='gray', linestyle='--', linewidth=2,
            label=f'Ajuste lineal:\n $y = e^{{{b:.2f}}} \\cdot x^{{{m:.2f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Rango (Log)")
    ax.set_ylabel("Frecuencia (Log)")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    return ax


# %%
_ = plot_zipf(synthetic_corpus_freq, m_synth, b_synth, 10000, "Ley de Zipf en un corpus sintético")

# %% [markdown]
# Compararemos este resultado con los corpus de `elotl`.

# %%
# !pip install elotl

# %%
import elotl.corpus as ec

# %%
elotl_corpora = {
    c: ec.load(c) for c in map(lambda x: x[0], ec.list_of_corpus())
}

# %% [markdown]
# Mostremos algunas métricas sencillas de los corpus:

# %%
import itertools
def get_corpus_size(corpus) -> int:
    return len(corpus)

def get_corpus_tokens(corpus) -> int:
    return sum(len(elem[1].split()) for elem in corpus)

def get_corpus_types(corpus) -> int:
    unique_words = set(itertools.chain.from_iterable(elem[1].split() for elem in corpus))
    return len(unique_words)
    
rows = [
    {
        "corpus": name,
        "size": get_corpus_size(obj),
        "tokens": get_corpus_tokens(obj),
        "types": get_corpus_types(obj),
    }
    for name, obj in elotl_corpora.items()
]

corpora_metrics = pd.DataFrame(rows).set_index("corpus")

# %%
corpora_metrics


# %%
def get_corpus_freqs(corpus) -> Counter:
    return Counter(itertools.chain.from_iterable(elem[1].split() for elem in corpus))


# %%
corpora = {}
for name, corpus in elotl_corpora.items():
    freqs = get_corpus_freqs(corpus)
    m, b = fit_linear(freqs, 500)
    corpora[f"Corpus {name}"] = (freqs, m, b)

# %%
corpora["Corpus sintético"] = (
    synthetic_corpus_freq, *fit_linear(synthetic_corpus_freq, 500)
)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (name, (freqs, m, b)) in enumerate(corpora.items()):
    
    plot_zipf(freqs, m, b, top_k=1000, title=name, ax=axes.flatten()[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# Podemos ver que aunque todos se pueden ajustar con una línea de pendiente aproximadamente $-1$, en el caso del corpus sintético, el comportamiento en las primeras palabras es atípico, pues presenta una distribución casi uniforme. Esto se debe a que la función que usamos para generar palabras no tiene forma de privilegiar un monograma sobre otro, salvo la distinción entre vocales y consonantes. De igual forma, al menos en principio, la frecuencia de cualquier bigrama solo debería depender en cuántas vocales tiene.

# %% [markdown]
# ## Diversidad lingüistica de México
#
# Utilizaremos la base de datos de lenguajes de glottolog:

# %%
import requests as r

# %%
import io
import zipfile

def get_dataframe_from_url(url: str) -> pd.DataFrame:
    """
    Downloads a file from a URL, unzips if necessary, and returns a 
    pandas DataFrame if the content is a CSV.
    """
    response = r.get(url)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download. Status code: {response.status_code}")

    buffer = io.BytesIO(response.content)
    
    if zipfile.is_zipfile(buffer):
        with zipfile.ZipFile(buffer) as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                raise ValueError("The zip archive does not contain any CSV files.")
            
            with z.open(csv_files[0]) as csv_file:
                return pd.read_csv(csv_file)
    
    if url.lower().endswith('.csv'):
        buffer.seek(0)
        return pd.read_csv(buffer)
    else:
        raise ValueError("The file is not a zip archive and does not have a .csv extension.")


# %%
glottolog_languoid_df = get_dataframe_from_url("https://cdstar.eva.mpg.de//bitstreams/EAEA0-608B-9919-A962-0/glottolog_languoid.csv.zip")

# %%
glott_languages = glottolog_languoid_df[glottolog_languoid_df.level == "language"]

# %%
glott_languages


# %%
def filter_by_extremal_points(extremal, df):
    lat_min = df["latitude"] >  extremal["meridional"]["lat"]
    lat_max = df["latitude"] <  extremal["septentrional"]["lat"]
    lon_min = df["longitude"] > extremal["occidental"]["lon"]
    lon_max = df["longitude"] < extremal["oriental"]["lon"]
    return df[lat_min & lat_max & lon_min & lon_max]


# %%
mexico_extreme_points = {
    "septentrional": {"lat": 32.6333, "lon": -114.7500},
    "occidental": {"lat": 32.5333, "lon": -117.0833},
    "meridional": {"lat": 14.5408, "lon": -92.2167},
    "oriental": {"lat": 21.1333, "lon": -86.7333}
}

# %%
glotto_mex = filter_by_extremal_points(mexico_extreme_points, glott_languages)

# %%
glotto_mex

# %%
import plotly.express as px

# %%
fig = px.scatter_map(
    glotto_mex, 
    lat="latitude", 
    lon="longitude", 
    hover_name="name",
    color="family_id",
    zoom=4,
    center={"lat": 23.6, "lon": -102.5},
    height=600
)

fig.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
china_extreme_points = {
    "septentrional": {"lat": 53.5600, "lon": 123.2500},
    "occidental": {"lat": 39.3833, "lon": 73.5000},
    "meridional": {"lat": 18.1500, "lon": 109.5000},
    "oriental": {"lat": 48.3333, "lon": 134.7667}
}

# %%
glotto_china = filter_by_extremal_points(china_extreme_points, glott_languages)

# %%
fig = px.scatter_map(
    glotto_china, 
    lat="latitude", 
    lon="longitude", 
    hover_name="name",
    color="family_id",
    zoom=4,
    center={"lat": 28, "lon": 115},
    height=600,
)

fig.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
mongolia_extreme_points = {
    "septentrional": {"lat": 52.1500, "lon": 98.9500},
    "occidental": {"lat": 48.8833, "lon": 87.7333},
    "meridional": {"lat": 41.5833, "lon": 105.0000},
    "oriental": {"lat": 46.7167, "lon": 119.9833}
}

# %%
glotto_mongolia = filter_by_extremal_points(mongolia_extreme_points, glott_languages)

# %%
glotto_mongolia

# %%
fig = px.scatter_map(
    glotto_mongolia, 
    lat="latitude", 
    lon="longitude", 
    hover_name="name",
    color="family_id",
    zoom=4,
    center={"lat": 48, "lon": 105},
    height=600
)

fig.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# ### ¿Qué tanta diversidad lingüística hay en México con respecto a otras regiones?
#
# En el mapa podemos apreciar una gran cantidad de lenguas, que pertenecen a una variedad de familias. Quizá en diversidad lingüistica México sería comparable a China, una país con aproximadamente 10 veces más población.
# En cambio al comparar con Mongolia podemos ver pocas familias lingüisticas y pocos lenguas.
# Es por esto que podríamos decir que México tiene una gran diversidad lingüistica.
#
#
# ### ¿Cuál es la zona que dirias que tiene mayor diversidad en México?
#
# La región de Oaxaca presenta una gran densidad de lenguas.
