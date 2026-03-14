#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
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
# # P2: Ley de Pzif
# Veremos si un idioma artificial creado aleatoreamente cumple con la ley de Pzif, y veremos como se comporta una lengua con pocos recursos digitales.
# %%
# Importamos las librerías necesarias
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# %% [markdown]
# ## Generando un idioma artificial.
# %%
# Definimos el alfabeto y la longitud de las palabras (con un espacio)
alfabeto = "abcdefghijklmnñopqrstuvwxyz "
letras_sueltas = np.random.choice(list(alfabeto), size=5000000, replace=True)
chorizo_de_letras = "".join(letras_sueltas)
texto_artificial = chorizo_de_letras.split()


# %%
# Longitud de nuestro texto artificial, 
len(texto_artificial)

# %%
conteo = Counter(texto_artificial)

# %%
df = pd.DataFrame(conteo.items(), columns = ["Palabras","Frecuencia"])

# %%
# DataFrame ordenado 
df_ord = df.sort_values("Frecuencia", ascending = False,).copy()
df_ord["Rango"] = range(1,len(df_ord)+1)
df_ord.head(20)

# %%
plt.plot(df_ord["Rango"],df_ord["Frecuencia"],marker = "o")
plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %%
plt.loglog(df_ord["Rango"], df_ord["Frecuencia"], marker="o")
plt.title("Ley de Zipf(Log-Log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %% [markdown]
# ## Estimando el parametro alfa (usando el notebook de clase)

# %%
ranks = np.array(df_ord["Rango"])
frequencies = np.array(df_ord["Frecuencia"])


# %%
def zipf_minimization_objective(
    alpha: np.float64, word_ranks: np.ndarray, word_frequencies: np.ndarray
) -> np.float64:
    predicted_log_freq = np.log(word_frequencies[0]) -alpha * np.log(word_ranks)
    return np.sum((np.log(word_frequencies) - predicted_log_freq) ** 2)


# %%
initial_alpha_guess = 1.0

optimization_result = minimize(
    zipf_minimization_objective,
    initial_alpha_guess,
    args=(ranks, frequencies)
)

estimated_alpha = optimization_result.x[0]

mean_squared_error = zipf_minimization_objective(
    estimated_alpha, ranks, frequencies
)

print(f"Estimated alpha: {estimated_alpha:.4f}")
print(f"Mean Squared Error: {mean_squared_error:.4f}")

# %%
plt.loglog(ranks, frequencies, "o", label="Datos")

zipf_fit = frequencies[0] * ranks**(-estimated_alpha)

plt.loglog(ranks, zipf_fit, label=f"Zipf α={estimated_alpha:.2f}")

plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.legend()

plt.show()

# %% [markdown]
# ## Analizando un corpus con pocos recursos digitales
# Analizaremos una biblia de Tzotzil de los altos de chiapas, exáctamente del Tzotzil de Zinacantán.

# %%
import pdfplumber
texto = ""
with pdfplumber.open("biblia_tzotzil.pdf") as pdf:
    for pagina in pdf.pages:
        texto += pagina.extract_text() + "\n"
texto = texto.lower()

# %%
import re
texto = re.sub(r"\d+", "", texto)
texto = re.sub(r"[^\w\s']", " ", texto)
texto_l = texto.split()
conteo_tzo = Counter(texto_l)
conteo_tzo.most_common(20)

# %%
df_tzo = pd.DataFrame(conteo_tzo.items(), columns = ["Palabras","Frecuencia"])

# %%
df_tzo.head(20)

# %%
# DataFrame ordenado 
df_ord_tzo = df_tzo.sort_values("Frecuencia", ascending = False,).copy()
df_ord_tzo["Rango"] = range(1,len(df_ord_tzo)+1)
df_ord_tzo.head(20)

# %%
plt.plot(df_ord_tzo["Rango"],df_ord_tzo["Frecuencia"],marker = "o")
plt.title("Ley de Zipf")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %%
plt.loglog(df_ord_tzo["Rango"],df_ord_tzo["Frecuencia"],marker = "o")
plt.title("Ley de Zipf (log-log)")
plt.xlabel("Rango")
plt.ylabel("Frecuencia")
plt.show()

# %%
