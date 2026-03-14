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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 2 - Lingüística Computacional

# %% [markdown]
# ## 1. Verificación empírica de la Ley de Zipf

# %% [markdown]
# ### 1.1 Lenguaje artificial

# %%
from collections import Counter
import random
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 6]
