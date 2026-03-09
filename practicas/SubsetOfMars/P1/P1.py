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

# %% [markdown]
# # Morfología
