#!/usr/bin/env python
# coding: utf-8

# # Práctica 3: Representaciones Vectoriales

# Por: **Hanabi Hernández Arce**
# 
# No. de cuenta: **322004416**
# 

# ## Matrices dispersas y búsqueda de documentos

# ### 1. Define una lista de 5 documentos cortos divididos en dos temas contrastantes.

# #### TEMA 1. El Ciclo del Agua

# In[1]:


texto1 = """El ciclo del agua es impulsado por la energía solar. El sol calienta la superficie del océano y otras aguas superficiales, lo que evapora el agua líquida y sublima el hielo, convirtiéndolo directamente de sólido a gas. Estos procesos impulsados por el sol mueven el agua hacia la atmósfera en forma de vapor de agua."""
# El ciclo del agua: https://es.khanacademy.org/science/biology/ecology/biogeochemical-cycles/a/the-water-cycle.


# In[2]:


texto2 = """El agua es vital para los seres vivos y sólo el 3% del total es dulce, del cual el 2% está en forma de hielo, por lo tanto, sólo el 1% está en los cuerpos de agua dulce y gracias al ciclo del agua circula constantemente entre los distintos estratos de la hidrósfera.
Sube a la atmósfera el vapor de agua proveniente tanto de la transpiración de las plantas, como de la evaporación del agua de los suelos, ríos, lagos, etc. y particularmente de los océanos, esto debido a la acción constante de la radiación solar; todo este vapor queda retenido en las nubes, que al enfriarse se condensa alrededor de partículas de polvo formando pequeñas gotas que se precipitan a la superficie terrestre como lluvia, granizo, nieve, aguanieve o también como neblina."""
# Ciclo del agua: https://e1.portalacademico.cch.unam.mx/alumno/biologia2/estructura-procesos-ecosistema/ciclo-agua


# #### TEMA 2. La Salud Renal

# In[3]:


texto3 = """Beber agua y otros líquidos saludables como leche de vaca, leches vegetales y sopas ayuda a tus riñones a eliminar los desechos de manera más eficiente. Una hidratación saludable significa tener la cantidad adecuada de agua en el cuerpo para mantenerte bien.
Elige agua y otros líquidos saludables cuando tengas sed. Evita las bebidas azucaradas y los refrescos. Bebe más agua cuando trabajes o hagas ejercicio intenso, y cuando haga mucho calor. """
# Consejos para proteger tu salud renal: https://www.kidney.org/es/news-stories/como-cuidar-mis-rinones-consejos-faciles-para-proteger-tu-salud-renal


# In[4]:


texto4 = """La salud renal es fundamental, ya que los riñones filtran los desechos y el exceso de líquidos en la sangre, pero cuando se presenta insuficiencia renal, los niveles de retención de líquidos, electrolitos y los desechos se acumulan en el cuerpo causando exceso de urea -compuesto químico cristalino e incoloro que se encuentra abundantemente en la orina y en la materia fecal- en la sangre."""
# Cuida tus riñones: https://www.gob.mx/salud/articulos/cuida-tus-rinones-de-las-enfermedades-renales-cronicas


# In[5]:


texto5 = """El riñón es un órgano par situado en la parte alta de la región retroperitoneal, a ambos lados de los grandes vasos paravertebrales a los que se une por su pedículo vascular, y provisto de un conducto excretor, el uréter, que desemboca en la vejiga urinaria. Está formado por una serie de estructuras vasculares y epiteliales que funcionan en relación estrecha y que lo convierten en el órgano primordial del sistema urinario. Este sistema es el encargado de formar la orina y de eliminarla del cuerpo. La secreción de orina y su eliminación son cometidos vitales, pues constituyen en conjunto uno de los mecanismos básicos de la homeostasis del medio interno; hasta el punto de que, como se ha dicho, la composición de la sangre y del medio interno está regida, no por lo que se ingiere, sino por lo que los riñones conservan."""
# Morfologia y funcion renal: https://www.pediatriaintegral.es/numeros-anteriores/publicacion-2013-07/morfologia-y-funcion-renal/


# In[6]:


docs = [texto1, texto2, texto3, texto4, texto5]


# ### 2. Crea una query "tramposa", esto es, crea un documento dirigido a alguna temática pero repitiendo intencionalmente palabras comunes o verbos genéricos que aparezcan en los documentos de la otra temática.

# In[7]:


query = "la importancia del agua es alta en este sistema, es tan fundamental que este líquido es lo más elemental en el proceso."
docs_update = docs + [query]


# ### 3. Vectoriza para crear una BoW y calcula la similitud coseno entre la query y los 5 documentos

# In[8]:


import nltk
import numpy as np
import pandas as pd


# In[9]:


import re
from nltk.tokenize import word_tokenize

def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language="spanish")
    # Ignoramos signos de puntuación y palabras de longitud > 1
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


cv = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)


# In[12]:


nltk.download('punkt_tab')


# In[13]:


bag_of_words_corpus = cv.fit_transform(docs_update)


# In[14]:


dic = cv.vocabulary_


# In[15]:


bag_of_words_corpus.toarray()


# In[16]:


def create_bow_dataframe(docs_raw: list, titles: list[str], vectorizer) -> pd.DataFrame:

    matrix = vectorizer.fit_transform(docs_raw)

    df = pd.DataFrame(
        matrix.toarray(), index=titles, columns=vectorizer.get_feature_names_out()
    )
    return df


# In[17]:


titles = ["CicloDelAguaKA", "CicloDelAguaCCH", "ConsejosRiñones", "RiñonesGobMX", "FunciónRiñon", "Query"]
docs_matrix = create_bow_dataframe(
    docs_update,
    titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)


# In[18]:


docs_matrix


# In[19]:


doc_1BoW = docs_matrix.loc["CicloDelAguaKA"].values.reshape(1, -1)
doc_2BoW = docs_matrix.loc["CicloDelAguaCCH"].values.reshape(1, -1)
doc_3BoW = docs_matrix.loc["ConsejosRiñones"].values.reshape(1, -1)
doc_4BoW = docs_matrix.loc["RiñonesGobMX"].values.reshape(1, -1)
doc_5BoW = docs_matrix.loc["FunciónRiñon"].values.reshape(1, -1)
queryBoW = docs_matrix.loc["Query"].values.reshape(1, -1)


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity

similitud_doc1_Query = cosine_similarity(doc_1BoW, queryBoW)
similitud_doc2_Query = cosine_similarity(doc_2BoW, queryBoW)
similitud_doc3_Query = cosine_similarity(doc_3BoW, queryBoW)
similitud_doc4_Query = cosine_similarity(doc_4BoW, queryBoW)
similitud_doc5_Query = cosine_similarity(doc_5BoW, queryBoW)

print(f"Similitud Coseno del texto1 con la query = {similitud_doc1_Query}")
print(f"Similitud Coseno del texto2 con la query = {similitud_doc2_Query}")
print(f"Similitud Coseno del texto3 con la query = {similitud_doc3_Query}")
print(f"Similitud Coseno del texto4 con la query = {similitud_doc4_Query}")
print(f"Similitud Coseno del texto5 con la query = {similitud_doc5_Query}")


# In[21]:


# El texto FunciónRiñon (texto5) es el más relacionado con la Query en la bag of words


# ### 4. Repite el proceso usando TF-IDF

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:


docs_matrix_tfidf = create_bow_dataframe(
    docs_update, titles, TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)


# In[24]:


doc_1TFIDF = docs_matrix_tfidf.loc["CicloDelAguaKA"].values.reshape(1, -1)
doc_2TFIDF = docs_matrix_tfidf.loc["CicloDelAguaCCH"].values.reshape(1, -1)
doc_3TFIDF = docs_matrix_tfidf.loc["ConsejosRiñones"].values.reshape(1, -1)
doc_4TFIDF = docs_matrix_tfidf.loc["RiñonesGobMX"].values.reshape(1, -1)
doc_5TFIDF = docs_matrix_tfidf.loc["FunciónRiñon"].values.reshape(1, -1)
queryTFIDF = docs_matrix_tfidf.loc["Query"].values.reshape(1, -1)


# In[25]:


similitud_doc1_Query = cosine_similarity(doc_1TFIDF, queryTFIDF)
similitud_doc2_Query = cosine_similarity(doc_2TFIDF, queryTFIDF)
similitud_doc3_Query = cosine_similarity(doc_3TFIDF, queryTFIDF)
similitud_doc4_Query = cosine_similarity(doc_4TFIDF, queryTFIDF)
similitud_doc5_Query = cosine_similarity(doc_5TFIDF, queryTFIDF)


# In[26]:


print(f"Similitud Coseno del texto1 con la query = {similitud_doc1_Query}")
print(f"Similitud Coseno del texto2 con la query = {similitud_doc2_Query}")
print(f"Similitud Coseno del texto3 con la query = {similitud_doc3_Query}")
print(f"Similitud Coseno del texto4 con la query = {similitud_doc4_Query}")
print(f"Similitud Coseno del texto5 con la query = {similitud_doc5_Query}")


# In[27]:


# Nuevamente, el texto del funcionamiento del riñon (texto5) es el más relacionado con la Query


# ### 5. Imprime un DataFrame o tabla comparativa que muestre los scores de similitud de BoW y TF-IDF del query contra los 5 documentos.

# In[28]:


textos = [texto1, texto2, texto3, texto4, texto5]
titulos_textos = ["CicloDelAguaKA", "CicloDelAguaCCH", "ConsejosRiñones", "RiñonesGobMX", "FunciónRiñon"]

textos_con_query = textos + [query]

cv = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
matriz_bow = cv.fit_transform(textos_con_query)

docs_bow = matriz_bow[:-1] 
query_bow = matriz_bow[-1:] 

similitudes_bow = cosine_similarity(docs_bow, query_bow).flatten()

tfidf = TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
matriz_tfidf = tfidf.fit_transform(textos_con_query)

docs_tfidf = matriz_tfidf[:-1]
query_tfidf = matriz_tfidf[-1:]

similitudes_tfidf = cosine_similarity(docs_tfidf, query_tfidf).flatten()

df_comparativo = pd.DataFrame({
    'Documento': titulos_textos,
    'Similitud BoW': similitudes_bow,
    'Similitud TF-IDF': similitudes_tfidf
})

df_comparativo.sort_values(by='Similitud TF-IDF', ascending=False)


# #### 5.1 ¿Cambió el documento clasificado como "más similar/relevante" al pasar de BoW a TF-IDF? Identifica el cambio si lo hubo.

# Recordemos que la query es: **"la importancia del agua es alta en este sistema, es tan fundamental que este líquido es lo más elemental en el proceso."**
# 
# Por como diseñé la query tramposa. El texto con mayor Similitud en BoW fue el texto4 "FunciónRiñon". Y, aunque en primer lugar quedó un texto del tema Salud Renal, claramente la query estaba más relacionada con el tema Ciclo del Agua, por lo que los segundo y tercer textos relacionados con la query fueron los de dicho tema.
# 
# El texto "RiñonesGobMX" tuvo una similitud considerablemente menor con la TF-IDF por lo que quedó en el cuarto puesto en similitud con la query.
# 
# Finalmente, el texto "ConsejosRiñones" fue el menos relacionado en ambas BoW y TF-IDF lo cual es esperable dada la query.

# #### 5.2 Explica brevemente, basándote en la penalización idf (Inverse Document Frequency), cómo y por qué TF-IDF procesó de manera distinta las palabras de tu "trampa léxica" en comparación con BoW.

# Ya que TF-IDF quita importancia a las palabras que más se repiten entre los textos, supongo que eso hizo que aunque "RiñonesGobMX" tuvo una mejor similitud que "CicloDelAguaCCH" en BoW, al quitarle valor a las palabras _sin significado,_ en promedio, su similitud con la query fue más baja. 
# 
# Igualmente, el texto en primer lugar bajó mucho su similaridad tras usar la TF-IDF.
# Esto se debe a mi diseño de query tramposa que provocó que comparten mucho vocabulario aunque las palabras con significado no sean tan parecidas entre la query y "FunciónRiñon".

# In[ ]:




