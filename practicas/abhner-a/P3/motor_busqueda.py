import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preparar_datos():
    # Documentos contrastantes
    documentos = [
        "El libro de Malvern detalla el estado tensional de un medio continuo.", 
        "El análisis tensional describe cómo las fuerzas deforman el material.", 
        "Calculamos el tensor de esfuerzo para entender el equilibrio tensional.",
        "El músico desliza el arco sobre las cuerdas del violín con suavidad.", 
        "La técnica del violín requiere precisión en las notas y la afinación."
    ]
    # Query con la trampa léxica "tensional"
    query = ["El violinista aplica un esfuerzo tensional al arco para lograr una nota tensional perfecta."]
    return documentos, query

def calcular_similitudes(docs, query):
    # Vectorización BoW
    cv = CountVectorizer()
    matriz_bow = cv.fit_transform(docs)
    q_bow = cv.transform(query)
    score_bow = cosine_similarity(q_bow, matriz_bow)[0]

    # Vectorización TF-IDF
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(docs)
    q_tfidf = tfidf.transform(query)
    score_tfidf = cosine_similarity(q_tfidf, matriz_tfidf)[0]

    return score_bow, score_tfidf
