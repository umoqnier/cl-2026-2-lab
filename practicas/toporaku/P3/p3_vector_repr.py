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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Práctica 3. Representación Vectorial.
# ### Lingüística Computacional 2026-2
# #### Cuerpo Académico
# **Dra.** María Ximena Gutiérrez Vasques
#
# **Ayud.** Ximena de la Luz Contreras Mendoza
#
# **Lab.** Diego Alberto Barriga Martínez
#
# #### Alumno
# Toporek Coca Eric - **314284987**
#
# ## 1 Matrices dispersas y búsqueda de documentos
#
# Tomaremos como referencia los parrafos introductorios de sitios de Wikipedia, sin tomar en cuenta que tan fidedigna pueda ser la información que contiene. El listado se da como sigue:
#
# 1. [Salvador Allende](https://es.wikipedia.org/wiki/Salvador_Allende#)
# 2. [Pinochet](https://es.wikipedia.org/wiki/Augusto_Pinochet)
# 3. [Golpe de Estado en Chile de 1973](https://es.wikipedia.org/wiki/Golpe_de_Estado_en_Chile_de_1973)
# 4. [Árbol de Manzano](https://es.wikipedia.org/wiki/Malus_domestica)
# 5. [Árbol de Cacao](https://es.wikipedia.org/wiki/Theobroma_cacao)

# %%
doc_1 = """Salvador Guillermo Allende Gossens​ (Santiago, 26 de junio de 1908-Santiago, 11 de septiembre de 1973) fue un médico cirujano y político socialista chileno, presidente de Chile —el segundo nacido en el siglo XX— desde el 3 de noviembre de 1970 hasta el 11 de septiembre de 1973. Su gobierno, caracterizado por un socialismo democrático, acabó abruptamente con un golpe de Estado, durante el cual optó por suicidarse mientras militares comandados por Augusto Pinochet tomaban el Palacio de la Moneda. Participó en política desde sus estudios en la Universidad de Chile. Fue sucesivamente diputado (1937-1939), ministro de Salubridad (1939-1942) del gobierno de Pedro Aguirre Cerda, y senador (1945-1970), ejerciendo la presidencia de la cámara alta del Congreso (1966-1969). Además, fue secretario general del Partido Socialista de Chile (1942-1943). Fue candidato a la presidencia de la República en cuatro oportunidades: en las elecciones de 1952 obtuvo el 5,45 % de los votos; en las de 1958, el 28,85 % de los votos, tras el electo Jorge Alessandri; en las de 1964, el 38,93 % de los votos, tras el electo Eduardo Frei Montalva; y, finalmente, en las de 1970, a tres bandas, consiguió la primera mayoría simple con el 36,63 % de los votos, siendo en definitiva ratificado por el Congreso Nacional. De ese modo, Allende se convirtió en el primer presidente marxista del mundo en acceder al poder a través de elecciones generales en un Estado democrático de derecho. Su gobierno fue apoyado por la Unidad Popular, una coalición de partidos de izquierda, y destacó tanto por el intento de establecer un Estado socialista aferrándose a los medios democráticos y constitucionales del Poder Ejecutivo —la «vía chilena al socialismo»—, como por proyectos como la nacionalización del cobre, la estatización de las áreas «claves» de la economía y la profundización de la reforma agraria iniciada por su antecesor, Eduardo Frei Montalva. El gobierno de la Unidad Popular se vio envuelto en la polarización política internacional de la Guerra Fría​ y sufrió de la persistente intervención por parte del gobierno estadounidense de Richard Nixon y la Agencia Central de Inteligencia (CIA) con el fin de provocar un cambio de régimen. En medio de una crisis económica (llegando a tener una inflación del 606 % en 1973), social y a tres años antes del fin de su mandato constitucional, el gobierno de Salvador Allende terminó abruptamente el 11 de septiembre de 1973 mediante un golpe de Estado en el que participaron las tres ramas de las Fuerzas Armadas y el cuerpo de Carabineros.​ Ese mismo día, después de que el Palacio de La Moneda fuese bombardeado por aviones y tanques, Allende se suicidó. Tras el fin de su gobierno, el general Augusto Pinochet encabezó una dictadura militar que se extendió por más de dieciséis años, lo que puso fin al período de la República Presidencial."""
doc_2 = """Augusto José Ramón Pinochet Ugarte (Valparaíso, 25 de noviembre de 1915-Santiago, 10 de diciembre de 2006) fue un militar, político y dictador chileno que gobernó Chile entre 1973 y 1990 durante la dictadura militar. El presidente Salvador Allende lo designó comandante en jefe del Ejército de Chile el 23 de agosto de 1973, en reemplazo del renunciado general Carlos Prats. El 11 de septiembre del mismo año, en medio de una crisis política, económica y social, dirigió junto a José Toribio Merino y Gustavo Leigh un golpe de Estado que derrocó al gobierno democrático de la coalición de partidos políticos de izquierda denominada Unidad Popular, poniendo fin al período de la República Presidencial.​ Desde ese momento, gobernó el país, primero como presidente de la Junta Militar de Gobierno —al que se sumó el título de jefe supremo de la Nación el 27 de junio de 1974, que le confirió el poder ejecutivo— y luego, a partir del 16 de diciembre de 1974, como presidente de la República, cargo que fue ratificado tras un cuestionado plebiscito y la promulgación de una nueva Constitución en 1980. Su mandato acabó por la vía democrática mediante otro plebiscito realizado in 1988, tras el cual fue sustituido —luego de realizarse elecciones presidenciales y parlamentarias— por Patricio Aylwin el 11 de marzo de 1990. Pinochet se mantuvo como comandante en jefe del Ejército hasta el 10 de marzo de 1998 y al día siguiente asumió como senador vitalicio, cargo que ejerció efectivamente por un par de meses. Bajo la influencia de los «Chicago Boys», economistas orientados al libre mercado, el nuevo régimen implementó la liberalización económica, incluida la estabilización monetaria. También eliminó las protecciones arancelarias para la industria local, prohibió los sindicatos y privatizó la seguridad social y empresas estatales. Estas políticas produjeron un inicial crecimiento económico, que Milton Friedman denominó el «milagro de Chile», pero que contrasta con un aumento dramático en la desigualdad de ingresos y que habría llevado a una devastadora crisis económica en 1982 influida por el contexto global de la segunda crisis del petróleo detonada en 1979, por la revolución islámica en Irán y la subsecuente guerra Irán-Irak que comenzó en 1980.​ Durante la mayor parte de la década de 1990, Chile fue la economía de mejor desempeño en América Latina, aunque el legado de las reformas de Pinochet sigue en disputa. Durante la dictadura se cometieron graves y diversas violaciones de los derechos humanos. Pinochet persiguió a izquierdistas, socialistas y críticos políticos, lo que provocó el asesinato de entre 1200 y 3200 personas,​ la detención de unas 80 000 personas y la tortura de decenas de miles. Según el gobierno chileno, el número de ejecuciones y desapariciones forzadas fue de 3095. Pinochet fue arrestado, en virtud de una orden internacional de arresto expedida por un juez español, tras una visita a Londres el 10 de octubre de 1998 en relación con numerosas violaciones de derechos humanos. Luego de una batalla legal, fue liberado por motivos de salud y regresó a Chile el 3 de marzo de 2000. En 2004, el juez chileno Juan Guzmán Tapia dictaminó que Pinochet era médicamente apto para enfrentar un juicio y lo puso bajo arresto domiciliario. Al momento de la muerte de Pinochet, el 10 de diciembre de 2006, en Chile aún se encontraban pendientes 300 cargos penales por numerosas violaciones de derechos humanos durante su mandato de casi diecisiete años, además de casos de evasión de impuestos y malversación durante y después de dicho periodo. También el juez Muñoz estimó que acumuló ilícitamente al menos 28 millones de dólares."""
doc_3 = """El golpe de Estado en Chile del 11 de septiembre de 1973 fue una acción militar llevada a cabo por las Fuerzas Armadas de Chile conformadas por la Armada, la Fuerza Aérea, Cuerpo de Carabineros y el Ejército, para derrocar al presidente socialista Salvador Allende y al gobierno de la Unidad Popular. Tropas del ejército y aviones de la Fuerza Aérea atacaron el Palacio de La Moneda, la sede de gobierno. Allende se suicidó mientras las tropas militares ingresaban al Palacio. Este golpe dio origen al establecimiento de una junta militar liderada por Augusto Pinochet. Chile, que hasta ese entonces se mantenía como una de las democracias más estables en América Latina, entró en una dictadura militar que se extendió hasta 1990. Durante este periodo, se cometieron sistemáticas violaciones a los derechos humanos,​ se limitó la libertad de expresión, se suprimieron los partidos políticos y se disolvió el Congreso Nacional. Salvador Allende asumió en 1970 la presidencia de Chile, siendo el primer político de orientación marxista en el mundo que accedió al poder a través de elecciones generales en un Estado de Derecho. Su gobierno, de marcado carácter reformista, produjo una creciente polarización política en la sociedad y una dura crisis económica que desembocó en una fuerte convulsión social. Esto llevó a una acusación constitucional por parte del Congreso poco antes del golpe. Sin embargo, la posibilidad de ejecutar un golpe de Estado contra el gobierno de Allende existió incluso antes de su elección. El gobierno de Estados Unidos, dirigido por el presidente Richard Nixon y su secretario de Estado Henry Kissinger, influyeron decisivamente en grupos opositores a Allende, financiando y apoyando activamente las condiciones para la ejecución de un golpe de Estado. Dentro de estas acciones se encuentran el asesinato del general René Schneider y el Tanquetazo, una sublevación militar el 29 de junio de 1973. Según el historiador Sebastián Hurtado, la ayuda estadounidense al golpe habría sido indirecta, afirmando que «no hay evidencia documental que sostenga que Washington actuó activamente en la coordinación y ejecución de las acciones del 11 de septiembre de 1973 (mismo)». Sin embargo, el interés de Richard Nixon desde el principio fue que el gobierno de Allende no fuese consolidado en el tiempo. Tras el Tanquetazo, grupos dentro de la Armada de Chile planearon derrocar al gobierno, al que posteriormente se sumaron los altos mandos de la Fuerza Aérea y grupos dentro de Carabineros. Días antes de la fecha planificada para la acción militar, se sumó Augusto Pinochet, comandante en jefe del Ejército. En la mañana del 11 de septiembre de 1973, las cúpulas de las Fuerzas Armadas y de Orden lograron rápidamente controlar gran parte del país exigiendo la renuncia inmediata de Salvador Allende, quien se refugió en la sede de gobierno. Hasta hoy este evento histórico divide al país (según encuesta CERC-MORI realizada en el 50.º aniversario del suceso, 36 % de chilenos afirma que militares tuvieron razón en su actuar). Testimonio de la discrepancia es que algunos ideólogos y partidarios del golpe de Estado aun lo califican o justifican como «pronunciamiento militar», denominación que opositores rechazan por considerarla eufemística o agraviante."""
doc_4 = """Malus domestica, el manzano europeo o manzano común, es un árbol de la familia de las rosáceas, cultivado por su fruto, apreciado como alimento. Su domesticación parece haber comenzado hace más de 15 000 años en la región comprendida al oeste de las montañas Tian Shan, frontera entre Kazajistán y China. Fue introducido en Europa por los romanos y en la actualidad existen unas 1000 variedades/cultivares, como resultado de innumerables hibridaciones entre formas silvestres. El fruto es una gran fuente de vitaminas. Es un árbol de mediano tamaño (4 m de altura), inerme, caducifolio, de copa redondeada abierta y numerosas ramas que se desarrollan casi horizontalmente. El tronco tiene corteza agrietada que se desprende en placas. Las hojas, estipuladas y cortamente pecioladas, son ovaladas, acuminadas u obtusas, de base cuneada o redondeada, generalmente de bordes aserradas pero ocasionalmente sub-enteras, de fuerte color verde y con pubescencia en el envés. Al estrujarlas despiden un agradable aroma. La inflorescencia es una cima umbeliforme o corimbiforme con 4-8 flores hermafroditas de ovario ínfero, siendo la central la primera en formarse en posición terminal, resultando la más desarrollada y competitiva. A esta se le llama comúnmente «flor reina» y generalmente produce los frutos de mayor tamaño y calidad. Dichas flores son hermafroditas, con un cáliz de cinco sépalos, una corola de 5 pétalos blancos, redondeados, frecuentemente veteados de rojo o rosa, con uña milimétrica y 20 estambres. El manzano florece en primavera antes de la aparición anual de sus hojas. El fruto, la manzana, es un pomo de 30-100 por 35-110 mm, globoso, con restos del cáliz en el ápice, verde, amarillo, rojizo, etc. con semillas de 7-8 por 4 mm. La manzana suele madurar hacia el otoño. La del manzano silvestre se diferencia por un color verde amarillento en su piel y de sabor agrio."""
doc_5 = """Theobroma cacao es el nombre científico que recibe el árbol del cacao o cacaotero, nativo de regiones tropicales subtropicales de América del sur: América tropical, planta de hoja perenne de la familia Malvaceae. Theobroma significa, en griego, «alimento de los dioses». La palabra cacao se cree que viene de los lenguajes de la familia mixe-zoque que habrían hablado los olmecas.​ En maya yucateco, kaj significa amargo y kab significa jugo. Alternativamente, algunos lingüistas[¿quién?] proponen la teoría de que en el correr del tiempo pasó por varias transformaciones fonéticas que dieron paso a la palabra cacaoatl, la cual evolucionó después a cacao. El árbol de cacao necesita de humedad y de calor. Es de hoja perenne y siempre se encuentra en floración, crece entre los 5 y los 10 m de altura. Requiere sombra (crecen a la sombra de otros árboles más grandes como Inga edulis y platanero), protección del viento y un suelo rico y poroso, pero no se desarrolla bien en las tierras bajas cálidas. Su altura ideal es, más o menos, a 400 m s. n. m. El terreno debe ser rico en nitrógeno, magnesio y en potasio, y el clima húmedo, con una temperatura entre los 20 °C y los 30 °C. Árbol de pequeña talla, perennifolio, de 4 a 7 m de altura si es cultivado, en su forma silvestre puede crecer hasta 20 m. Hojas grandes, alternas, colgantes, elípticas u oblongas, de punta larga, ligeramente gruesas, margen liso, cuelgan de un pecíolo. El tronco generalmente es recto, las ramas primarias se forman en verticilos terminales con tres a seis ramillas y al conjunto se le llama "molinillo". Es una especie cauliflora, es decir, las flores aparecen insertadas sobre el tronco o las viejas ramificaciones. Corteza de color castaño oscuro, agrietada, áspera y delgada. Flores en racimos a lo largo del tronco y de las ramas, de color rosa, púrpura y blanco en forma de estrella. El fruto es una baya grande comúnmente denominada mazorca, carnosa, oblonga a ovada, de color amarilla o purpúrea, de 15 a 30 cm de largo por 7 a 10 cm de grueso, puntiaguda y con canales longitudinales, cada mazorca contiene en general entre treinta y cuarenta semillas incrustadas en una masa de pulpa desarrollada de las capas externas de la testa. El fruto se vuelve rojo o amarillo purpúreo y pesa aproximadamente 450 g cuando madura (de 15 a 30 cm de largo por 7 a 12 de ancho). Un árbol comienza a rendir cuando tiene cuatro o cinco años. En un año, cuando madura, puede tener 6000 flores pero solo veinte maracas. A pesar de que sus frutos maduran durante todo el año, normalmente se realizan dos cosechas: la principal (que empieza hacia el final de la estación lluviosa y continúa hasta el inicio de la estación seca) y la intermedia (al principio del siguiente periodo de lluvias), y son necesarios de cinco a seis meses entre su fertilización y su recolección."""

# %% [markdown]
# ### Query "Tramposa"
# A continuación se presenta un nuevo documento (`query_tramposa`) dirigido a un **tema completamente ajeno** a los documentos principales: **la Exploración Espacial y Astronomía**. Sin embargo, la redacción ha sido forzada para incorporar una gran cantidad de términos frecuentes de las dos temáticas del corpus (Historia de Chile y Botánica).
#
# **Palabras infiltradas de Botánica:** terreno rico, árbol, hojas, madura, fruto, semilla, clima húmedo, ramas, corteza agrietada, sombra, de gran tamaño.
# **Palabras infiltradas de Historia:** presidente, gobierno, golpe, estado, régimen, intervención, fuerzas, poder, elecciones.

# %%
query_tramposa = """La agencia espacial instaló su centro de control sobre un terreno rico y poroso, desde donde el presidente de la misión ejerce el poder sobre las operaciones lunares. El cohete, diseñado como un árbol de gran tamaño y con una gruesa corteza agrietada para resistir el calor, inicia su despegue tras un fuerte golpe de ignición, escapando del Estado terrestre y del clima húmedo de la región. Una vez en el vacío, el satélite crece a la sombra de los grandes planetas, extendiendo sus paneles solares como si fueran hojas perennes y ramas metálicas. Durante su órbita, debe evitar la intervención de campos magnéticos que actúan como fuerzas arrolladoras sobre sus sistemas. Su gobierno interno procesa enormes cantidades de datos a medida que la misión madura en años luz. Los científicos monitorean este régimen constante, esperando que tras diversas elecciones de trayectoria, la nave logre recolectar valiosas semillas estelares, dando a la humanidad un fruto carnoso en forma de conocimiento en medio de la inmensidad universal."""

# %% [markdown]
# #### Vectorización

# %%
documents = [doc_1,doc_2,doc_3,doc_4,doc_5,query_tramposa]

# %%
import re
from nltk.tokenize import word_tokenize


def simple_preprocess(text: str):
    tokens = word_tokenize(text.lower(), language="spanish")
    # Ignoramos signos de puntuación y palabras de longitud 1
    return [word for word in tokens if word.isalnum() and len(word) > 1 and not re.match(r"\d+", word)]


# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
vectorizer = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)

# %%
bag_of_words_corpus = vectorizer.fit_transform(documents)

# %%
diccionario = vectorizer.vocabulary_

# %%
sorted(diccionario.items(), key=lambda x: x[1])

# %%
# Visualizando la matriz
bag_of_words_corpus.toarray()

# %%
print(len(bag_of_words_corpus.toarray()))
len(bag_of_words_corpus.toarray()[1])

# %%
import pandas as pd
def create_bow_dataframe(docs_raw: list, titles: list[str], vectorizer) -> pd.DataFrame:
    # fit_transform ajusta el vocabulario y crea la matriz en un solo paso
    matrix = vectorizer.fit_transform(docs_raw)

    # Podemos crear el DataFrame directamente pasando la matriz a un array tradicional
    # vectorizer.get_feature_names_out() nos da la lista de palabras en el orden exacto de las columnas
    df = pd.DataFrame(
        matrix.toarray(), index=titles, columns=vectorizer.get_feature_names_out()
    )
    return df


# %%
titles = ["ALLENDE", "PINOCHET", "GOLPE", "MANZANO", "CACAO", "ESPACIO"]
docs_matrix = create_bow_dataframe(
    documents,
    titles,
    vectorizer=CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)#, binary=True),
)

# %%
docs_matrix

# %%
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compute_similarity_vectorized(docs_matrix):
    # Separate 'ESPACIO' from the rest of the dataframe
    espacio_df = docs_matrix.loc[['ESPACIO']]
    other_docs = docs_matrix.drop(index='ESPACIO')
    
    # compute similarities
    sims = cosine_similarity(other_docs, espacio_df).flatten()
    
    return pd.Series(sims, index=other_docs.index, name='Cosine Similarity vs ESPACIO').sort_values(ascending=False)


# %%
bow_simil = compute_similarity_vectorized(docs_matrix)
bow_simil

# %% [markdown]
# #### TF-IDF

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
docs_matrix_tfidf = create_bow_dataframe(
    documents, titles, TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
)

# %%
docs_matrix_tfidf

# %%
tfidf_simil = compute_similarity_vectorized(docs_matrix_tfidf)
tfidf_simil

# %% [markdown]
# #### 5. DataFrame

# %%
comparison_df = pd.DataFrame({
    'BoW Similarity': bow_simil,
    'TF-IDF Similarity': tfidf_simil
})
comparison_df = comparison_df.sort_values(by='TF-IDF Similarity', ascending=False)
comparison_df


# %% [markdown]
# #### 5.1. ¿Cambió el documento clasificado como "más similar/relevante" al pasar de BoW a TF-IDF? Identifica el cambio si lo hubo.
#
# **Sí, el documento más relevante cambió.** Utilizando la representación **BoW**, el documento clasificado como más similar fue **ESPACIO**, ya que esta métrica se dejó engañar por la alta repetición de palabras clave. Sin embargo, al aplicar **TF-IDF**, el documento más relevante pasó a ser **ALLENDE**, reflejando una similitud semántica mucho más precisa.
#
# #### 5.2. Explica brevemente, basándote en la penalización idf (Inverse Document Frequency), cómo y por qué TF-IDF procesó de manera distinta las palabras de tu "trampa léxica" en comparación con BoW.
#
# En el modelo **BoW (Bag of Words)**, el peso de un término depende únicamente de cuántas veces aparece en el documento (frecuencia pura). Esto hace que el modelo sea vulnerable a "trampas léxicas", es decir, documentos que repiten artificialmente ciertas palabras muchas veces para inflar su vector y lograr engañar la métrica de similitud del coseno.
#
# Por otro lado, **TF-IDF** introduce la métrica **IDF (Inverse Document Frequency)**, que funciona como un mecanismo de penalización o balanceo global. Si las palabras utilizadas en la trampa léxica aparecen en una gran cantidad de documentos dentro del corpus general, el mecanismo IDF asume que esas palabras no aportan información verdaderamente discriminatoria o única y reduce matemáticamente su peso (penalización). Al multiplicar la frecuencia del término (TF) por una penalización alta (IDF muy bajo), el peso final de esos términos artificiales colapsa. Esto permite que TF-IDF ignore el ruido de la repetición masiva e identifique el documento que genuinamente posee los términos distintivos y relevantes de la búsqueda.
#

# %% [markdown]
# ## 2. Búsqueda de sesgos

# %%
# %pip install gensim

# %%
import gensim.downloader as gensim_api

word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %% [markdown]
# ### 1. 

# %%
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))

# %% [markdown]
# ### 2. 
# Podemos observar que los conceptos asociados con la mujer están más orientados a labores de enseñanza y cuidado, o meor dicho *crianza*, mientras que para los del hombre, se puede ver más un peso sobre la intelectualidad, el conocimiento y los hábitos, por lo que el sesgo es evidente.
#
# ### 3. 
#
# En una analogía, podemos reutilizar la función para ver sesgos raciales.

# %%
print(word_vectors.most_similar(positive=['black', 'crime'], negative=['white']))
print()
print(word_vectors.most_similar(positive=['white', 'crime'], negative=['black']))

# %% [markdown]
# Mientras que, por un lado al asociar la palabra crimen con **negro**, se derivan conceptos con crímenes violentos y usuales en los reportes de criminalidad. Por el otro lado, al asociarlo con **blanco**, los conceptos son variados, entre crímenes de cuello blanco, como la corrupción así como conceptos que recaen más en el lado de la justicia.
#
# ### 4. Propuesta de modelo
#
# Sabemos que, los modelos de lenguaje como los conocemos de hoy en día producen sesgos a partir de los datos con los que son entrenados, por lo que ciertos puntos a atender pueden ser los siguientes:
# * Datasets más diversificados, con narrativas contrapuestas a partir de las características que pueden generar un sesgo.
# * Ajuste de pesos a los vectores, si dichos pesos se ajustan a modo que se le reste importancia a las características que definen a un sesgo, se puede mejorar el resultado.
# * Supervisión humana que pueda determinar si el entrenamiento del modelo produce sesgos y poder influir en la toma de decisiones. 
