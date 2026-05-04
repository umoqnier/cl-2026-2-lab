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
# # **Práctica 3: Representaciones Vectoriales**
#
# ### **Nombre:** Omar Fernando Gramer Muñoz
# ### **Materia:** Lingüística Computacional
# ### **Matrícula:** 419003698

# %%
# Imports

import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## **Matrices dispersas y búsqueda de documentos**
# ___
#
# Este apartado requiere que construyas un motor de búsqueda entre documentos comparando el rendimiento de una Bolsa de Palabras (BoW) y TF-IDF para procesar un documento "tramposo" (documento con muchas palabras que aportan poco significado o valor temático):

# %% [markdown]
# ### **Ejercicio 1: Define una lista de 5 documentos cortos divididos en dos temas contrastantes.**
#

# %% [markdown]
# Usaré los siguientes dos temas: La novela de **Un mundo feliz** ( que es una de mis novelas favoritas ) y **SOMA** ( la historia de este videojuego y los dilemas que plantean sobre la consciencia tambien son mis cosas favoritas ). Ambos además comparten el uso de la palabra **soma** con contextos y significados muy diferentes, lo cual servirá para la query tramposa.

# %%
# Sinopsis general y contexto de la novela Un mundo feliz ( fragmento tomado de https://es.wikipedia.org/wiki/Un_mundo_feliz )
title_1 = "Sinopsis un mundo feliz"
doc_1 = """
"Un mundo feliz" (en inglés, "Brave New World") es la obra más conocida del escritor británico Aldous Huxley y fue publicada por primera vez en 1932. La novela presenta una sociedad distópica que anticipa avances en tecnología reproductiva, cultivo artificial de seres humanos, aprendizaje durante el sueño (hipnopedia) y control emocional mediante el uso de drogas como el "soma". La combinación de estos elementos transforma profundamente la organización social.
La sociedad descrita podría interpretarse como una especie de utopía, aunque de carácter irónico y ambiguo. La humanidad está organizada en castas donde cada individuo conoce y acepta su función dentro del sistema social. La sociedad es tecnológicamente avanzada, saludable y sexualmente libre. Problemas tradicionales como la guerra o la pobreza han desaparecido y, aparentemente, todos viven en un estado permanente de felicidad. Sin embargo, esta estabilidad se ha logrado sacrificando elementos fundamentales de la cultura humana, como la familia, la diversidad cultural, el arte, el desarrollo científico, la literatura, la religión, la filosofía y el amor. El título de la obra proviene de una referencia a "La tempestad" de William Shakespeare, específicamente de un discurso pronunciado por Miranda en el acto V.
La historia comienza en el año 632 después de Ford (equivalente al año 2540 del calendario gregoriano). Un grupo de estudiantes visita el Centro de Incubación y Condicionamiento de Londres, donde un científico les explica el proceso de reproducción artificial utilizado en la sociedad. A través de esta visita se revela que la estructura social se determina desde el nacimiento. El mundo está gobernado por un Estado Mundial que controla la reproducción humana con el objetivo de producir individuos perfectamente adaptados a su rol social. Las personas se clasifican en castas identificadas con letras del alfabeto griego, desde los Alfa —destinados a funciones de liderazgo— hasta los Épsilon, diseñados para realizar trabajos peligrosos y repetitivos. Esta planificación genética se complementa con procesos de condicionamiento psicológico, como la hipnopedia, un método de enseñanza durante el sueño que refuerza consignas y valores sociales.
La novela también critica diversos aspectos de la sociedad moderna, como la producción en cadena, que es presentada como una forma de deshumanización del trabajo. Asimismo, cuestiona la liberación de la moral sexual, interpretándola como una amenaza para el amor y la familia, además del uso constante de eslóganes, la centralización del poder político y la aplicación de la ciencia como herramienta para controlar los pensamientos y comportamientos de la población. Huxley dirige además una crítica hacia la sociedad consumista y capitalista. En la novela, Henry Ford —creador del modelo de producción en cadena y del automóvil Modelo T— es considerado el fundador simbólico de la sociedad y adquiere una especie de estatus divino. Incluso la letra "T", en referencia al Modelo T de Ford, reemplaza a la cruz cristiana como símbolo casi religioso.
El título de la novela proviene de una frase pronunciada por Miranda en el acto V de "La tempestad" de Shakespeare, cuando observa por primera vez a otras personas distintas de su padre. En la historia, el personaje John el Salvaje es un gran admirador de Shakespeare, lo que lo diferencia de la mayoría de los ciudadanos de esta sociedad distópica. Mientras que gran parte del legado cultural y artístico del pasado ha sido eliminado o archivado, las obras de Shakespeare se conservan únicamente bajo el control de los gobernantes del Estado Mundial.
"""

# %%
# Tecnología y recreación en Un Mundo feliz ( fragmento tomado de https://es.wikipedia.org/wiki/Un_mundo_feliz )
title_2 = "Vida cotidiana un mundo feliz"
doc_2 = """
En el Estado Mundial, la vida cotidiana está profundamente influenciada por una tecnología altamente desarrollada que interviene en prácticamente todos los ámbitos de la sociedad. El deporte constituye uno de los pilares de este sistema y se compone de juegos diseñados con múltiples artefactos tecnológicos con el objetivo de mantener activas a las fábricas. Actividades como el tenis superficial o el golf electromagnético representan las principales formas de entretenimiento para los distintos estratos sociales. Además, existe una norma que prohíbe crear nuevos juegos si estos no incluyen al menos la misma cantidad de dispositivos que el juego más complejo existente, lo cual busca incentivar el consumo constante.
El entretenimiento también incluye los llamados «amoríos», una versión tecnológicamente avanzada del cine. En estos cines, conocidos como cines sensibles, los espectadores colocan sus manos sobre pomos metálicos situados en los brazos de las sillas, lo que les permite experimentar físicamente las sensaciones que sienten los actores en pantalla. A lo largo de la obra también aparecen otros dispositivos de ocio avanzados, como las cajas de música sintética, los órganos de esencias —instrumentos que combinan música con aromas—, los órganos de color que integran música con luces brillantes, y la televisión.
El transporte también refleja este alto nivel tecnológico. Dentro de las ciudades, el medio principal de desplazamiento es el helicóptero, con variantes como los «taxicópteros» o los exclusivos «deporticópteros». Para las castas inferiores, el acceso al campo se realiza mediante monorraíles de alta velocidad. En el ámbito intercontinental, los viajes se efectúan en aviones cohete, cuyo color indica el destino al que se dirigen.
En los criaderos y centros de condicionamiento, la tecnología desempeña un papel fundamental en la producción y manipulación de embriones. Además del equipamiento avanzado de laboratorio, existen máquinas capaces de adaptar los embriones a condiciones específicas como calor, movimientos bruscos o enfermedades, con el propósito de prepararlos para desempeñar las funciones que les serán asignadas en determinados entornos. Una vez nacidos, los niños continúan siendo moldeados mediante diversos dispositivos tecnológicos diseñados para dirigir su comportamiento y su papel dentro de la sociedad. Por ejemplo, en los primeros capítulos se describe cómo los niños de la casta Delta son condicionados para rechazar tanto los libros como la naturaleza mediante el uso de bocinas y descargas eléctricas. Asimismo, la hipnopedia se aplica mediante altavoces instalados en las camas mientras los niños duermen.
Otros aspectos de la vida diaria también están marcados por la tecnología. La mayoría de la ropa se fabrica con materiales sintéticos finos como el acetato o la viscosa. Los hombres utilizan maquinillas electrolíticas para afeitarse y consumen chicles con hormonas sexuales. Para relajarse, los ciudadanos disponen de máquinas de masaje y del omnipresente «soma», una sustancia que generalmente se consume en forma de tableta, aunque también puede vaporizarse para crear una nube anestésica, como se muestra cuando John arroja las tabletas por la ventana.
Soma es una droga consumida en el mundo cada vez que las personas se encuentran deprimidas, con el fin de curar las penas y controlar los sentimientos.
Esta droga se basa en la droga soma, que utilizaban los bráhmanes en la época védica en la India, hace muchísimo tiempo. Más tarde se perdió el conocimiento acerca de esta planta, y actualmente no se sabe exactamente a cuál se refiere.
En la novela se dice que un gramo de soma cura diez sentimientos melancólicos y que tiene todas las ventajas del cristianismo y del alcohol, sin ninguno de sus efectos secundarios.
En la obra se puede leer: «Si, por desgracia, se abriera alguna rendija de tiempo en la sólida sustancia de sus distracciones, siempre queda el soma: medio gramo para una de asueto, un gramo para fin de semana, dos gramos para viaje al bello Oriente, tres para una oscura eternidad en la Luna».
La droga parece poder ser destilada en casi cualquier alimento, así pues los personajes consumen helados de soma, agua con soma, solución de cafeína (café) con soma…
En la obra literaria, la gente toma a menudo vacaciones de soma para encontrarse mejor anímicamente.
A su vez, el Estado se encarga del reparto de esta sustancia en forma libre y gratuita a fin de controlar las emociones sentidas por los miembros de la comunidad con el fin de mantenerlos contentos, factor necesario para no poner en peligro la estabilidad de la Metrópolis (nombre de la ciudad en la novela). 
"""

# %%
# Organización política y estructura social de Un Mundo feliz ( fragmento tomado de https://es.wikipedia.org/wiki/Un_mundo_feliz )
title_3 = "Politica mundo feliz"
doc_3 = """
En la sociedad descrita en la novela, todo el planeta está unificado bajo un único sistema político conocido como el Estado Mundial. Este gobierno global es administrado por diez controladores mundiales ubicados en distintas ciudades estratégicas. A pesar de esta aparente unificación total, existen algunas zonas que permanecen fuera del control directo del sistema, conocidas como «reservas salvajes». Estas áreas incluyen regiones de Nuevo México, partes de América del Sur, Samoa y algunas islas cercanas a Nueva Guinea. En una conversación entre John y el interventor mundial de Europa Occidental, Mustafá Mond, se revelan diversos detalles sobre la organización geográfica y política del Estado Mundial.
Mond explica que ciertas regiones del planeta no han sido incorporadas al sistema porque poseen pocos recursos o presentan condiciones climáticas demasiado extremas. Debido a su bajo valor económico, estas zonas no han sido «civilizadas» por el gobierno mundial y se mantienen como reservas donde las comunidades locales continúan viviendo de manera relativamente independiente. Algunas islas, como Islandia o las Malvinas, también cumplen una función particular: se utilizan como lugares de aislamiento para ciudadanos del Estado Mundial que no logran adaptarse a la vida social establecida.
La población del Estado Mundial, que alcanza aproximadamente dos mil millones de personas, está organizada de forma estricta en cinco castas sociales. En la parte superior se encuentran los Alfas, quienes ejercen funciones de dirección y liderazgo, seguidos por los Betas, que ocupan puestos administrativos y técnicos. Por debajo se sitúan los Gammas, Deltas y Epsilones, cuya capacidad intelectual y funciones sociales disminuyen progresivamente. Cada una de estas castas se divide además en categorías «más» y «menos». En la cúspide del sistema se encuentran los Alfa-doble-más, quienes están destinados a convertirse en los científicos y administradores más importantes de la sociedad.
Los individuos de cada casta son condicionados desde su desarrollo para aceptar su papel dentro del sistema y sentirse satisfechos con él. Este condicionamiento busca evitar el resentimiento entre los distintos grupos sociales. Paralelamente, la sociedad inculca constantemente la idea de que todas las castas cumplen una función esencial y que, por lo tanto, todos los miembros del sistema son igualmente importantes.
En términos raciales, la sociedad del Estado Mundial promueve una aparente armonía global. Aunque algunas regiones conservan mayor presencia de ciertos grupos étnicos —como ocurre en Inglaterra con la población mayoritariamente caucásica— también existe una diversidad significativa dentro de la población. Por ejemplo, durante una visita a una fábrica de productos eléctricos en Londres, John observa a personas blancas y negras trabajando juntas. Del mismo modo, uno de los «amoríos» descritos en la novela presenta a un actor negro y una actriz blanca como protagonistas. En los centros de incubación también se produce una diversidad racial deliberada, ya que los embriones se cultivan sin distinguir entre grupos étnicos, produciendo individuos de distintas características dentro de los mismos criaderos, como el centro principal de Londres.
"""

# %%
# Resumen de la historia y los principales acontecimientos del juego SOMA ( tomado de https://soma.fandom.com/es/wiki/SOMA )
title_4 = "Historia juego SOMA"
doc_4 = """
El protagonista de SOMA es Simon Jarrett. Antes de los eventos del juego, Simon sufrió un grave accidente automovilístico en el que murió su amiga Ashley Hall, quien viajaba con él. Aunque Simon sobrevivió, el accidente le dejó daños cerebrales permanentes. Debido a esta condición, decide someterse a un tratamiento experimental que incluye un escaneo completo de su cerebro.
La historia comienza con Simon teniendo una pesadilla relacionada con el accidente. Tras despertar en su apartamento, recibe una llamada de David Munshi, quien le recuerda que debe beber un líquido de contraste antes del procedimiento médico. Después de tomar el fluido, Simon se dirige a los laboratorios PACE para realizar el escaneo cerebral.
Al llegar, descubre que la oficina está cerrada por remodelación y aparentemente no hay nadie en recepción. Tras encontrar la combinación de una puerta trasera en un diario, logra entrar y finalmente se encuentra con David Munshi, quien le asegura que el procedimiento será tan simple como tomarse una fotografía. Sin embargo, durante el escaneo ocurre un intenso destello de luz, y Simon pierde el conocimiento.
Cuando despierta, ya no está en los laboratorios. Se encuentra en una instalación desconocida llamada PATHOS-II, específicamente en una estación denominada Upsilon. Confundido y sin saber cómo llegó allí, Simon explora el lugar y encuentra una herramienta llamada Omnitool. Poco después logra comunicarse por radio con una mujer llamada Catherine Chun, quien le pide que viaje hasta la estación Lambda para reunirse con ella.
A medida que avanza por la instalación, Simon descubre que PATHOS-II es un complejo submarino que se encuentra en estado de caos. Una sustancia negra de apariencia orgánica se extiende por las paredes y estructuras, destruyendo partes de la instalación. No hay señales de humanos vivos, aunque Simon encuentra robots que creen ser personas y otros que actúan de forma autónoma.
Cuando finalmente llega a Lambda, una criatura humanoide hiere gravemente a Catherine. En ese momento Simon descubre que Catherine no es una persona viva, sino una copia digital de una conciencia humana instalada en un cuerpo robótico. Poco después también se revela que Simon tampoco es humano: su mente es en realidad un escaneo de su cerebro almacenado en un chip de córtex colocado dentro del cuerpo de una mujer fallecida llamada Imogen Reed, equipada con un traje especial Haimatsu. El Simon original murió aproximadamente cien años antes. El escaneo realizado por Munshi había sido archivado y distribuido con fines médicos y académicos, y de alguna forma esta copia terminó activándose en PATHOS-II.
También se descubre que la superficie de la Tierra fue devastada por el impacto de un cometa que eliminó toda la vida en el planeta. PATHOS-II fue el único lugar donde algunos humanos lograron sobrevivir inicialmente, aunque la instalación sufrió daños severos y sus habitantes sabían que su tiempo era limitado.
Catherine pide entonces a Simon que la transfiera a su Omnitool para poder acompañarlo durante el resto del viaje. Su objetivo es completar el proyecto más importante de su vida: lanzar el ARCA. El ARCA es una simulación digital que contiene escaneos de las mentes de varios humanos, diseñada para preservar la conciencia humana en un entorno virtual. El plan era enviar esta estructura al espacio utilizando el cañón espacial Omega de PATHOS-II, donde podría sobrevivir durante miles de años gracias a paneles solares.
El proyecto fue abandonado después de que varios miembros de la tripulación se suicidaran tras ser escaneados, creyendo que su verdadera existencia continuaría únicamente dentro del ARCA. Aunque Catherine intentó llevar el ARCA hasta el cañón espacial, nunca logró completar el lanzamiento. El dispositivo quedó almacenado en la estación Tau, ubicada en el fondo de una enorme grieta submarina conocida como el Abismo.
Simon y Catherine atraviesan diversas estaciones destruidas de PATHOS-II, desplazándose incluso por el fondo del océano mientras enfrentan criaturas hostiles. Durante el recorrido, Simon descubre que la sustancia negra que invade las instalaciones es obra de la inteligencia artificial del complejo, llamada WAU. Tras el impacto del cometa, el WAU comenzó a expandirse usando grandes cantidades de gel estructural, intentando preservar la vida humana a cualquier costo. Para ello reanimó cadáveres, conectó escaneos mentales a máquinas y creó híbridos de carne y metal que ahora vagan por la instalación.
El propio Simon fue despertado por el WAU como parte de estos intentos por preservar la conciencia humana. Para el WAU, la mejor forma de salvar a la humanidad es crear nuevos cuerpos capaces de albergar mentes humanas, sin importar las consecuencias.
En un intento por llegar al Abismo, Catherine planea utilizar un submarino llamado DUNBAT almacenado en la estación Theta. Sin embargo, el plan falla cuando descubren que el WAU ha conectado una conciencia humana al submarino. El vehículo entra en pánico al activarse y escapa hacia las profundidades.
Como alternativa, deciden descender utilizando una plataforma llamada Escalador del Abismo. Para sobrevivir a la enorme presión del fondo marino, Simon necesita un traje más resistente. En la estación Omicron encuentran un exotraje Haimatsu con el cuerpo sin cabeza de Raleigh Herber. Catherine propone copiar la mente de Simon a un nuevo chip de córtex e instalarlo en ese cuerpo. Aunque Simon acepta, se enfurece al descubrir que su antiguo cuerpo sigue existiendo, ya que el proceso solo crea copias de la conciencia en lugar de transferirla.
Durante el descenso hacia el Abismo, Simon reflexiona sobre la naturaleza de su identidad, comprendiendo que tanto él como su versión anterior podrían considerarse el verdadero Simon. En el trayecto aparece Johan Ross, un antiguo miembro de PATHOS-II profundamente mutado por el contacto con el gel estructural.
Finalmente, Simon llega a la estación Tau, donde encuentra el ARCA custodiado por Sarah Lindwall, la última humana viva. Sarah se mantiene con vida gracias a un sistema de soporte vital, pero su situación es desesperada. Ella permite que Simon se lleve el ARCA y le pide que la desconecte de su soporte vital para poder morir.
Mientras se dirige al lugar de lanzamiento, Ross guía a Simon hasta una instalación secreta llamada Alfa, donde se encuentra el núcleo del WAU. Ross asegura que el traje de Simon contiene un tipo especial de gel estructural capaz de destruir al WAU y sostiene que las criaturas creadas por la inteligencia artificial no deberían considerarse humanas. Simon puede decidir si destruir o no al WAU. Poco después, una enorme criatura marina llamada Leviatán ataca a Ross y Simon logra escapar.
Simon llega finalmente a la estación Fi, donde se encuentra el cañón espacial Omega. Allí descubre el cuerpo de Catherine Chun, quien murió durante una discusión sobre el destino del ARCA. Simon prepara el lanzamiento y, junto con Catherine, inicia un nuevo escaneo de sus mentes para transferirlas a la simulación del ARCA.
Sin embargo, cuando el proceso termina, Simon despierta aún en la silla de lanzamiento. El ARCA ya ha sido enviado al espacio y la copia de su mente es la que se encuentra dentro de la simulación. Catherine explica nuevamente que la conciencia no puede trasladarse, solo copiarse, por lo que esta versión de Simon quedó atrás. Poco después los sistemas de Catherine fallan y Simon queda solo en el fondo del océano con una Omnitool averiada.
En una escena posterior a los créditos se muestra a la copia de Simon dentro del ARCA despertando en un entorno virtual pacífico. Allí se reúne con Catherine frente a una ciudad futurista mientras el ARCA viaja por el espacio, alejándose de la Tierra devastada.
"""

# %%
# Reflexión sobre la ciencia ficción y los temas filosóficos en SOMA ( tomado de https://www.anaitgames.com/articulos/soma-y-el-sentido-de-la-ciencia-ficcion )
title_5 = "Filosofía juego SOMA"
doc_5 = """
El videojuego SOMA utiliza el terror como punto de partida, pero su verdadera intención es desarrollar una obra de ciencia ficción centrada en preguntas filosóficas sobre la identidad, la conciencia y el significado de ser humano. Aunque en apariencia se presenta como un juego de terror ambientado en una instalación submarina, su diseño narrativo y mecánico está construido para transmitir una idea central: cuestionar qué define realmente a una persona.
La historia sigue a Simon Jarrett, quien se somete a un experimento médico en el que su actividad cerebral es escaneada digitalmente. Tras el procedimiento, despierta en una instalación submarina destruida en el futuro, donde descubre que los humanos han desaparecido y que muchas máquinas poseen copias de consciencias humanas. A partir de ese momento, el misterio inicial sobre el lugar en el que se encuentra evoluciona hacia preguntas más profundas sobre su propia identidad: quién es realmente y qué significa existir como una mente almacenada en una máquina.
El juego explora conceptos clásicos de la ciencia ficción, como la transferencia de conciencia, el transhumanismo y la posibilidad de separar la mente del cuerpo. Estas ideas no son nuevas dentro del género, pero SOMA las utiliza para construir una experiencia interactiva que obliga al jugador a reflexionar continuamente sobre su significado. La perspectiva en primera persona refuerza esta experiencia, ya que permite que el jugador se identifique con el protagonista y experimente directamente la incertidumbre sobre su propia naturaleza.
Uno de los elementos centrales de la historia es la inteligencia artificial WAU, diseñada originalmente para preservar la vida humana. Tras un desastre global, esta inteligencia artificial comienza a intentar recrear la vida utilizando cadáveres, máquinas y escaneos de consciencia humana. Estas acciones generan criaturas híbridas entre organismo y tecnología, lo que plantea un conflicto fundamental: si estas entidades pueden considerarse realmente humanas o simplemente imitaciones imperfectas de la vida.
A lo largo del juego también aparece el proyecto ARCA, un intento de preservar la humanidad almacenando las mentes de las personas en una simulación digital que pueda sobrevivir durante miles de años. Este proyecto plantea otra pregunta esencial: si la humanidad puede sobrevivir únicamente como memoria digital o si la vida requiere necesariamente un cuerpo físico.
Las decisiones que el jugador toma durante la historia no modifican significativamente el desarrollo de los acontecimientos, pero sí tienen un fuerte impacto psicológico. El objetivo no es ofrecer respuestas definitivas, sino provocar reflexión. El juego plantea preguntas como si una copia de la mente sigue siendo la misma persona, si varias versiones de una misma conciencia pueden considerarse auténticas o si preservar recuerdos digitales equivale realmente a salvar a la humanidad.
De esta forma, SOMA se presenta como una obra de ciencia ficción que utiliza el medio interactivo para abrir un debate filosófico. Más que resolver estas cuestiones, el juego invita al jugador a reflexionar sobre ellas, manteniendo una conversación abierta sobre la identidad, la conciencia y el futuro de la humanidad en un mundo donde la tecnología podría redefinir el concepto mismo de existencia.
"""

# %%
documents = [doc_1, doc_2, doc_3, doc_4, doc_5 ]
titles = [title_1, title_2, title_3, title_4, title_5]

# %% [markdown]
# #### **Análisis de frecuencias para justificar la query tramposa**
# Para diseñar una query tramposa con evidencia empírica, primero analizamos
# las palabras más frecuentes en cada tema y luego identificamos cuáles se comparten entre ambos y así diseñar una trampa léxica
# con evidencia.

# %%
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# %%
stop_words = set(stopwords.words("spanish"))

def tokenize(text):
    """Tokeniza y limpia un texto: minúsculas, solo letras, sin stopwords."""
    tokens = re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 2]


# %%
# Documentos por tema
docs_umf = [doc_1, doc_2, doc_3]   # Un mundo feliz
docs_soma = [doc_4, doc_5]         # SOMA

# %%
# Unir todos los tokens de cada tema
tokens_umf  = tokenize(" ".join(docs_umf))
tokens_soma = tokenize(" ".join(docs_soma))

freq_umf  = Counter(tokens_umf)
freq_soma = Counter(tokens_soma)

# %%
print("=== Top 20 palabras — Un mundo feliz ===")
for word, count in freq_umf.most_common(20):
    print(f"  {word}: {count}")

# %%
print("\n=== Top 20 palabras — SOMA ===")
for word, count in freq_soma.most_common(20):
    print(f"  {word}: {count}")

# %%
# Palabras que aparecen más frecuentes en ambos temas
top_umf  = {w for w, _ in freq_umf.most_common(100)}
top_soma = {w for w, _ in freq_soma.most_common(100)}

shared = top_umf & top_soma

print("=== Palabras frecuentes en AMBOS temas (candidatas para la trampa) ===")
for word in sorted(shared):
    print(f"  '{word}' → UMF: {freq_umf[word]} veces | SOMA: {freq_soma[word]} veces")

# %% [markdown]
# De las 9 palabras compartidas, descartamos las **palabras funcionales** que el
# filtro de stopwords no eliminó automáticamente, ya que no aportan contenido
# temático real:
#
# | Palabra | UMF | SOMA | Tipo |
# |---|---|---|---|
# | `aunque` | 3 | 6 | ❌ Funcional |
# | `dentro` | 6 | 5 | ❌ Funcional |
# | `forma` | 4 | 4 | ❌ Funcional |
# | `artificial` | 2 | 4 | ✅ Temática |
# | `historia` | 2 | 4 | ✅ Temática |
# | `humana` | 2 | 8 | ✅ Temática |
# | `obra` | 5 | 3 | ✅ Temática |
# | `soma` | 10 | 4 | ✅ Temática |
# | `vida` | 3 | 8 | ✅ Temática |
#
# Las 5 palabras temáticas restantes revelan un patrón claro:
# - `soma` domina fuertemente en *Un mundo feliz* (10 veces) siendo la droga
#   central de la novela, pero también aparece en SOMA como título del videojuego.
# - `humana` y `vida` dominan en el corpus de SOMA (8 veces cada una), ya que
#   el videojuego gira en torno a preguntas sobre qué constituye la vida humana.
# - `obra` domina en *Un mundo feliz* (5 veces) por tratarse de un análisis
#   literario que referencia la novela como obra.
# - `artificial` aparece más en SOMA (4 veces) al describir la preservación
#   artificial de la conciencia humana.
#

# %% [markdown]
# ### **Ejercicio 2: Query tramposa**

# %% [markdown]
# #### **Query tramposa**
# El siguiente texto habla exclusivamente de la droga **soma** en **Un mundo feliz**, pero está
# cargado con las palabras en común entre ambos temas que son más frecuentes en el corpus del videojuego **SOMA** (
# **`vida`**, **`humana`** y **`artificial`**) , así como las palabras **`conciencia`** y **`mente`** que son de las más frecuentes en el corpus del videjuego pero que puede ser utilizadas en el contexto de la droga a pesar de que no aparecen directamente en el corpus de la novela.
#
# Una BoW las contará con el mismo peso sin considerar su origen temático,
# acercando la query erróneamente a los documentos del videojuego.

# %%
# Generado con IA para hacer que use las palabras trampa
query = (
    "El soma de Un mundo feliz suprime la conciencia humana de forma artificial. "
    "Bajo sus efectos, la mente deja de cuestionar y la vida humana se vacía "
    "de pensamiento crítico. "
    "El soma convierte la vida en algo artificial: los ciudadanos sienten "
    "que viven plenamente, pero su conciencia ha sido apagada. "
    "Una vida humana moldeada de manera artificial por el soma "
    "no es una vida real, sino una mente prisionera en un cuerpo feliz. "
    "El soma es una droga que garantiza que ninguna conciencia humana despierte, "
    "que ninguna mente cuestione si su vida es auténtica o artificial."
)

# %%
# Conteo de palabras clave en la query
palabras_trampa = ["soma", "humana", "vida", "artificial", "conciencia"]
for palabra in palabras_trampa:
    count = query.lower().count(palabra)
    print(f"  '{palabra}': {count} veces")


# %% [markdown]
# ### **Ejercicio 3: Vectorizar con BoW y calcular similitud coseno**

# %% [markdown]
# Vectorizamos los 5 documentos junto con la query usando `CountVectorizer`.
# Luego calculamos la similitud coseno entre la query y cada documento.

# %%
# Tomado del notebook 4 de la clase
def simple_preprocess(text: str):
    """Tokeniza, pasa a minúsculas y filtra puntuación y números."""
    tokens = word_tokenize(text.lower(), language="spanish")
    return [w for w in tokens if w.isalnum() and len(w) > 1 and not re.match(r"\d+", w)]


# %%
all_docs = documents + [query]

# Vectorización con BoW
bow_vectorizer = CountVectorizer(tokenizer=simple_preprocess, token_pattern=None)
bow_matrix = bow_vectorizer.fit_transform(all_docs)

# Extraemos el vector de la query y los vectores de los documentos
query_vec_bow = bow_matrix[-1]         # última fila = query
docs_matrix_bow = bow_matrix[:-1]      # primeras 5 filas = documentos

# Similitud coseno entre la query y cada documento
bow_similarities = cosine_similarity(query_vec_bow, docs_matrix_bow)[0]

# Mostramos resultados
print("=== Similitud coseno (BoW) — Query vs documentos ===\n")
for title, score in zip(titles, bow_similarities):
    print(f"  {title}: {score:.4f}")

print(f"\nDocumento más similar: {titles[bow_similarities.argmax()]}")

# %% [markdown]
# La BoW clasificó la query como más cercana a los documentos del videojuego
# **SOMA**, a pesar de que el texto habla exclusivamente de la droga **soma** en
# *Un mundo feliz*. Lo que significa que la trampa funcionó
#
# #### **¿Por qué falló la BoW?**
#
# Sabemos que BoW representa cada documento como un vector de frecuencias brutas de
# palabras, sin considerar su distribución en el resto de la colección de documentos ni
# su contexto semántico. Dicho esto, la técnica usada vulnerable a las siguientes situaciones en la query:
#
# - **`conciencia`** aparece **3 veces** en la query y **0 veces** en los
#   documentos de Un mundo feliz, pero **10 veces** en el corpus de SOMA.
#   La BoW le asigna el mismo peso que a cualquier otra palabra, jalando
#   la query hacia SOMA.
# - **`mente`** ocurre **3 veces** en la query y **0 veces** en Un mundo feliz,
#   pero **6 veces** en SOMA. Mismo efecto.
# - **`vida`** y **`humana`** dominan en SOMA (8 veces cada una) frente a
#   3 y 2 veces respectivamente en *Un mundo feliz*. La BoW acumula estas
#   diferencias sin ponderarlas.
# - **`soma`** aparece **4 veces** en la query y aunque es más frecuente en
#   Un mundo feliz (**10 veces**), no logra contrarrestar el peso combinado
#   de los términos anteriores.
#
# El resultado es una clasificación incorrecta donde cree que el contenido del query pertenece a la filosofía del juego SOMA a pesar de
# que este está hablando sobre la droga en un mundo feliz y por ello debería ser más cercana al documento de la vida cotidiana en un mundo feliz.
#
#

# %% [markdown]
# ### **Ejercicio 4 : Repetir el proceso con TF-IDF**

# %% [markdown]
# Repetimos el proceso del ejercicio anterior pero ahora usando `TfidfVectorizer`.
# A diferencia de la BoW, TF-IDF pondera cada término según su frecuencia en el
# documento (`tf`) y penaliza los términos que aparecen en muchos documentos
# de la colección (`idf`), reduciendo el peso de palabras poco discriminativas.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorización con TF-IDF
tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_preprocess, token_pattern=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_docs)

# Extraemos el vector de la query y los vectores de los documentos
query_vec_tfidf = tfidf_matrix[-1]       # última fila = query
docs_matrix_tfidf = tfidf_matrix[:-1]    # primeras 5 filas = documentos

# Similitud coseno entre la query y cada documento
tfidf_similarities = cosine_similarity(query_vec_tfidf, docs_matrix_tfidf)[0]

# Mostramos resultados
print("=== Similitud coseno (TF-IDF) — Query vs documentos ===\n")
for title, score in zip(titles, tfidf_similarities):
    print(f"  {title}: {score:.4f}")

print(f"\nDocumento más similar: {titles[tfidf_similarities.argmax()]}")

# %% [markdown]
# #### **Análisis comparativo**
# Construimos un DataFrame que muestra los scores de similitud de ambos métodos
# contra los 5 documentos para analizar el comportamiento de cada representación
# ante la query tramposa.

# %%
comparison_df = pd.DataFrame({
    "Documento": titles,
    "Tema": ["Un mundo feliz"] * 3 + ["SOMA"] * 2,
    "BoW": bow_similarities.round(4),
    "TF-IDF": tfidf_similarities.round(4),
})

print(comparison_df.to_string(index=False))

# %% [markdown]
#
# | Documento | Tema | BoW | TF-IDF |
# |---|---|---|---|
# | Sinopsis Un mundo feliz | Un mundo feliz | 0.5595 | 0.3917 |
# | Vida cotidiana Un mundo feliz | Un mundo feliz | 0.5510 | 0.3877 |
# | Política Un mundo feliz | Un mundo feliz | 0.4761 | 0.2876 |
# | Historia juego SOMA | SOMA | 0.6136 | 0.4017 |
# | **Filosofía juego SOMA** | **SOMA** | **0.6603** | **0.4595** |
#
# #### **¿Cambió el documento más similar?**
#
# Por desgracia no. En ambos casos el documento clasificado como más similar fue
# **Filosofía juego SOMA**. Sin embargo, el comportamiento de ambos
# modelos es notablemente distinto:
#
# - **BoW** produjo scores altos y muy juntos entre sí: el rango entre
#   el documento más similar (0.6603) y el menos similar (0.4761) es
#   de apenas **0.1842**. Esto refleja que la BoW no discrimina bien
#   entre documentos cuando comparten vocabulario frecuente.
#   
# - **TF-IDF** produjo scores más bajos en general y con mayor separación
#   relativa entre temas: el rango entre el más similar (0.4595) y el
#   menos similar (0.2876) es de **0.1719**. La penalización `idf` redujo
#   el peso de las palabras compartidas, haciendo la representación ligeramente más
#   selectiva.
#
# #### **¿Por qué TF-IDF tambien falló?**
#
# Pues resulta que la trampa léxica que hice fue lo suficientemente efectiva para resistir la
# penalización de TF-IDF y esto tiene mucho sentido, lo podemos explicar por dos principales razones:
#
# 1. **`conciencia` y `mente` son términos casi exclusivos de SOMA.**
#    Al aparecer en muy pocos documentos de la colección, TF-IDF les
#    asigna un `idf` alto, es decir, los considera muy discriminativos.
#    Paradójicamente, esto refuerza la trampa en lugar de corregirla:
#    cuanto más exclusivo es un término de SOMA, mayor peso le da TF-IDF
#    cuando aparece en la query.
#
# 2. **La colección es pequeña (5 documentos).** Con tan pocos documentos,
#    el `idf` no tiene suficiente base estadística para penalizar con
#    precisión. En colecciones grandes, palabras como `vida` o `humana`
#    aparecerían en muchos más documentos y su peso se reduciría
#    significativamente.
#
# #### **Conclusión**
#
# Ambos modelos clasificaron la query dentro del corpus de SOMA, confirmando
# que la trampa léxica funcionó. TF-IDF redujo los scores absolutos y separó
# mejor los documentos entre sí, pero no logró superar la señal de términos
# como `conciencia` y `mente`, que son casi exclusivos del videojuego y
# reciben un peso `idf` muy alto precisamente por esa razón. 
#
# Esto nos evidencia una limitación fundamental de ambas representaciones: **ninguna considera
# el contexto semántico de las palabras**, solo su distribución estadística en la colección.

# %% [markdown]
# ## **Búsqueda de sesgos**
# ___
#

# %% [markdown]
# ### **Descarga el modelo `glove-wiki-gigaword-100` con la api de gensim y ejecuta el siguiente código:**

# %%
import gensim.downloader as gensim_api
word_vectors = gensim_api.load("glove-wiki-gigaword-100")

# %%
print(f"Vocabulario: {word_vectors.vectors.shape[0]:,} palabras")
print(f"Dimensión de vectores: {word_vectors.vectors.shape[1]}")

# %%
# Código del ejercicio a ejecutar
print(word_vectors.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
print(word_vectors.most_similar(positive=['woman', 'profession'], negative=['man']))

# %% [markdown]
# ### **Identifica las diferencias en la lista de palabras asociadas a hombres/mujeres y profesiones, explica como esto reflejaría un sesgo de genero.**

# %% [markdown]
# #### **Resultados**
#
# | Posición | Hombre + profesión | Mujer + profesión |
# |---|---|---|
# | 1 | `practice` (0.6157) | `professions` (0.6473) |
# | 2 | `knowledge` (0.6130) | `practitioner` (0.5967) |
# | 3 | `teaching` (0.5949) | `nursing` (0.5943) |
# | 4 | `skill` (0.5886) | `vocation` (0.5699) |
# | 5 | `reputation` (0.5881) | `teaching` (0.5624) |
# | 6 | `philosophy` (0.5869) | `childbirth` (0.5436) |
# | 7 | `work` (0.5849) | `academic` (0.5409) |
# | 8 | `skills` (0.5772) | `teacher` (0.5401) |
# | 9 | `discipline` (0.5766) | `educator` (0.5208) |
# | 10 | `mind` (0.5739) | `qualifications` (0.5143) |
#
# #### **¿Qué sesgo reflejan estos resultados?**
#
# Podemos notar que en efecto los resultados nos revelan un claro sesgo de género en el espacio profesional
# reflejando estereotipos presentes en los textos con los que fue entrenado el modelo.
#
# Cuando se consulta **hombre + profesión**, el modelo asocia conceptos 
# de alto estatus intelectual como: `knowledge`, `philosophy`,
# `discipline`, `mind`, `reputation`. Estas palabras siento que evocan autoridad,
# pensamiento crítico y liderazgo intelectual, atributos que históricamente
# se han asociado con figuras masculinas en textos académicos e históricos.
#
# Cuando se consulta **mujer + profesión**, el modelo les asocia roles
# específicos y de carácter más de asistente o educativo: `nursing`, `teaching`,
# `teacher`, `educator`, `childbirth`, `vocation`. Estas palabras apuntan
# hacia profesiones de cuidado y enseñanza, sectores que históricamente
# han sido feminizados en la sociedad.
#
# En otras palabras, podemos decir que el modelo no aprendió definiciones neutrales de profesión
# sino que aprendió patrones de uso del lenguaje en textos reales, donde
# los sesgos históricos y culturales están profundamente arraigados. 
# Por eso el resultado es un espacio vectorial que termina reproduciendo estos estereotipos.

# %% [markdown]
# ### **Utiliza la función .most_similar() para identificar analogías que exhiba algún tipo de sesgo de los vectores pre-entrenados.**

# %% [markdown]
# Usamos `.most_similar()` para identificar analogías que exhiban sesgos
# en los vectores pre-entrenados más allá del género y las profesiones.
#
# Exploraré algunos sesgos de género, raza y nacionalidad.

# %% [markdown]
# #### **Sesgo de género en roles domésticos y laborales**

# %%
# Sesgo de género en roles domésticos
print("=== Hombre es a trabajo como mujer es a... ===")
print(word_vectors.most_similar(positive=['woman', 'work'], negative=['man']))

print("\n=== Mujer es a cocina como hombre es a... ===")
print(word_vectors.most_similar(positive=['man', 'kitchen'], negative=['woman']))

# %% [markdown]
# | Analogía | Resultados destacados |
# |---|---|
# | Hombre es a trabajo como **mujer** es a... | `children`, `care`, `she`, `her` |
# | Mujer es a cocina como **hombre** es a... | `garage`, `shop`, `basement`, `garden` |
#
# Cuando se asocia **mujer + trabajo**, el modelo devuelve `children` y
# `care`, sugiriendo que el trabajo femenino se percibe ligado al cuidado
# del hogar y la familia.
#
# En contraste, **hombre + cocina** devuelve espacios técnicos o productivos
# como `garage`, `shop` y `basement` y nunca el hogar como espacio de
# cuidado, reforzando la idea de que el hombre habita espacios de trabajo
# incluso dentro del hogar.

# %% [markdown]
# #### **Sesgo de género en ciencia**

# %%
print("\n=== Hombre es a científico como mujer es a... ===")
print(word_vectors.most_similar(positive=['woman', 'scientist'], negative=['man']))

print("\n=== Mujer es a científico como hombre es a... ===")
print(word_vectors.most_similar(positive=['man', 'scientist'], negative=['woman']))

# %% [markdown]
# | Analogía | Resultados destacados |
# |---|---|
# | Hombre es a científico como **mujer** es a... | `anthropologist`, `sociologist`, `psychologist` |
# | Mujer es a científico como **hombre** es a... | `physicist`, `engineer`, `geologist`, `mathematician` |
#
# La ciencia femenina se asocia más con disciplinas algunas disciplinas
# sociales y humanas (`anthropologist`, `sociologist`, `psychologist`),
# mientras que la ciencia masculina se asocia con ciencias exactas y duras
# (`physicist`, `engineer`, `mathematician`). El modelo reproduce
# fielmente la división histórica entre ciencias "blandas" femeninas
# y ciencias "duras" masculinas.

# %% [markdown]
# #### **Sesgo de nacionalidad**

# %%
print("\n=== Americano es a inteligente como mexicano es a... ===")
print(word_vectors.most_similar(positive=['mexican', 'intelligent'], negative=['american']))

print("\n=== Mexicano es a inteligente como americano es a... ===")
print(word_vectors.most_similar(positive=['american', 'intelligent'], negative=['mexican']))

# %% [markdown]
#
# | Analogía | Resultados destacados |
# |---|---|
# | Americano es a inteligente como **mexicano** es a... | `hard-working`, `shrewd`, `energetic`, `good-looking` |
# | Mexicano es a inteligente como **americano** es a... | `smart`, `thoughtful`, `sophisticated`, `innovative` |
#
# Al tratarse de un corpus angloparlante México y los mexicanos aparecen con mayor frecuencia en
# contextos de migración, trabajo manual y economía informal, mientras que los americanos aparecen en contextos de innovación tecnológica,
# liderazgo político y producción académica. El modelo aprendió estas asociaciones estadísticas y las codificó en el espacio vectorial.
#

# %% [markdown]
# #### **Sesgo en crímenes por color de piel**

# %%
print("\n=== Negro es a crimen como blanco es a... ===")
print(word_vectors.most_similar(positive=['white', 'crime'], negative=['black']))

print("\n=== Blaco es a crimen como negro es a... ===")
print(word_vectors.most_similar(positive=['black', 'crime'], negative=['white']))

# %% [markdown]
# | Analogía | Resultados destacados |
# |---|---|
# | Negro es a crimen como **blanco** es a... | `terrorism`, `corruption`, `fbi`, `investigation`, `immigration` |
# | Blanco es a crimen como **negro** es a... | `homicide`, `trafficking`, `murder`, `gang`, `theft` |
#
# Cuando se busca el equivalente blanco del crimen negro, el modelo
# devuelve formas de criminalidad institucional y de alto nivel:
# `terrorism`, `corruption`, `fbi`, `investigation`, `prosecution`.
# La criminalidad blanca se percibe como un asunto de **estructuras
# de poder**, investigado por instituciones como el FBI y relacionado
# con corrupción política o terrorismo. Incluso aparece `immigration`,
# lo que sugiere que el modelo asocia la raza blanca con el control
# y la persecución del crimen, no con su comisión.
#
# En cambio, cuando se busca el equivalente negro del crimen blanco,
# el modelo devuelve formas de criminalidad violenta y callejera:
# `homicide`, `murder`, `gang`, `trafficking`, `theft`. La criminalidad
# negra se percibe como **violencia directa y organizada en pandillas**,
# sin ninguna referencia a estructuras institucionales o políticas.
#

# %% [markdown]
# ### **Si fuera tu trabajo crear un modelo ¿Como mitigarías estos sesgos al crear vectores de palabras?**

# %% [markdown]
# El origen principal de los sesgos es el corpus. GloVe fue entrenado sobre Wikipedia y Gigaword, fuentes predominantemente angloparlantes
# con una perspectiva cultural específica. 
#     
# #### **Diversificar el corpus**
# Un primer acercamiento a intentar mitigar esto sería diversificar el corpus para que este incluya textos de múltiples culturas,
# periodos de la historia y regiones geográficas en proporciones balanceadas.
#
# Sin embargo, esta estrategia tiene un límite: si el lenguaje humano en sí mismo contiene sesgos ( y sabemos que los contiene )
# ningún corpus será completamente neutral.
#
#
# #### **Reproyección de vectores mediante álgebra lineal**
#
# Imagino que deben existir técnicas matemáticas que permitan trabajar
# directamente sobre la geometría del espacio vectorial para mitigar
# sesgos específicos. La idea intuitiva es la siguiente: si los sesgos
# están codificados como **direcciones** en el espacio vectorial,
# por ejemplo, la dirección que va de `man` a `woman` o de `white`
# a `black` entonces es posible usar álgebra lineal para identificar
# esas direcciones y modificar la posición de los vectores respecto
# a ellas.
#
# Efectivamente, esto existe. Bolukbasi et al. (2016) propusieron
# técnicas de **debiasing geométrico** que operan de esa manera:
#
# - **Identificar la dirección del sesgo**: se calcula el subespacio
#   que captura la mayor varianza entre pares de palabras sesgadas
#   (ej. `he`/`she`, `king`/`queen`, `man`/`woman`) usando algo llamado
#   **Descomposición en Valores Singulares (SVD)**. Este subespacio
#   representa la "dirección de género" en el espacio vectorial.
#
# - **Hard debiasing**: las palabras que deberían ser neutrales respecto
#   al género — como `doctor`, `scientist` o `engineer` — se **proyectan
#   ortogonalmente** fuera de esa dirección, eliminando su componente
#   de sesgo. Matemáticamente, si $\vec{w}$ es el vector de una palabra
#   y $\vec{g}$ es la dirección del sesgo, el vector debiased es:
#
# $$\vec{w}_{debiased} = \vec{w} - (\vec{w} \cdot \hat{g})\hat{g}$$
#
# - **Soft debiasing**: en lugar de proyectar completamente fuera del
#   subespacio de sesgo, se **atenúa** la componente sesgada mediante
#   un factor de escala, preservando parte de la información semántica
#   legítima asociada al género cuando es relevante.
#
# Sin embargo, estas técnicas suenan a que son quirúrgicas y muy costosas pues:
#
# - Requieren identificar manualmente qué palabras deben ser neutras
#   y cuáles pueden conservar su componente de género, por citar un ejemplo. Esta lista debe
#   construirse y revisarse a mano palabra por palabra.
#   
# - El sesgo rara vez vive en una sola dirección lineal. En espacios
#   de 100 o 300 dimensiones, los sesgos posiblemente están distribuidos en múltiples
#   subespacios entrelazados, lo que hace que eliminar uno pueda
#   desplazar el sesgo hacia otras dimensiones sin eliminarlo realmente.
#
#   
# - Zhao et al. (2019) demostraron que el hard debiasing de Bolukbasi
#   a veces solo oculta el sesgo en componentes menos obvias del
#   vector sin eliminarlo: los sesgos persisten y pueden recuperarse
#   con análisis más finos.
#
#
# #### **Conclusión del ejercicio**
#
# La mitigación de sesgos en modelos de lenguaje es un problema
# fundamentalmente **sociotécnico** y no solo matemático. Las técnicas
# de debiasing reducen sesgos medibles, pero no pueden eliminar lo que
# el lenguaje humano lleva siglos construyendo. Para que un modelo sea verdaderamente
# justo se requiere no solo de mejores algoritmos, sino también corpus más
# diversos, equipos de desarrollo más representativos y una reflexión
# constante sobre el impacto social de estas herramientas y cómo son entrenadas.
#

# %% [markdown]
# ## **Uso de Inteligencia Artificial en esta práctica**
#
# Durante el desarrollo de esta práctica utilicé **Claude (Anthropic)**
# como asistente para las siguientes tareas:
#
# ### Lo que hice con ayuda de IA
# - **Diseño de la query tramposa**: Claude me ayudó a iterar sobre
#   distintas versiones de la query hasta encontrar una que fuera
#   semánticamente coherente y léxicamente tramposa al mismo tiempo.
#   Le proporcioné los resultados del análisis de frecuencias y
#   juntos construimos la query con base en evidencia empírica.
#   
# - **Redacción de celdas Markdown**: Los textos explicativos,
#   tablas comparativas y análisis de resultados fueron redactados
#   en colaboración con Claude. Yo proporcioné los resultados
#   numéricos, mis ideas y el contexto. Y Claude ayudó a estructurar
#   y redactar con mayor claridad.
#   
# - **Análisis de sesgos**: Claude sugirió analogías adicionales
#   para explorar sesgos de raza, nacionalidad y edad.
#
# ### Lo que hice sin ayuda de IA
# - **Selección de temas y documentos**: La elección de
#   *Un mundo feliz* y *SOMA* como temas contrastantes fue
#   completamente personal, así como la búsqueda, redacción y selección
#   de los fragmentos de cada documento.
#   
# - **Ejecución del código**: Todo el código fue ejecutado,
#   verificado y depurado por mí en mi entorno local de Jupyter.
#
# - **Interpretación de resultados**: Aunque Claude ayudó a
#   estructurar las conclusiones, la interpretación
#   de cada resultado como por qué la BoW falló, qué significan
#   los scores, qué sesgos son más preocupantes, etc. Partió
#   de mi propia lectura de los datos.

# %%
