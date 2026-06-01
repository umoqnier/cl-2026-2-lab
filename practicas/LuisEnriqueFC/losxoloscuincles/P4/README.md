LOS XOLOSCUINCLES INTEGRANTES:
Dehara Encinos Abhner Adhair
Rivas Rodriguez Luis Enrique
Segura González Marthell
# INVESTIGACIÓN (Perplexity)

### ¿Qué es la perplejidad y cómo se calcula?

La perplejidad es una métrica de evaluación intrínseca utilizada en la teoría de la información y el procesamiento de lenguaje natural (PLN) para cuantificar la incertidumbre de un modelo de probabilidad al predecir una muestra. Intuitivamente, representa el factor de ramificación promedio ponderado; es decir, el número promedio de opciones equiprobables que el modelo considera al predecir el siguiente elemento de la secuencia.
Se puede referir como una métrica utilizada para medir el rendimiento de diferentes modelos de lenguaje, esto ayuda a medir la eficiencia de entendimiento de un modelo de lenguaje, principalmente en la predicción de las secuencias de palabras en tareas de n-gramas, al pasar por la prueba un texto no conocido por el modelo si se da una alta perplejidad se dice que el modelo esta "sorprendido" por dicho texto, al no haber podido predecir o reconocer esas frases o secuencias.

### Fórmula Matemática

Dada una secuencia de palabras $W = w_1, w_2, \dots, w_N$, la perplejidad $PP(W)$ de un modelo de lenguaje probabilístico $P$ se define como el inverso geométrico de la probabilidad de la secuencia normalizada por su longitud $N$:
$$PP(W) = P (w_1, w_2, \dots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, \dots, w_N)}}$$
Aplicando la regla de la cadena para la probabilidad conjunta, esto se expresa como:
$$PP(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1, \dots, w_{i-1})}}$$
Dado que el cálculo de la productoria de probabilidades pequeñas provoca un desbordamiento inferior (underflow) en punto flotante, en la práctica se calcula utilizando la entropía cruzada media ($H$). Si la función de pérdida del modelo neuronal es la entropía cruzada categórica, la perplejidad es simplemente la exponenciación de dicha pérdida:
$$PP(W) = \exp\left( - \frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \mid w_{<i}) \right) = e^{H(W)}$$

Otras expresiones que se pueden encontrar como la de hugging face nos indica que la expresión de esta métrica está dada por la fórmula:
$PPL=exp(-\frac{1}{t}∑_{i=1}^t log p_\theta(x_i|x_{<i}))$

Donde:

$p_\theta(x_i|x_{<i})$ es conocida como log-likehood (medida de que tan bien un modelo procesa los datos) en nuestro caso la probabilidad de cierta frase.
Así existen ciertas diferencias entre alguna fuentes respecto a la definición usando diferentes bases para los logaritmos ajustando la ecuación con alguna constante adicional pero siempre se comparte que se trate de una exponencial además del uso de la entropía como aspecto importante.

### Relación entre perplejidad y calidad del modelo

La perplejidad esta íntimamente conectada con la probabilidad vista en clase de las secuencias de frases o n-gramas haciendo que si las predicciones obtienen una menor perplejidad indica que el modelo asigna una mayor probabilidad a las secuencias de texto reales del conjunto de prueba. Esto significa que a mayor probabilidad de una frase menor será la "sorpresa" de nuestro modelo, así podemos decir que son inversamente proporcionales entre sí.
Por lo tanto, un modelo con menor perplejidad tiene una mayor capacidad predictiva y se considera de mejor calidad dentro de ese dominio específico.

#### Ventajas y Limitaciones

•	Ventajas: Es una métrica computacionalmente rápida de evaluar y no requiere la implementación de tareas extrínsecas complejas (downstream tasks) para medir mejoras iterativas durante el entrenamiento.
    En sí tiene muchos usos entre ellos la de como mencionamos medición de rendimiento y por ende comparación entre modelos de lenguaje, generación de texto y usos en modelos para la corrección del lenguaje en diferentes medios.
•	Limitaciones: Las limitaciones que se presentan normalmente radican en su fase de entrenamiento basado en el nivel de vocabulario que pueda tener teniendo así una alta sensibilidad bajo palabras que sean desconocidas del mismo. Su limitación más severa es que no es directamente comparable entre modelos con vocabularios diferentes. Un modelo con un vocabulario más pequeño tendrá artificialmente una perplejidad menor debido a que el espacio de probabilidad es más reducido, lo cual es crítico al comparar un modelo de palabras enteras contra uno de sub-words. Además, una baja perplejidad no siempre garantiza un mejor desempeño en aplicaciones reales (como traducción o resumen).
 ### REFERENCIAS:
Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing (3rd ed. draft). Capítulo 3: N-gram Language Models. Stanford University.

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of Machine Learning Research, 3, 1137-1155.

Shannon, C. E. (1948). A Mathematical Theory of Communication. The Bell System Technical Journal, 27(3), 379-423.

Solís, E. S., & Solís, E. S. (2024, January 8). Perplexity: Resumen de la medida de incertidumbre en el procesamiento del lenguaje natural. Tu Resumen. https://turesumen.com/ciencia/perplexity-resumen-de-la-medida-de-incertidumbre-en-el-procesamiento-del-lenguaje-natural/#google_vignette

Admin. (2026, February 4). Perplejidad (en modelos de lenguaje) - Avahi. Avahi. https://avahi.ai/glossary/perplejidad-en-modelos-de-lenguaje/?lang=es

Morgan, A. (2025, November 17). Perplexity for LLM evaluation. Comet. https://www.comet.com/site/blog/perplexity-for-llm-evaluation/

Perplejidad de los modelos de longitud fija · Hugging Face. (n.d.). https://huggingface.co/docs/transformers/v4.56.2/es/perplexity

Learn Statistics Easily. (2024, August 6). Qué es: Perplejidad: comprensión de la métrica. LEARN STATISTICS EASILY. https://es.statisticseasily.com/glosario/%C2%BFQu%C3%A9-es-la-perplejidad-entendiendo-la-m%C3%A9trica%3F/#google_vignette

GeeksforGeeks. (2025, July 23). Log likelihood. GeeksforGeeks. https://www.geeksforgeeks.org/data-science/log-likelihood/

# CREACIÓN DE MODELOS DE LENGUAJE

(Los códigos de este apartado se encuentran en los documentos de .py y .ipynb respectivamente, estos solo cómo código y sin ser ejecutados con una breve explicación de lo que se realiza).

Se adjuntan los links de descarga de los modelos creados

Modelo base:
https://drive.google.com/file/d/1hCJWC69WmTSk-Gc8kjS2xsqBuIuT8Xw4/view?usp=drive_link

Modelo subword:
https://drive.google.com/file/d/1raIGQsiMLVyPjRtBiS5QDeuuacyQFrku/view?usp=drive_link

## ANALISÍS ENTRE MODELOS

Una vez terminados los modelos los resultados que se observaron bajo las prueba propuesta fueron:

### PRUEBA DE PERPLEJIDAD:

|Métrica|Modelo Base|Modelo subword|
-----------------------------------
|Perplejidad|41.05|162118.52|
-----------------------------------
|OOV|57.74%|50.49%|
-----------------------------------

Como se observa en la prueba quien presenta menor perplejidad frente a la prueba de génesis fue el modelo base, sin embargo, de igual manera aunque con una diferencia no tan considerable el modelo base hace mayor uso de OOV's (out of vocabulary) o palabras fuera del vocabulario haciendo una reducción propia del vocabulario que podría usar o predecir, a pesar de ello observamos que el uso de OOV en ambas pruebas del modelo subword resultó con un porcentaje no tan alejado del otro por lo cual podríamos asumir que para este caso a pesar de una perplejidad alta existe en cierta instancia una reducción de uso de OOV, esto podría decirnos que el vocabulario usado por el modelo subword se incrementa un pequeño porcentaje de palabras posibles al dividirlas en secuencias más pequeñas que las del base, esto se evidencia en la medida de tamaño de vocabulario entre ambos siendo el subword claramente más amplio al menos 5 veces más que el de modelo base.

Esto claramente nos indica una mayor presencia de diversidad en la tokenización de palabras, sin embargo, dados los resultados obtenidos podemos decir de igual forma que durante el entrenamiento hubo algún factor que nos dio un resultado de perplejidad mayor a pesar de dicha diversidad a la hora de pasar por la prueba de corpus desconocido quizá afectando en la dicción de ciertas palabras o en su estructura afectando la sintaxis o la gramática de ciertas secuencias de palabras.

Una forma de mejorar los modelos en cada uno de sus resultados finales podría ser repitiendo las épocas, pues como sabemos estos modelos van aprendiendo poco a poco, quizá también darles durante dicha fase un poco de  más diversidad en textos, que estos incluso puedan no estar relacionados pero al ser de un mismo idioma aumentar las probabilidades o posibles secuencias a formar, así al dar textos con probables conceptos o contextos completamente distintos este pueda de mejor forma asignar pesos a cada secuencia, pasando por varias y reduciendo asimismo el uso de OOV en ambos, incluso mejorando con ambas recomendaciones el bajar la perplejidad en modelo subword.

Las ventajas que esto supone en ambos casos es que probablemente en el uso del modelo base siempre se generaría una perplejidad baja pero aunado a una desventaja general de poca diversidad lingüística y de muchos tokens desconocidos, las ventajas de un modelo subwords radican en lo contrario en una mayor diversidad en las secuencias probables, pero con una perplejidad muchísimo más alta en presencia de textos cuyas secuencias sean desconocidas o nuevas, incluso carentes de sentido para el modelo, a pesar de ello una ventaja de ambos es su posible mejora a través de distintas técnicas de tokenización o de probabilidades y entrenamientos que nutran el repertorio de cada uno de ellos, aunque, por supuesto esto significaría más tiempo de entrenamiento hasta tener resultados más satisfactorios en el análisis de textos nuevos.

En resumen consideramos que aquel que tiene mejor desempeño es el modelo subword, por el simple hecho que a pesar de que su perplejidad sea más alta a diferencia del modelo base, se infiere en el menor uso de OOV's, en general podemos decir que la baja perplejidad del modelo base es artificial debido a su uso mayor de tokens OOV reduciendo la diversidad propia del vocabulario y no siendo consistente con lo que se esperaría de un modelo de lenguaje al intentar predecir o incluso generar texto nuevo basado en su entendimiento del lenguaje.

