# Práctica 1 - Niveles Lingüísticos


## Instrucciones de ejecución

Para correr el software (el notebook `p1_niv_ling.ipynb`), asegúrate de tener instalado Python y Jupyter Notebook (o JupyterLab). Puedes instalar las dependencias necesarias y lanzar el notebook ejecutando los siguientes pasos en tu terminal:

1. Clona el repositorio y navega hasta la carpeta de la práctica:
   ```bash
   cd /ruta/a/cl-2026-2-lab/practicas/toporaku/P1/
   ```

2. Instala las dependencias necesarias (se recomienda usar un entorno virtual):
   ```bash
   pip install notebook pandas matplotlib scikit-learn sklearn-crfsuite nltk requests rich
   ```

3. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Abre el archivo `p1_niv_ling.ipynb` desde la interfaz de Jupyter y ejecuta las celdas en orden.

## Dependencias adicionales: `difflib`

Para esta práctica, se ha utilizado la biblioteca **`difflib`** de Python. Es una herramienta integrada (built-in) en la librería estándar de Python, por lo que no necesitas instalarla a través de `pip`. Su uso principal en este código es a través de la función `get_close_matches`, la cual nos permite obtener aproximaciones y corregir posibles errores tipográficos al buscar términos que no existen exactamente en los diccionarios o corpus cargados. Ayuda a ofrecer la coincidencia más cercana en lugar de simplemente devolver un string vacío.

## Uso de LLM

**Categoría de uso: 4**

Se acudió a Modelos de Lenguaje Grande (LLM, por sus siglas en inglés) como herramienta de asistencia para generar el código de las gráficas (plots). Esto debido a que utilizo LLMs como apoyo de memoria o asistente de programación cuando olvido la sintaxis y el uso estructurado de las herramientas de `matplotlib` para la creación de subplots y la visualización de los datos (distribución de longitudes de morfemas y categorías) y por supuesto la generación de este archivo.
