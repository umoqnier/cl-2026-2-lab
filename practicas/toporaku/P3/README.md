# Práctica 3: Representación Vectorial

## 1. Instrucciones y Dependencias

Para poder ejecutar la libreta de esta práctica correctamente, es necesario contar con un entorno de Python y las siguientes librerías instaladas:

- `numpy`
- `pandas`
- `scikit-learn`
- Entorno para abrir libretas de Jupyter (como `jupyter notebook`, `jupyterlab` o la extensión de Python en VS Code).

### Pasos para ejecutarla:
1. Abre tu terminal y colócate en este directorio (`toporaku/P3`).
2. Activa tu entorno virtual (si usas uno). Si estás utilizando `uv` u otra herramienta de gestión, asegúrate de que el ambiente esté seleccionado.
3. Instala las dependencias necesarias. Puedes hacerlo usando `pip`:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
4. Abre la libreta `p3_vector_repr.ipynb`:
   ```bash
   jupyter notebook p3_vector_repr.ipynb
   ```
   *O ábrela directamente con el editor de código de tu preferencia.*
5. Ejecuta todas las celdas de forma secuencial, de arriba hacia abajo, para asegurarte de que las variables se carguen correctamente y se generen los resultados de similitud (BoW y TF-IDF).

---

## 2. Declaración de uso de Inteligencia Artificial (LLMs)

Durante el desarrollo de esta entrega, el uso de Modelos de Lenguaje (LLMs) se restringió a un nivel de asistencia básico o **Nivel 1**.

Específicamente, la IA se empleó de forma exclusiva para:
- **Optimización de código en DataFrames**: Sugerir y mejorar la sintaxis de las iteraciones de pandas y el cálculo vectorizado de la similitud del coseno usando `scikit-learn` y `numpy`.
- **Formato y redacción**: La estructuración y creación de este mismo archivo `README.md`.

Todo el análisis reflexivo, la creación de las trampas léxicas y las justificaciones teóricas sobre las diferencias entre TF-IDF e Inverse Document Frequency fueron resueltos por cuenta propia.
