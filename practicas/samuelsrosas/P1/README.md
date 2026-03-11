# Práctica 1: Exploración de Niveles del Lenguaje

Contiene el desarrollo de la Práctica 1 para la asignatura de Lingüística Computacional. El objetivo es explorar los niveles fonético y morfológico mediante el uso de herramientas de procesamiento de lenguaje natural en Python.

## Contenido

- **Fonética**: Implementación de un modelo de Campos Aleatorios Condicionales (CRF) para aproximar la transcripción fonológica (IPA) de palabras en español que no se encuentran en un lexicón predefinido.
- **Morfología**: Análisis estadístico comparativo de tres lenguas (Italiano, Ruso e Inglés) basado en el ratio de morfemas por palabra e índices de flexión/derivación.

## Archivos

- `Practica01LC.ipynb`: Notebook interactivo con explicaciones detalladas y visualizaciones.
- `Practica01LC.py`: Script de Python con el código fuente de la práctica.

## Requisitos y Dependencias

Para ejecutar esta práctica de forma local o en un entorno virtual, asegúrate de tener instalado Python 3.10+ y las siguientes bibliotecas:

```bash
pip install pandas requests sklearn-crfsuite matplotlib seaborn rich
