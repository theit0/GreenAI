# Estructura del Repositorio

Este repositorio está organizado para facilitar el desarrollo, entrenamiento y despliegue de una red neuronal utilizando PyTorch y Streamlit. A continuación se detalla la estructura de carpetas y archivos:

## 📁 data/

Contiene los datasets utilizados para entrenar y evaluar la red neuronal.

**Ejemplos de contenido:**
- Scripts de preprocesamiento de datos.

## 📁 dev/

Carpeta para el desarrollo experimental del modelo.

**Ejemplos de contenido:**
- `model_dev.ipynb`: Notebooks en Jupyter utilizados para exploración, pruebas y diseño del modelo.
- Scripts adicionales relacionados con pruebas de concepto y prototipos.

## 📁 prod/

Contiene el código de producción y archivos esenciales para ejecutar la aplicación final.

**Contenido:**
- `app.py`: Script principal de la aplicación web. Usa Streamlit para la interfaz gráfica.
- `modelo.pth`: Archivo con los pesos del modelo entrenado en formato PyTorch.
- `README.md`: Documentación del proyecto con instrucciones de uso.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto (e.g., PyTorch, Streamlit).
- `utils.py`: Funciones auxiliares como carga del modelo y preprocesamiento de datos.

---
