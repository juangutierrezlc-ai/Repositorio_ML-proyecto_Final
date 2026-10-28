# Repositorio_ML-proyecto_Final
Este proyecto corresponde al proyecto final de la materia de ML, el cual tiene como finalidad realizar un modelo de Machine Learning bajo un esquema de MLOps, siguiendo lineamintos de colaboracion y despliegue productivo.


# Estrutura de Proyecto
# PROYECTO FINAL: MACHINE LEARNING

**Docente:** Juan Sebastián Parra Sánchez  
**Fecha de entrega:** 11 de noviembre de 2026

## 📁 Estructura del proyecto

```text
PROYECTO FINAL MACHINE LEARNING/
│
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── setup.bat
└── README.md
```

## 📌 Descripción de los archivos

### Carpeta `mlops_pipeline/`

Contiene los archivos relacionados con las etapas del proceso de Machine Learning.

| Archivo | Descripción |
|---|---|
| `Cargar_datos.ipynb` | Carga y lectura del conjunto de datos. |
| `comprension_eda.ipynb` | Comprensión de los datos y análisis exploratorio (EDA). |
| `ft_engineering.py` | Preparación, transformación y creación de características (*Feature Engineering*). |
| `model_training_evaluation.py` | Entrenamiento y evaluación del modelo de Machine Learning. |
| `model_deploy.py` | Implementación o despliegue del modelo entrenado. |
| `model_monitoring.py` | Seguimiento y monitoreo del comportamiento del modelo. |

### Archivos principales

- **`Base_de_datos.csv`:** contiene el conjunto de datos utilizado para desarrollar el proyecto.
- **`requirements.txt`:** almacena las librerías de Python necesarias para ejecutar el proyecto.
- **`.gitignore`:** especifica los archivos y carpetas que Git debe ignorar.
- **`setup.bat`:** archivo de comandos para automatizar tareas de configuración en Windows.
- **`README.md`:** documento principal con la información y las instrucciones del proyecto.

## 🛠️ Tecnologías y herramientas

- Python
- Jupyter Notebook
- Pandas y NumPy
- Scikit-learn
- Git y GitHub

## 🚀 Instalación y ejecución

1. Clonar el repositorio de GitHub.
2. Ingresar a la carpeta del proyecto.
3. Crear y activar un entorno virtual de Python.
4. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

5. Ejecutar los notebooks y scripts correspondientes a cada etapa del proceso de Machine Learning.

## 🎯 Objetivo del proyecto

Desarrollar un proceso de Machine Learning organizado por etapas, desde la carga y comprensión de los datos hasta el entrenamiento, la evaluación, el despliegue y el monitoreo del modelo.


