"""Valida que el repositorio tenga la estructura de carpetas y archivos esperada."""

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

ARCHIVOS_REQUERIDOS = [
    "Base_de_datos.csv",
    "requirements.txt",
    "requirements-prod.txt",
    ".gitignore",
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "sonar-project.properties",
    "mlops_pipeline/Cargar_datos.ipynb",
    "mlops_pipeline/eda_analisis.ipynb",
    "mlops_pipeline/ft_eng.py",
    "mlops_pipeline/model_training_evaluation.py",
    "mlops_pipeline/model_deploy.py",
    "mlops_pipeline/model_monitoring.py",
    "mlops_pipeline/app_monitoreo.py",
]


@pytest.mark.parametrize("ruta", ARCHIVOS_REQUERIDOS)
def test_archivo_existe(ruta):
    assert (RAIZ / ruta).is_file(), f"Falta el archivo {ruta}"


def test_jenkinsfile_bonus():
    if not (RAIZ / "Jenkinsfile").is_file():
        pytest.skip("Jenkinsfile (bonus) aún no agregado")
    assert "pipeline" in (RAIZ / "Jenkinsfile").read_text(encoding="utf-8")


def test_scripts_no_vacios():
    for script in (RAIZ / "mlops_pipeline").glob("*.py"):
        assert script.stat().st_size > 0, f"{script.name} está vacío"


def test_dataset_tiene_variable_objetivo():
    encabezado = (RAIZ / "Base_de_datos.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "Pago_atiempo" in encabezado
