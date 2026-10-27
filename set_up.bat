@echo off
REM ============================================================
REM  setup.bat - Configuracion del entorno virtual del proyecto
REM  Proyecto Final Machine Learning - MLOps Pipeline
REM ============================================================

setlocal
set VENV_DIR=venv
set KERNEL_NAME=mlops_pipeline

echo.
echo ============================================
echo   Configurando el entorno del proyecto
echo ============================================
echo.

REM --- 1. Verificar que Python este instalado ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo Descargalo desde https://www.python.org/downloads/
    echo y marca la opcion "Add Python to PATH" al instalar.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] %%i encontrado.

REM --- 2. Verificar que exista requirements.txt ---
if not exist requirements.txt (
    echo [ERROR] No se encontro requirements.txt en esta carpeta.
    echo Ejecuta este archivo desde la raiz del repositorio.
    pause
    exit /b 1
)

REM --- 3. Crear el entorno virtual si no existe ---
if exist %VENV_DIR%\Scripts\activate.bat (
    echo [OK] El entorno virtual "%VENV_DIR%" ya existe, se reutilizara.
) else (
    echo [..] Creando entorno virtual "%VENV_DIR%"...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
)

REM --- 4. Activar el entorno virtual ---
call %VENV_DIR%\Scripts\activate.bat
echo [OK] Entorno virtual activado.

REM --- 5. Actualizar pip ---
echo [..] Actualizando pip...
python -m pip install --upgrade pip

REM --- 6. Instalar dependencias ---
echo [..] Instalando librerias de requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias. Revisa el mensaje anterior.
    pause
    exit /b 1
)
echo [OK] Librerias instaladas.

REM --- 7. Registrar el entorno como kernel de Jupyter ---
echo [..] Registrando el kernel de Jupyter "%KERNEL_NAME%"...
python -m ipykernel install --user --name=%KERNEL_NAME% --display-name "Python (%KERNEL_NAME%)"

echo.
echo ============================================
echo   Entorno listo
echo ============================================
echo.
echo Para activarlo en una nueva terminal usa:
echo     %VENV_DIR%\Scripts\activate
echo.
echo En los notebooks selecciona el kernel: Python (%KERNEL_NAME%)
echo.

endlocal
pause