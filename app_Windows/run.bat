@echo off
TITLE Analizador de Voz IA

ECHO ----------------------------------------------------------------------
ECHO  AVISO IMPORTANTE:
ECHO  Asegurese de tener instalados los drivers de NVIDIA CUDA version 12.8
ECHO  o superior para un correcto funcionamiento, de lo contrario se debe 
ECHO  encargar personalmente de la gestion de librerias
ECHO ----------------------------------------------------------------------
ECHO.


ECHO Activando el entorno virtual...
CALL .venv\Scripts\activate

ECHO Instalando dependencias desde requirements.txt. Esto toma unos minutos
pip install -r requirements.txt

ECHO Entorno activado. Iniciando la aplicacion de Streamlit...
streamlit run streamlit_app.py

pause
