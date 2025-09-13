#! /bin/bash

python3 -m venv .venv
echo "Activando el entorno virtual"
source .venv/bin/activate

echo "Instalando dependencias desde requirements.txt. Este proceso se puede alargar unos minutos"
pip install -r requirements.txt

echo "Iniciando la aplicación de Streamlit"
streamlit run streamlit_app.py