#! /bin/bash

python3 -m venv .venv
pip install -r requirements.txt
streamlit run streamlit_app.py