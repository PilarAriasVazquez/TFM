# 🎧 DeepFake Voice Detection – TFM

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Detección de voces falsas mediante modelos de IA y generación de voz TTS**  
> Proyecto desarrollado como parte del **Trabajo de Fin de Máster** en [Nombre de tu universidad].

---

## 🧠 Descripción del proyecto

Breve explicación de lo que hace tu aplicación:  
- ¿Cuál es el problema que resuelve?  
- ¿Por qué es importante?  
- ¿Qué tecnologías usa?  

**Ejemplo:**
> Este proyecto utiliza modelos de **machine learning** y **procesamiento de audio** para detectar posibles voces generadas artificialmente (*deepfake voice*).  
> Además, integra **Hugging Face**, **PyTorch**, **Streamlit** y **ElevenLabs** para la clasificación, generación y visualización de resultados.

---

## ✨ Características principales

- 🎤 **Grabación de audios**: Captura de voz en tiempo real para el dataset.
- 🧩 **Clasificación de voces**: Modelos entrenados con MFCC + embeddings.
- 🧠 **Fine-tuning con Hugging Face**: Optimización de modelos preentrenados.
- 🔊 **Generación de voz TTS**: Integración con **ElevenLabs** para sintetizar voces.
- 🌐 **Interfaz web**: Aplicación interactiva construida con **Streamlit**.
- 📞 **Twilio API** *(opcional)*: Automatización de llamadas de prueba.

---

## 🗂️ Estructura del repositorio

```bash
TFM/
├── notebooks/
│   ├── TextClassification_Hugging_Face.ipynb
│   ├── Text_Classification_Embeddings.ipynb
│   ├── finetuning.ipynb
│   └── hugginface-model.ipynb
├── src/
│   ├── grabador_audio.py
│   ├── clasificacion_mfcc_pytorch.py
│   ├── elevenlabsClient.py
│   ├── generacion_voces_TTS.py
│   ├── transcripciones.py
│   └── llamada_twilio.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── audios/
│   ├── guiones_llamadas_etiquetados.json
│   └── transcripciones.csv
├── requirements.txt
└── README.md


🚀 Instalación y ejecución
1️⃣ Clonar el repositorio

```bash
git clone https://github.com/PilarAriasVazquez/TFM.git
cd TFM
```

2️⃣ Crear un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

4️⃣ Ejecutar la app Streamlit
```bash
streamlit run streamlit_app.py
```

🧪 Uso del proyecto

| Funcionalidad      | Archivo / Comando                      | Descripción                                     |
| ------------------ | -------------------------------------- | ----------------------------------------------- |
| 🎤 Grabar audios   | `python grabador_audio.py`             | Inicia la grabadora y guarda audios etiquetados |
| 🧩 Clasificar voz  | `python clasificacion_mfcc_pytorch.py` | Aplica el modelo para detectar deepfakes        |
| 🔊 Generar voces   | `python generacion_voces_TTS.py`       | Genera audios sintéticos con ElevenLabs         |
| 📜 Transcripciones | `python transcripciones.py`            | Crea un CSV con transcripciones automáticas     |
| 🌐 App web         | `streamlit run streamlit_app.py`       | Lanza la interfaz web                           |


Crea un archivo .env en la raíz con tus credenciales:
```bash
ELEVENLABS_API_KEY=tu_api_key
TWILIO_SID=tu_sid
TWILIO_TOKEN=tu_token
```


🛠️ Tecnologías utilizadas
<p align="center"> <img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow,github" /> <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" width="160"/> </p>

<p align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff6f61,100:6a5acd&height=150&section=footer"/> </p> ```