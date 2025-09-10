# DeepFake Voice Detection – TFM

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Detección de DeepFake Voice con análisis de contenido fraudulento mediante un enfoque multimodal de Deep Learning**  
> Proyecto desarrollado como parte del **Trabajo de Fin de Máster** en la Universidad Complutense de Madrid.

---

## Descripción del proyecto

En este proyecto se va a desarrollar un sistema para la detección de voces falsas generadas por Inteligencia Artificial (IA) y la identificación de posibles estafas telefónicas a través de un enfoque multimodal de Deep Learning (DL). La propuesta combina dos componentes principales: un modelo acústico entrenado para discriminar entre voz humana y voz sintética, y un modelo de procesamiento de lenguaje natural aplicado sobre la transcripción de la llamada para reconocer patrones característicos de fraude. 

## Características principales

- **Grabación de audios**: Captura de voz en tiempo real para el dataset.
- **Fine-tuning con Hugging Face**: Optimización de modelos preentrenados de clasificación de audio.
- **Clasificación de voces**: Modelos entrenados con MFCC + embeddings.
- **Generación de voz TTS**: Integración con **ElevenLabs** para sintetizar voces a partir de textos.
- **Interfaz web**: Aplicación interactiva construida con **Streamlit**.
- **Twilio API**: Automatización de llamadas con agente de IA como demo.

---

## Estructura del repositorio

```bash
TFM/
├── notebooks/
│   ├── Audio_Classification_Fine-Tuning.ipynb
|   ├── Audio_Classification_GenerateDataSet.ipynb
│   ├── Audio_Classification_MFCC.ipynb
|   ├── TextClassification_Hugging_Face.ipynb
│   └── Text_Classification_Embeddings.ipynb
├── src/
│   ├── api_utils.py
│   ├── clasificacion_mfcc_pytorch.py
│   ├── grabador_audio.py
│   ├── generacion_voces_TTS.py
│   ├── transcripciones.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── audios/
│   ├── guiones_llamadas_etiquetados.json
│   └── transcripciones.csv
├── requirements.txt
└── README.md
```

## Instalación y ejecución
Clonar el repositorio

```bash
git clone https://github.com/PilarAriasVazquez/TFM.git
cd TFM
```

Crear un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

Instalar dependencias
```bash
pip install -r requirements.txt
```

Ejecutar la app Streamlit
```bash
cd app
streamlit run streamlit_app.py
```

## Uso del proyecto

| Funcionalidad      | Archivo / Comando                      | Descripción                                     |
| ------------------ | -------------------------------------- | ----------------------------------------------- |
| Grabar audios   | `python grabador_audio.py`             | Inicia la grabadora y guarda audios etiquetados |
| Clasificar voz  | `python clasificacion_mfcc_pytorch.py` | Aplica el modelo para detectar deepfakes        |
| Generar voces   | `python generacion_voces_TTS.py`       | Genera audios sintéticos con ElevenLabs         |
| Transcripciones | `python transcripciones.py`            | Crea un CSV con transcripciones automáticas     |
| App web         | `streamlit run streamlit_app.py`       | Lanza la interfaz web                           |


Crea un archivo .env en la raíz con tus credenciales:
```bash
ELEVENLABS_API_KEY=tu_api_key
TWILIO_SID=tu_sid
TWILIO_TOKEN=tu_token
TWILIO_FROM_NUMBER=tu_telefono_origen
TWILIO_TO_NUMBER=tu_telefono_destino
TWILIO_URL=tu_url_twilio
```


## 🛠️ Tecnologías utilizadas
<p align="center"> <img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow,github" /> <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" width="160"/> </p>

<p align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff6f61,100:6a5acd&height=150&section=footer"/> </p> ```
