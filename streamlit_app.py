import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import queue
import threading
from functools import partial
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit.components.v1 as components

# --- 1. Constantes y Configuración (sin cambios) ---
# ... (Tu sección de constantes y carga de centroides se mantiene igual) ...
MODELO_DETECCION = "jayalakshmikopuri/deepfake-audio-detector-v12"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
MODELO_TRANSCRIPCION = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"
SAMPLE_RATE = 16000
ENERGY_THRESHOLD_DB = -40
SILENCE_THRESHOLD_S = 1.0
MAX_RECORDING_S = 10.0
CHUNK_INTERVAL_S = 0.5
BLOCKSIZE_VAD = int(SAMPLE_RATE * CHUNK_INTERVAL_S)

try:
    centroid_non = pd.read_csv("Centroides/centroid_non.csv").to_numpy()
    centroid_scam = pd.read_csv("Centroides/centroid_scam.csv").to_numpy()
    st.success("Centroides de texto cargados correctamente.")
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos de centroides.")
    st.stop()


# --- 2. Carga de Modelos (sin cambios) ---
@st.cache_resource
def load_models():
    # ... (Tu función de carga de modelos se mantiene igual) ...
    st.info("Cargando modelos de IA... Esto puede tardar unos minutos la primera vez.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    detector_voz = pipeline("audio-classification", model=MODELO_DETECCION, device=device)
    transcriptor = pipeline("automatic-speech-recognition", model=MODELO_TRANSCRIPCION, device=device)
    embedder = SentenceTransformer(MODELO_EMBEDDINGS, device=device)

    st.success("¡Modelos cargados correctamente!")
    return {"detector": detector_voz, "transcriptor": transcriptor, "embedder": embedder}

def predict_label(texts, model, centroid_scam, centroid_non):
    emb = model.encode(texts, normalize_embeddings=True)
    sim_scam = cosine_similarity(emb, centroid_scam).ravel()
    sim_non = cosine_similarity(emb, centroid_non).ravel()
    preds = (sim_scam > sim_non).astype(int)
    margin = sim_scam - sim_non
    return preds, margin


# --- 3. Lógica de Audio y Procesamiento (MODIFICADA) ---
def get_audio_devices():
    devices = sd.query_devices()
    return [dev['name'] for dev in devices if dev['max_input_channels'] > 0]

def audio_callback(indata, frames, time_info, status, audio_queue):
    if status: print(f"⚠️ Status de audio: {status}")
    audio_queue.put(indata.copy())
    
def log_message(log_queue, message):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_queue.put(f"[{timestamp}] {message}")

# --- CAMBIO: La función de procesamiento ahora también recibe la cola de alertas ---
def procesar_bloque_de_voz(audio_chunk, models, log_queue, alert_queue):
    log_message(log_queue, "-"*50)
    log_message(log_queue, f"Procesando audio de {len(audio_chunk)/SAMPLE_RATE:.2f} segundos...")

    # --- CAMBIO: Reordenamos y clarificamos los logs ---
    # 1. Transcripción
    resultado_transcripcion = models["transcriptor"](audio_chunk, generate_kwargs={"language": "spanish"})
    texto_transcrito = resultado_transcripcion['text'].strip()
    log_message(log_queue, f"📜 Texto Transcrito: '{texto_transcrito}'")

    # 2. Análisis de Audio (¿Quién habla?)
    resultado_deteccion = models["detector"]({"sampling_rate": SAMPLE_RATE, "raw": audio_chunk}, top_k=2)
    mejor_resultado_audio = max(resultado_deteccion, key=lambda x: x['score'])
    etiqueta_audio = mejor_resultado_audio['label']
    confianza_audio = mejor_resultado_audio['score']
    
    # Mapeamos a etiquetas consistentes
    veredicto_audio = "IA Generada" if etiqueta_audio.lower() in ['spoof', 'aivoice', 'fake'] else "Humano"
    log_message(log_queue, f"🎤 Análisis de Voz: {veredicto_audio} (Confianza: {confianza_audio:.2f})")
    
    # 3. Análisis de Texto (¿Es sospechoso?)
    if texto_transcrito:
        preds_texto, margin = predict_label(
            [texto_transcrito], models["embedder"], centroid_scam, centroid_non
        )
        veredicto_texto = "SCAM" if preds_texto[0] == 1 else "NO SCAM"
        log_message(log_queue, f"🤔 Análisis de Texto: {veredicto_texto} (Margen: {margin[0]:.4f})")
    else:
        veredicto_texto = "NO SCAM" # Asumimos no scam si no hay texto

    # --- CAMBIO: Lógica para enviar alertas al pop-up ---
    UMBRAL_CONFIANZA_IA = 0.85 # Solo alertar si la confianza en IA es alta
    if veredicto_audio == "IA Generada" and confianza_audio > UMBRAL_CONFIANZA_IA:
        alert_queue.put(f"¡Alerta de Voz! La voz parece generada por IA (Confianza: {confianza_audio:.2f}).")
    
    if veredicto_texto == "SCAM":
        alert_queue.put(f"¡Alerta de Texto! El contenido de la conversación parece una estafa (Margen: {margin[0]:.4f}).")


def audio_processing_thread(models, audio_queue, log_queue, alert_queue, stop_event):
    # ... (El resto de esta función se mantiene exactamente igual) ...
    estado = "ESPERANDO_VOZ"
    buffer_grabacion = np.array([], dtype=np.float32)
    contador_silencio_s = 0.0
    
    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.1)
            energia_actual = np.sqrt(np.mean(data**2))
            es_habla = 20 * np.log10(energia_actual) > ENERGY_THRESHOLD_DB if energia_actual > 0 else False

            if estado == "ESPERANDO_VOZ" and es_habla:
                log_message(log_queue, "Comienzo de voz detectado. Grabando...")
                buffer_grabacion = data.squeeze()
                estado = "GRABANDO"
            
            elif estado == "GRABANDO":
                buffer_grabacion = np.concatenate((buffer_grabacion, data.squeeze()))
                if not es_habla:
                    contador_silencio_s += CHUNK_INTERVAL_S
                else:
                    contador_silencio_s = 0.0

                duracion_total_s = len(buffer_grabacion) / SAMPLE_RATE
                procesar_ahora = (contador_silencio_s >= SILENCE_THRESHOLD_S or duracion_total_s >= MAX_RECORDING_S)
                
                if procesar_ahora:
                    if len(buffer_grabacion) > SAMPLE_RATE * 0.5:
                        # Pasamos la cola de alertas a la función de procesamiento
                        procesar_bloque_de_voz(buffer_grabacion, models, log_queue, alert_queue)
                    estado = "ESPERANDO_VOZ"
        except queue.Empty:
            continue
        
# --- 4. Interfaz de Streamlit (MODIFICADA) ---
st.title("🎙️ Analizador de Voz y Texto en Tiempo Real")

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
    st.session_state.run_data = {}
    st.session_state.log_history = []

models = load_models()

st.sidebar.header("Controles")
input_devices = get_audio_devices()
selected_device = st.sidebar.selectbox("Selecciona el dispositivo de entrada", input_devices, disabled=st.session_state.is_running)

col1, col2 = st.sidebar.columns(2)
start_button = col1.button("▶️ Iniciar", type="primary", use_container_width=True, disabled=st.session_state.is_running)
stop_button = col2.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.is_running)

if start_button:
    # --- CAMBIO: Creamos también la cola de alertas ---
    st.session_state.log_history = []
    audio_q = queue.Queue()
    log_q = queue.Queue()
    alert_q = queue.Queue() # Nueva cola para los pop-ups
    stop_ev = threading.Event()
    
    callback_with_queue = partial(audio_callback, audio_queue=audio_q)

    try:
        stream = sd.InputStream(
            device=selected_device, samplerate=SAMPLE_RATE, channels=1,
            dtype='float32', blocksize=BLOCKSIZE_VAD, callback=callback_with_queue
        )
        stream.start()
        
        thread = threading.Thread(
            target=audio_processing_thread,
            # Pasamos la cola de alertas al hilo
            args=(models, audio_q, log_q, alert_q, stop_ev)
        )
        thread.start()
        
        st.session_state.run_data = {"stream": stream, "thread": thread, "stop_event": stop_ev, "log_queue": log_q, "alert_queue": alert_q}
        st.session_state.is_running = True
        st.rerun()
    except Exception as e:
        st.error(f"No se pudo iniciar el stream: {e}")
        st.session_state.is_running = False

if stop_button:
    # ... (Esta sección de detener no cambia) ...
    if st.session_state.is_running and st.session_state.run_data:
        st.session_state.run_data["stop_event"].set()
        st.session_state.run_data["thread"].join(timeout=1)
        st.session_state.run_data["stream"].stop()
        st.session_state.run_data["stream"].close()
    
    st.session_state.is_running = False
    st.session_state.run_data = {}
    st.rerun()

st.header("Registro de Actividad")
status_placeholder = st.empty()
console_placeholder = st.empty()

def main_loop():
    # --- CAMBIO: Extraemos y mostramos las alertas ---
    if st.session_state.is_running:
        status_placeholder.success(f"🔴 Escuchando en '{selected_device}'...")
        alert_queue = st.session_state.run_data.get("alert_queue")
        if alert_queue:
            while not alert_queue.empty():
                message = alert_queue.get()
                st.toast(message, icon='🚨') # ¡Aquí se muestra el pop-up!

        log_queue = st.session_state.run_data.get("log_queue")
        if log_queue:
            new_logs = []
            while not log_queue.empty():
                new_logs.append(log_queue.get())
            st.session_state.log_history = new_logs + st.session_state.log_history
    else:
        status_placeholder.info("⚪ Detenido. Presiona 'Iniciar' para comenzar.")

    log_text = "\n".join(st.session_state.log_history)
    console_placeholder.code(log_text, language='log', height=400)

main_loop()

if st.session_state.is_running:
    time.sleep(0.2) # Pequeña pausa para no sobrecargar
    st.rerun()
