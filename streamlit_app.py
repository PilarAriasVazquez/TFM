"""
Aplicación de Streamlit para la detección en tiempo real de voces generadas por IA y análisis de texto.
Utiliza modelos preentrenados para identificar voces sintéticas y evaluar la probabilidad de que el contenido textual sea fraudulento.
Guarda los resultados en un archivo de log y muestra alertas en la interfaz de usuario.
"""

# Importamos librerías necesarias

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

# --- Constantes y Configuración --- VAD = Voice Activity Detection

MODELO_DETECCION = "juanfran-hr/wav2vec2-base-scam-es-common_voice-elevenlabs"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
MODELO_TRANSCRIPCION = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"
SAMPLE_RATE = 16000 # Frecuencia de muestreo estándar para modelos de audio
ENERGY_THRESHOLD_DB = -40 # Umbral de energía en dB para detectar voz
SILENCE_THRESHOLD_S = 1.0 # Tiempo en segundos para considerar el final de una frase
MAX_RECORDING_S = 10.0 # Duración máxima de grabación en segundos
CHUNK_INTERVAL_S = 0.5 # Intervalo de análisis en segundo -> Granularidad con la que el sistema escucha; Si el ordenador tiene buen rendimiento, se puede bajar a 0.3s o 0.2s para mejorar 
BLOCKSIZE_VAD = int(SAMPLE_RATE * CHUNK_INTERVAL_S) # Cantidad de muestreas de audio que la librería soundevice procesa en cada bloque
LOG_FILENAME = "funcionamiento_log.txt" # Archivo donde se guardan los logs

# Cargamos los centroides para la clasificación de texto
try:
    centroid_non = pd.read_csv("Centroides/centroid_non.csv").to_numpy()
    centroid_scam = pd.read_csv("Centroides/centroid_scam.csv").to_numpy()
    st.success("Centroides de texto cargados correctamente.")
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos de centroides.")
    st.stop()


# --- Carga de Modelos ---
@st.cache_resource # Por el funcionamiento de streamlit, que no cargue todo cada vez que se presiona un botón
def load_models():
    """
    Carga y retorna los modelos necesarios para la detección de voz y análisis de texto.
    input: None
    output: Diccionario con los modelos cargados.

    """
    st.info("Cargando modelos de IA... Esto puede tardar unos minutos la primera vez.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector_voz = pipeline("audio-classification", model=MODELO_DETECCION, device=device)
    transcriptor = pipeline("automatic-speech-recognition", model=MODELO_TRANSCRIPCION, device=device)
    embedder = SentenceTransformer(MODELO_EMBEDDINGS, device=device)
    st.success("¡Modelos cargados correctamente!")
    return {"detector": detector_voz, "transcriptor": transcriptor, "embedder": embedder}


# --- Predicción de texto ---

def predict_label(texts, model, centroid_scam, centroid_non):
    """
    Predice si los textos son SCAM o NO SCAM basándose en la similitud con centroides predefinidos.
    input:
        texts: Lista de textos a clasificar.
        model: Modelo de embeddings para convertir textos en vectores.
        centroid_scam: Vector del centroide para la clase SCAM.
        centroid_non: Vector del centroide para la clase NO SCAM.
    output:
        preds: Lista de predicciones (1 para SCAM, 0 para NO SCAM).
        margin: Lista de márgenes de confianza (similitud SCAM - similitud NO SCAM).
    """
    # Obtenemos los embeddings normalizados
    emb = model.encode(texts, normalize_embeddings=True)
    # Calculamos similitudes con ambos centroides
    sim_scam = cosine_similarity(emb, centroid_scam).ravel()
    sim_non = cosine_similarity(emb, centroid_non).ravel()
    # Predicción basada en la mayor similitud
    preds = (sim_scam > sim_non).astype(int)
    # Cálculo del margen de confianza
    margin = sim_scam - sim_non
    return preds, margin


# --- Obtenemos lista de dispositivos disponibles para el análisis (seleccionar Output Cable VB)  ---
def get_audio_devices():
    """ 
    Retorna una lista de nombres de dispositivos de entrada de audio disponibles.
    input: 
        None
    output: 
        Lista de nombres de dispositivos.
    """
    devices = sd.query_devices()
    return [dev['name'] for dev in devices if dev['max_input_channels'] > 0] # filtramos solo dispositivos de entrada, es decir, max_input_channels > 0 

# --- Añade a la cola los audios para el procesamiento ---
def audio_callback(indata, frames, time_info, status, audio_queue):
    """
    Callback para el stream de audio. Coloca los datos de audio en una cola para su procesamiento.
    input:
        indata: Datos de audio entrantes.
        frames: Número de frames en el bloque.
        time_info: Información temporal del bloque.
        status: Estado del stream.
        audio_queue: Cola para enviar los datos de audio al hilo de procesamiento.
    output: 
        None (los datos se envían a través de la cola).
        """
    if status: print(f"Status de audio: {status}")
    audio_queue.put(indata.copy())

# --- Guardar información ---
def log_message(log_queue, message):
    """
    Añade un mensaje de log con timestamp a la cola de logs que se muestra en la interfaz web.
    input:
        log_queue: Cola para enviar mensajes de log a la UI.
        message: Mensaje de log a añadir.
    output: 
        None
    """
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_entry = f"[{timestamp}] {message}"
    log_queue.put(log_entry)

def write_detection_to_log_file(timestamp, veredicto_audio, confianza_audio, veredicto_texto, margin_texto, texto_transcrito):
    """
    Guarda los resultados de la detección en un archivo de log.
    input:
        timestamp: Timestamp de la detección.
        veredicto_audio: Resultado de la detección de voz.
        confianza_audio: Confianza de la detección de voz.
        veredicto_texto: Resultado del análisis de texto.
        margin_texto: Margen de confianza del análisis de texto
        texto_transcrito: Texto transcrito del audio.
    output:
        None
    """
    log_entry = (
        f"[{timestamp}] "
        f"VOZ={veredicto_audio} (Conf={confianza_audio:.2f}), "
        f"TEXTO={veredicto_texto} (Margen={margin_texto:.4f}), "
        f"TRANSCRIPCION='{texto_transcrito}'"
    )
    try:
        with open(LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"ERROR AL ESCRIBIR EN LOG: {e}")


# --- Lógica y procesamiento de la aplicación ---
def procesar_bloque_de_voz(audio_chunk, models, log_queue, alert_queue):
    """
    Ejecuta los modelos de detección y transcripción sobre un bloque de audio y maneja los resultados.

    input:
        audio_chunk: Array numpy con el audio a procesar.
        models: Diccionario con los modelos cargados.
        log_queue: Cola para enviar mensajes de log a la UI.
        alert_queue: Cola para enviar alertas a la UI.
    output:
        None 
    """
     
    # Mensajes de estado solo para la interfaz de la app
    log_message(log_queue, "-"*50)
    log_message(log_queue, f"Procesando audio de {len(audio_chunk)/SAMPLE_RATE:.2f} segundos...") # Relaciona el número de muestras de audio con la frecuencia de muestreo. 
    # Ej: Si se ha grabado un fragmento de 3 segundos, y la frecuencia de muestreo es 16000 Hz, entonces el número de muestras es 3 * 16000 = len(audio_chunk) = 48000 muestras.

    # Pasa el audio en crudo al modelo para detección de voz generada por IA, devuelve las dos etiquetas más probables: [{'score': 0.98, 'label': 'AIVoice'}, {'score': 0.02, 'label': 'HumanVoice'}]
    resultado_deteccion = models["detector"]({"sampling_rate": SAMPLE_RATE, "raw": audio_chunk}, top_k=2)
    # El score más alto es la etiqueta predicha para este fragmento de audio 
    mejor_resultado_audio = max(resultado_deteccion, key=lambda x: x['score'])
    # guardamos etiqueta y confianza
    etiqueta_audio = mejor_resultado_audio['label']
    confianza_audio = mejor_resultado_audio['score']
    print(f"etiqueta_audio: {etiqueta_audio}, y confianza: {confianza_audio}")
    # Decidimos el veredicto final basándonos en la etiqueta. Como hemos probado diferentes modelos y algunos usan etiquetas distintas, consideramos varias posibles etiquetas para voz IA
    veredicto_audio = "IA Generada" if etiqueta_audio.lower() in ['spoof', 'aivoice', 'fake'] else "Humano"

    # Obtenemos las transcripcioens del audio 
    resultado_transcripcion = models["transcriptor"](audio_chunk, generate_kwargs={"language": "spanish"})
    texto_transcrito = resultado_transcripcion['text'].strip()
    
    if texto_transcrito:
        # Si hay texto, analizamos si es SCAM o NO SCAM
        preds_texto, margin = predict_label([texto_transcrito], models["embedder"], centroid_scam, centroid_non)
        veredicto_texto = "SCAM" if preds_texto[0] == 1 else "NO SCAM"
        # Guardamos el margen de confianza
        margin_texto = margin[0]
    else:
        veredicto_texto = "NO DETERMINADO"
        margin_texto = 0.0

    #  Mostramos la información completa en la UI
    log_message(log_queue, f"Análisis de Voz: {veredicto_audio} (Confianza: {confianza_audio:.2f})")
    log_message(log_queue, f"Análisis de Texto: {veredicto_texto} (Margen: {margin_texto:.4f})")
    log_message(log_queue, f"Texto Transcrito: '{texto_transcrito}'")

    #  Guardamos solo los resultados clave en el archivo de log
    timestamp_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_detection_to_log_file(timestamp_actual, veredicto_audio, confianza_audio, veredicto_texto, margin_texto, texto_transcrito)

    # Lógica de alertas, si detecta voz IA o SCAM, envía un mensaje a la cola de alertas para que la UI lo muestre
    UMBRAL_CONFIANZA_IA = 0.85
    if veredicto_audio == "IA Generada" and confianza_audio > UMBRAL_CONFIANZA_IA:
        alert_queue.put(f"¡POSIBLE VOZ ARTIFICIAL DETECTADA!\n\nConfianza: {confianza_audio:.2f}")
    if veredicto_texto == "SCAM":
        alert_queue.put(f"¡CONTENIDO SOSPECHOSO DETECTADO!\n\nEl texto parece fraudulento (Confianza: {margin_texto:.4f}).")


def audio_processing_thread(models, audio_queue, log_queue, alert_queue, stop_event):
    """
    Escucha continuamente el audio, detecta bloques de voz y decide cuándo procesarlos. Gestiona el estado de grabación y el buffer de audio.
    Es decir, es un grabador de voz automático, que graba solo cuando hay voz y corta cuando hay silencio.
    Trabaja en un hilo separado para no bloquear la interfaz de usuario.
    input:
        models: Diccionario con los modelos cargados.
        audio_queue: Cola desde la que se reciben los datos de audio.
        log_queue: Cola para enviar mensajes de log a la UI.
        alert_queue: Cola para enviar alertas a la UI.
        stop_event: Evento para detener el hilo cuando sea necesario.   
    output:
        None
    """
    estado = "ESPERANDO_VOZ"
    buffer_grabacion = np.array([], dtype=np.float32) # Buffer para almacenar el audio grabado
    contador_silencio_s = 0.0 # Contador de tiempo de silencio en segundos para ver cuándo cortar la grabación y procesar el audio, evitando fragmentos muy largos
    while not stop_event.is_set(): # Mientras no se pulse el botón de detener, en cuyo caso activa el stop_event
        try: 
            # data es una lista de números, cada número es una "foto" de la altura (amplitud) de la onda en ese instante
            # si hay silencio, está llena de valores muy pequeños cercanos a 0, si hay voz, son valores mucho más grandes, tanto positivos como negativos
            data = audio_queue.get(timeout=0.1) # Espera hasta 0.1s por nuevos datos de audio
            # La energía del audio es una medida de cuán "grandes" son estos números en promedio. 
            # RMS (Root Mean Square) es una forma común de calcular esta energía. Es de ingeniería de sonido, y se obtiene un solo número que representa la "potencia/volumen" promedio del trozo de audio
            energia_actual = np.sqrt(np.mean(data**2)) # Calcula la energía RMS del bloque de audio: Un número que indica cuán "fuerte" es el sonido en este bloque
            # Se trata en dB (decibelios), porque el oído humano no percibe el volumen de forma lienal, sino logarítmica -> si lo tratamos así es más intuitivo
            # Convertimos a dB con la fórmula estándar el valor RMS a dB: 20 * log10(RMS)
            # Energy_threshold_db es un umbral que hemos definido para decidir si hay voz o no. En audio digital, el volumen máximo es 0dB, y todo lo demás negativo. Un silencio absoluto, podría ser -90dB
            # Por hacerlo más robusto frente a ruidos ambientales, se define en -40dB
            es_habla = 20 * np.log10(energia_actual) > ENERGY_THRESHOLD_DB if energia_actual > 0 else False # Determina si hay voz basándose en el umbral de energía
            if estado == "ESPERANDO_VOZ" and es_habla: # Si detecta voz, comienza a grabar
                log_message(log_queue, "Comienzo de voz detectado. Grabando...")
                buffer_grabacion = data.squeeze() # Inicia el buffer con el primer bloque de voz
                estado = "GRABANDO"
            elif estado == "GRABANDO": # Si ya está grabando, sigue añadiendo bloques al buffer
                buffer_grabacion = np.concatenate((buffer_grabacion, data.squeeze())) # Añade el nuevo bloque al final del buffer 
                if not es_habla: # Si el trozo actual es silencio, suma 0.5s al contador, si es voz lo resetea; Esto es para cortar la grabación cuando hay silencio
                    contador_silencio_s += CHUNK_INTERVAL_S
                else:
                    contador_silencio_s = 0.0
                duracion_total_s = len(buffer_grabacion) / SAMPLE_RATE # Duración total de la grabación en segundos
                procesar_ahora = (contador_silencio_s >= SILENCE_THRESHOLD_S or duracion_total_s >= MAX_RECORDING_S) # Decide si es momento de procesar el audio -> comprueba si se ha cumplido alguna de las dos condciones: a) la pausa ya es de 1s o b) la grabación total ya dura 10s, para no saturar el sistema
                if procesar_ahora: # si hay que procesar el audio
                    if len(buffer_grabacion) > SAMPLE_RATE * 0.5: # si hay al menos 0.5 segundos de audio grabado
                        procesar_bloque_de_voz(buffer_grabacion, models, log_queue, alert_queue) # Procesa el bloque de voz grabado, enviamos el buffer completo, los modelso, y las colas de log y alertas
                    estado = "ESPERANDO_VOZ" # Vuelve al estado inicial
        except queue.Empty:
            continue
        
# --- Interfaz de Streamlit ---
st.title("Analizador de Voz y Texto en Tiempo Real")

if 'is_running' not in st.session_state: # Variable para controlar si el sistema está activo o no, solo se ejecuta la primera vez que se carga la app. Prepara la memorai de la sesión
    st.session_state.is_running = False
    st.session_state.run_data = {}
    st.session_state.log_history = []

models = load_models()  # Cargamos los modelos (solo la primera vez que se inicia la app gracias a @st.cache_resource)

if "alert_message" not in st.session_state: # Variable para controlar si hay una alerta activa en la UI. Se inicializa la primera vez que se carga la app
    st.session_state.alert_message = None
# Mostramos los dispositivos de audio disponibles en un desplegable en la barra lateral
st.sidebar.header("Controles")
input_devices = get_audio_devices()
selected_device = st.sidebar.selectbox("Selecciona el dispositivo de entrada", input_devices, disabled=st.session_state.is_running)
# Información sobre el archivo de log
st.sidebar.info(f"Los resultados de detección se guardan en '{LOG_FILENAME}'")
col1, col2 = st.sidebar.columns(2)
# Botones de iniciar y detener
start_button = col1.button("▶ Iniciar", type="primary", use_container_width=True, disabled=st.session_state.is_running)
stop_button = col2.button("⏹ Detener", use_container_width=True, disabled=not st.session_state.is_running)
if start_button: # Si se pulsa el botón de iniciar
    # Creamos objetos necesarios para la captura en segundo plano: las colas de comunicación y la señal de parar 
    st.session_state.log_history = [] # Reseteamos el historial de logs en la UI
    audio_q = queue.Queue() # Cola para enviar datos de audio desde el callback al hilo de procesamiento
    log_q = queue.Queue() # Cola para enviar mensajes de log a la UI
    alert_q = queue.Queue() # Cola para enviar alertas a la UI
    stop_ev = threading.Event() # Evento para detener el hilo de procesamiento cuando se pulse el botón de detener
    callback_with_queue = partial(audio_callback, audio_queue=audio_q) # Creamos el callback con la cola de audio ya definida
    try:
        # Iniciamos el stream de audio sobre el dispositivo seleccionado; 
        # # channel = audio mono (simplifica el procesamineto y reduce a la mitad la cantidad de datos a procesar)
        # # dtype float32 porque los modelos funcionan mejor con este formato; Tipo de dato que se usa para cada muestra de audio, es la calidad. El ordenador representa la onda de sonido como una secuencia de números
        # # blocksize = número de muestras que se procesan en cada bloque, definido al inicio en función del intervalo de chunk
        # # callback = función que se llama cada vez que hay un nuevo bloque de audio, en este caso nuestra función que añade el bloque a la cola
        stream = sd.InputStream(device=selected_device, samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCKSIZE_VAD, callback=callback_with_queue)
        stream.start() # Comienza a capturar audio en segundo plano, llamando al callback cada bloque
        thread = threading.Thread(target=audio_processing_thread, args=(models, audio_q, log_q, alert_q, stop_ev)) # Creamos el hilo de procesamiento
        thread.start() # Inicia el hilo de procesamiento
        # Guardamos los objetos necesarios en el estado de la sesión para poder detenerlos luego
        st.session_state.run_data = {"stream": stream, "thread": thread, "stop_event": stop_ev, "log_queue": log_q, "alert_queue": alert_q}
        st.session_state.is_running = True # Indicamos que el sistema está activo
        st.rerun() # Recarga la app para actualizar el estado de los botones
    except Exception as e:
        st.error(f"No se pudo iniciar el stream: {e}")
        st.session_state.is_running = False
if stop_button: # Si se pulsa el botón de detener
    if st.session_state.is_running and st.session_state.run_data: # Si el sistema está activo y hay datos en la sesión
        st.session_state.run_data["stop_event"].set() # Señalamos al hilo que debe parar
        st.session_state.run_data["thread"].join(timeout=1) # Esperamos a que el hilo termine
        st.session_state.run_data["stream"].stop() # Detenemos el stream de audio
        st.session_state.run_data["stream"].close() # Cerramos el stream
    st.session_state.is_running = False # Indicamos que el sistema ya no está activo
    st.session_state.run_data = {} # Limpiamos los datos de la sesión
    st.rerun()  # Recarga la app para actualizar el estado de los botones
st.header("Registro de Actividad") # Sección para mostrar logs y alertas
status_placeholder = st.empty() # Placeholder para mensajes de estado
console_placeholder = st.empty() # Placeholder para la consola de logs
def main_loop(): 
    if st.session_state.is_running: # Si el sistema está activo 
        status_placeholder.success(f"Escuchando en '{selected_device}'...") # Mensaje de estado del dispositovo seleccionado
        # Obtenemos la cola de alertas
        alert_queue = st.session_state.run_data.get("alert_queue")
        # Si hay alertas en la cola, las mostramos como 'toasts' (en steramlit, son notificaciones emergentes)
        if alert_queue:
            while not alert_queue.empty():
                message = alert_queue.get()
                st.toast(message, icon='🚨') 
    else:
        status_placeholder.info("Detenido. Presiona 'Iniciar' para comenzar.")

    # La lógica para actualizar la consola de logs 
    if st.session_state.is_running:
        log_queue = st.session_state.run_data.get("log_queue")
        if log_queue:
            new_logs = []
            while not log_queue.empty():
                new_logs.append(log_queue.get())
            st.session_state.log_history = new_logs + st.session_state.log_history # Añade los nuevos logs al principio del historial para que no tener que estar 

    log_text = "\n".join(st.session_state.log_history)
    console_placeholder.code(log_text, language='log', height=400)
main_loop()
if st.session_state.is_running and not st.session_state.alert_message: # Si el sistema está activo, ejecutamos el bucle principal cada 0.2s para actualizar la UI
    time.sleep(0.2)
    st.rerun()