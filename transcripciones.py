import whisper
import os
import glob

# --- Configuración ---
# 1. Elige el tamaño del modelo de Whisper. Opciones: "tiny", "base", "small", "medium", "large".
#    "base" es un buen punto de partida (rápido y bastante preciso).
#    Usa "medium" o "large" para mayor precisión si tienes una buena GPU.
MODEL_SIZE = "large"


# 2. Especifica la carpeta donde tienes guardados tus archivos de audio.
AUDIO_FOLDER = "Audios_48samplerate/Normal"

# 3. Nombre del archivo de salida que se generará.
OUTPUT_FILE = "Audios_48samplerateTranscripciones/Normal/metadata.csv"
# --------------------


def transcribe_audios():
    """
    Busca archivos de audio en la carpeta especificada, los transcribe usando Whisper
    y guarda los resultados en un archivo metadata.csv.
    """
    # Carga el modelo de Whisper
    print(f"Cargando el modelo '{MODEL_SIZE}' de Whisper...")
    try:
        model = whisper.load_model(MODEL_SIZE)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de tener PyTorch instalado (`pip install torch`) si tienes una GPU NVIDIA.")
        return

    # Comprueba si la carpeta de audios existe
    if not os.path.isdir(AUDIO_FOLDER):

        print(f"Error: La carpeta '{AUDIO_FOLDER}' no fue encontrada.")
        print("Por favor, crea la carpeta y coloca tus archivos de audio dentro.")
        return

    # Busca todos los archivos de audio compatibles en la carpeta
    # Puedes añadir más extensiones si las necesitas (ej: "*.flac")
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.wav")) + \
                  glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")) + \
                  glob.glob(os.path.join(AUDIO_FOLDER, "*.m4a"))

    if not audio_files:
        print(f"No se encontraron archivos de audio en la carpeta '{AUDIO_FOLDER}'.")
        return

    print(f"Se encontraron {len(audio_files)} archivos de audio. Empezando transcripción...")


    # Abre el archivo de salida en modo 'append' (añadir) para no sobreescribir
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for audio_file in audio_files:
            print(f"Procesando {audio_file}...")
            
            # Realiza la transcripción
            try:
                result = model.transcribe(audio_file, language="es", fp16=False)
                transcription = result["text"].strip()
                
                # Obtiene solo el nombre del archivo, sin la ruta
                filename = os.path.basename(audio_file)
                
                # Escribe la línea en el formato para Piper
                f.write(f"{filename}|{transcription}\n")
                
            except Exception as e:
                print(f"No se pudo procesar el archivo {audio_file}. Error: {e}")

    print("-" * 20)
    print(f"¡Proceso completado! Las transcripciones han sido guardadas en '{OUTPUT_FILE}'.")
    print("¡Recuerda revisar el archivo para corregir posibles errores!")


if __name__ == "__main__":
    transcribe_audios()