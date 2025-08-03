import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

# PARA VER DISPOSITIVOS DE ENTRADA DISPONIBLES
"""
import sounddevice as sd

print("🎙️ Dispositivos de entrada disponibles:\n")
for i, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']}")

# IMPORTANTE: Cambia el índice del dispositivo en la línea 27

"""
# CONFIGURACIÓN
carpeta_destino = "C:/Users/pilar/OneDrive/Escritorio/MasterCopia/TFMPersonal/Audios_48samplerate/Normal"  # se creará si no existe
n_muestras = 40              # número de grabaciones
duracion = 10                # duración de cada muestra en segundos
samplerate = 48000          # compatible con Tortoise TTS

# Crear carpeta si no existe
# os.makedirs(carpeta_destino, exist_ok=True)

print(f"🎙️ Vamos a grabar {n_muestras} muestras de voz. Cada una durará {duracion} segundos.")
print("Habla de forma natural, en español, con frases variadas.")
print("Pulsa ENTER para empezar cada grabación./n")

for i in range(n_muestras):
    input(f"👉 Pulsa ENTER para grabar la muestra {i+1}/{n_muestras}...")
    print("🎧 Grabando...")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate,channels=1, dtype='float32', device=10)
    sd.wait()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = os.path.join(carpeta_destino, f"sample_{i+1}_{timestamp}.wav")
    sf.write(nombre_archivo, audio, samplerate)
    print(f"✅ Muestra guardada en: {nombre_archivo}/n")

print("✅ Grabación completa. Ya puedes usar los audios con Tortoise TTS.")
