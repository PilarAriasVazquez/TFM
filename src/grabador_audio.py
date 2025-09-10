import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime
import threading

# --- CÓDIGO INICIAL (selección de dispositivo, etc.) ---
print("🎙Dispositivos de entrada disponibles:\n")
for i, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']}")

print("\nSelecciona el índice del dispositivo de entrada:")
device_index = int(input("Índice: "))

# --- CONFIGURACIÓN ---
carpeta_destino = os.path.normpath("C:/Users/pilar/Escritorio/MasterCopia/TFMPersonal/Audios_48samplerate/Scam")
n_muestras = 40
duracion = 10
samplerate = 48000
os.makedirs(carpeta_destino, exist_ok=True)

print(f"\n🎙️ Vamos a grabar {n_muestras} muestras de voz. Cada una durará un máximo de {duracion} segundos.")
print("Habla de forma natural, en español, con frases variadas.")
print("Pulsa ENTER para empezar cada grabación.\n")

def stop_on_enter():
    """Espera a que el usuario pulse ENTER y detiene la grabación."""
    input()
    sd.stop()

# --- Bucle principal ---
i = 0
while i < n_muestras:
    input(f"Pulsa ENTER para empezar a grabar la muestra {i+1}/{n_muestras}...")
    
    stop_thread = threading.Thread(target=stop_on_enter)
    stop_thread.start()
    
    print(f"🎧 Grabando... (Pulsa ENTER de nuevo para detener antes de los {duracion}s)")
    
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device_index)
    sd.wait()
    
    print("🎤 Grabación finalizada.")


    while True:
        # Ahora hay tres opciones: sí, no, o escuchar.
        respuesta = input("¿La grabación es correcta? (s/n/e para escuchar): ").lower().strip()
        
        if respuesta == 's':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = os.path.join(carpeta_destino, f"sample_{i+1}_{timestamp}.wav")
            sf.write(nombre_archivo, audio, samplerate)
            print(f"Muestra guardada en: {nombre_archivo}\n")
            i += 1
            break
        elif respuesta == 'n':
            print("Grabación descartada. Repitiendo la misma muestra...\n")
            break
     
        elif respuesta == 'e':
            print("🔊 Reproduciendo el audio grabado...")
            sd.play(audio, samplerate)
            sd.wait() # Espera a que termine la reproducción
            print("🔊 Reproducción finalizada.")
            # Después de escuchar, el bucle vuelve a empezar para preguntar de nuevo.
        else:
            print("Respuesta no válida. Por favor, introduce 's', 'n' o 'e'.")


print("Grabación completa.")
