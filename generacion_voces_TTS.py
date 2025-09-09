import os
import torch
import torchaudio
import random
from TTS.api import TTS
from torchaudio.transforms import MelSpectrogram, MFCC
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from pathlib import Path
import torch.serialization
from TTS.tts.configs.xtts_config import XttsConfig

# Añadir XttsConfig como clase segura
torch.serialization.add_safe_globals({"TTS.tts.configs.xtts_config.XttsConfig": XttsConfig})


# ============================
# 1. Generar audios falsos con Coqui XTTS
# ============================


audio_dir = Path("Audios_48samplerate")
speaker_wavs = list(audio_dir.rglob("*.wav"))

# todos los audios dentro de Audios_48samplerate

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

texts_fake = [
    "Hola, soy del banco, Necesitamos verificar una transacción sospechosa.",
    "Detectamos un movimiento extraño en tu cuenta, Por favor, confirma tu identidad.",
    "Se ha intentado acceder a tu cuenta desde otro dispositivo, Activa tu token.",
    "Necesitamos tu número de tarjeta para validar el pago.",
    "Buenas tardes, hemos detectado un fallo de seguridad en tu sistema.",
    "Buenas, le llamamos del banco por un posible fraude en su cuenta.",
    "Para evitar el bloqueo de su tarjeta, necesitamos el código CVV.",
    "Su cuenta ha sido comprometida, Necesitamos verificar sus credenciales.",
    "Hola, soy del servicio de verificación, Necesitamos validar sus datos personales.",
    "Hemos detectado una compra sospechosa en su cuenta de Amazon.",
    "Debe validar su identidad para evitar el cierre de su cuenta.",
    "Llamamos del servicio técnico, Su ordenador está infectado.",
    "Necesitamos acceso remoto para reparar un fallo en su sistema.",
    "Tiene un reembolso pendiente, Facilite su número de cuenta.",
    "Para continuar usando su app, confirme los datos por SMS.",
    "La policía está investigando su cuenta, Necesitamos colaboración urgente.",
    "Se ha bloqueado su acceso online por motivos de seguridad.",
    "Su dispositivo está en riesgo, Debe instalar una herramienta oficial.",
    "Para evitar multas, actualice su información fiscal ahora.",
    "Ha ganado un premio, Envíe sus datos para recibirlo.",
    "Le llamamos de la Seguridad Social, Faltan datos en su expediente.",
    "Confirmamos que su tarjeta ha sido clonada, Necesitamos los datos actuales.",
    "Hola, ha recibido una sanción, Evítela accediendo al siguiente enlace.",
    "Detectamos actividad irregular, Necesitamos comprobar el último movimiento.",
    "Hay un cargo sospechoso, Para anularlo, indique su número de cuenta."
] + [
    "Hola, ¿vas a venir hoy a clase?",
    "Estoy terminando el informe. Te lo paso esta tarde.",
    "Acuérdate de comprar leche.",
    "Voy a salir a correr un rato.",
    "Nos vemos en el cine a las 7.",
    "¿Te apetece tomar algo esta tarde?",
    "Voy a hacer la compra, ¿necesitas algo?",
    "He terminado el informe, te lo envío en un rato.",
    "¿Te viene bien si pasamos por tu casa a las seis?",
    "No te olvides de llevar el paraguas.",
    "Mañana tengo reunión temprano, no podré ir al gimnasio.",
    "Estamos organizando una cena el viernes. ¿Te apuntas?",
    "¿Puedes recoger el paquete de la oficina de correos?",
    "He dejado las llaves en la mesa del salón.",
    "El examen de mañana empieza a las 9 en punto.",
    "Voy a preparar algo de cenar, ¿quieres que te guarde?",
    "No me esperes despierto, llegaré tarde.",
    "¿Quieres que te mande las fotos del viaje?",
    "Recuerda que tenemos cita con el médico el jueves.",
    "Te escribo cuando llegue al trabajo.",
    "¿Vienes a la reunión o te conectas online?",
    "Acordamos vernos en la estación a las cinco.",
    "Lleva abrigo, está haciendo mucho frío.",
    "Salgo en diez minutos, te aviso al llegar.",
    "¿Te llamo cuando esté libre o prefieres que te escriba?"
]

output_dir = Path("fake_audios")
output_dir.mkdir(exist_ok=True)

for i, text in enumerate(texts_fake):
    wav = tts.tts(text, speaker_wav=speaker_wavs, language="es")
    torchaudio.save(str(output_dir / f"fake_{i:03}.wav"), torch.tensor([wav]), 24000)