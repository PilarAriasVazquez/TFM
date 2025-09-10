import os

from dotenv import load_dotenv
from elevenlabs import VoiceSettings, save
from elevenlabs.client import ElevenLabs
from io import BytesIO
from typing import List
from twilio.rest import Client

load_dotenv()


TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
URL = os.getenv("TWILIO_URL")

class ElevenLabsAPI:
   def __init__(self):
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        raise ValueError("No existe la variable de entorno ELEVENLABS_API_KEY")
    self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    def clone_voice(self, name:str, audios:List[str]):
        """
       Esta función utiliza la API de ElevenLabs para clonar una voz a partir de uno o varios archivos de audio.

        Parámetros:
        ----------
        name : str
            Nombre que se le asignará a la voz clonada.
    
        audios : list
            Lista de rutas de archivos de audio (por ejemplo, en formato .mp3 o .wav) que se utilizarán
            como referencia para entrenar y generar la nueva voz.
        """
       
        audio_files = [BytesIO(open(audio, "rb").read()) for audio in audios]
        self.client.voices.ivc.create(name = name, files = audio_files)
    
      

    def text_to_speech(self, text: str, voice_id: str, save_file_path: str) -> str:
        """
        Esta función utiliza la API de ElevenLabs para convertir un texto en un archivo de audio,
        usando una voz específica, y lo guarda en el disco.

        Parámetros:
        ----------
        text : str
            El texto que se desea convertir a voz.
        
        voice_id : str
            El identificador único de la voz que se utilizará para la conversión.
        
        save_file_path : str
            Ruta base y nombre del archivo donde se guardará el audio generado.
            La función añadirá automáticamente la extensión `.opus`.

        Retorno:
        --------
        str
            Devuelve la ruta completa del archivo de audio guardado.
        """

        response = self.client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="opus_48000_32",
            text=text,
            model_id="eleven_v3",
            voice_settings = VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            )
        )

        save(response, f"{save_file_path}.opus")
        print(f"{save_file_path}: A new audio file was saved successfully!")
        return save_file_path

class TwilioAPI():
    def __init__(self):
        self.account_id = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not self.account_id or not self.auth_token:
            raise ValueError("Faltan los credenciales de Twilio en el archivo de configuración")
        self.client = client = Client(self.account_id, self.auth_token)

    def make_twilio_call(self, to_number:str = TO_NUMBER, from_number:str = FROM_NUMBER, url:str = URL) -> None:
        """
        Esta función utiliza la API de Twilio para realizar una llamada telefónica automatizada.

        Parámetros:
        ----------
        to_number : str, opcional
            Número de teléfono de destino, incluyendo el código de país.
            Por defecto, toma el valor de la constante `TO_NUMBER`.

        from_number : str, opcional
            Número de teléfono registrado en Twilio desde el cual se realizará la llamada.
            Por defecto, toma el valor de la constante `FROM_NUMBER`.

        url : str, opcional
            URL que Twilio utilizará para obtener instrucciones sobre qué reproducir durante la llamada.
            Por defecto, toma el valor de la constante `URL`.

        Retorno:
        --------
        None
            La función no devuelve ningún valor. Sin embargo, imprime en consola el `Call SID`
            generado por Twilio, que es un identificador único para la llamada.
        """

        if not to_number or from_number:
            raise ValueError("Introduce los números de teléfono en el archivo de configuración")

        call = self.client.calls.create(
            to = to_number, 
            from_= from_number,
            url = url  # Reproduce el archivo de audio
        )

        print(f'Call SID: {call.sid}')
