import os

from dotenv import load_dotenv
from elevenlabs import VoiceSettings, play, save
from elevenlabs.client import ElevenLabs
from io import BytesIO
import uuid

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

VOICE_ID_JUANFRAN ="toH9H8ONdalie9j9Vg0c"
VOICE_ID_PILI ="pLesqRdHEp1qWzGCqD9s"
VOICE_ID_ALVARO ="l8HZqnTh1I9Uhgiw7KCl"

def clone_voice(name, audios):
   audio_files = [BytesIO(open(audio, "rb").read()) for audio in audios]
   client.voices.ivc.create(
         name = name,
         files = audio_files
   )
      

def text_to_speech(text, voice_id):
   
   return client.text_to_speech.convert (
      text = text,
      voice_id = voice_id,
      model_id = "eleven_v3",
      output_format = "mp3_44100_128",
      voice_settings=VoiceSettings(
         stability= 0.0,
         similarity_boost= 1.0,
         style= 0.0,
         use_speaker_boost=True
      )
   )


def text_to_speech_file(text: str, voice_id: str, save_file_path: str) -> str:
    
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_v3",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
         )
    )
    
    save_file_path = f"{save_file_path}.mp3"
    
    save(response, save_file_path)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    
    return save_file_path


if __name__ == "__main__":
   text = "Hola mundo esto es una prueba"
   voice_id = VOICE_ID_JUANFRAN
   text_to_speech_file(text=text, voice_id=voice_id, save_file_path="holamundo2")