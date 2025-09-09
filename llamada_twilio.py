from twilio.rest import Client

# Credenciales de Twilio
account_sid = '*****************'  
auth_token = '******************'    
client = Client(account_sid, auth_token)

# Números de teléfono
from_number = '‪+1**********‬'  # El número de Twilio que se ha comprado
to_number = '‪+34*********‬'    # El número móvil al que se llama

# URL del archivo de audio 
audio_url = '************************'

# Hacer la llamada
call = client.calls.create(
    to=to_number,
    from_=from_number,
    url=f'***************'  # Reproduce el archivo de audio
)

print(f'Call SID: {call.sid}')

