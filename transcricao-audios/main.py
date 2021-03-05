from dotenv import load_dotenv
import os
import speech_recognition as sr
from os import path
import asyncio
import websockets
import ssl
import nest_asyncio
import json
import time
import re
from multiprocessing import Pool, freeze_support
import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# codigo do wit
def transcricaoWit(diretorio_audio):
    # setup variaveis do ambiente
    load_dotenv('.env')
    WIT_AI_KEY=os.getenv('WIT_AI_KEY')
    
    transcricao=""
    r = sr.Recognizer()
    with sr.AudioFile(diretorio_audio) as source:
        audio = r.record(source)
        try:
            transcricao = r.recognize_wit(audio, key=WIT_AI_KEY)
        except sr.UnknownValueError:
            transcricao = "erro de entendimento"
        except sr.RequestError:
            transcricao = "erro na requisicao"
        
        # espera 60 segundos pois a api é limitada em 60 requisiçoes por minuto
        time.sleep(20)
        # escreve resultado das transcrições no arquivo output/transcricoes_wit.txt
        with open("output/transcricoes_wit.txt", "a") as txt_file:
            txt_file.write(str(diretorio_audio.split("\\").pop().replace(".wav", "")) + "_" + str(transcricao) + "\n")
               
        return transcricao

# função async para requisição
def startTranscricaoVosk(diretorio_audio):
    asyncio.get_event_loop().run_until_complete(transcricaoVosk(diretorio_audio))
    
async def transcricaoVosk(diretorio_audio):
    async with websockets.connect('wss://api.alphacephei.com/asr/pt/', ssl=ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)) as websocket:
        response=[]
        wf = open(diretorio_audio, "rb")
        while True:
            data = wf.read(8000)

            if len(data) == 0:
                break

            await websocket.send(data)
            response.append(await websocket.recv())

        await websocket.send('{"eof" : 1}')
        response.append(await websocket.recv())
        
        transcricao=""
        for res in response:
            data = json.loads(res)
            text = data.get("text","")
            if text != "":    
                transcricao = transcricao + " " + text
        
        transcricao=transcricao.strip()
        # escreve resultado das transcrições no arquivo output/transcricoes_vosk.txt
        with open("output/transcricoes_vosk.txt", "a") as txt_file:
            txt_file.write(str(diretorio_audio.split("\\").pop().replace(".wav", "")) + "_" + str(transcricao) + "\n")
    
        return response


if __name__ == '__main__':
    freeze_support()
    print("Transcrever com:")
    print("\t 1 - Wit")
    print("\t 2 - Vosk")
    print("\t 3 - Wit e Vosk")
    opcao = input("Escolha uma opção: ")
    
    inicio = time.time()
    
    #limpa arquivos txt
    with open("output/transcricoes_wit.txt", "w") as txt_file:
        txt_file.write(str(""))
    #with open("output/transcricoes_vosk.txt", "w") as txt_file:
    #    txt_file.write(str(""))
    
    # nomes dos arquivos
    inputdirectory = path.dirname(path.realpath(__file__)) + '\input'
    diretorios_audios = []
    nomes_audios = []
    #cont=0
    #maximo=10
    for path, subdirs, files in os.walk(inputdirectory):
        for name in  sorted(files, key=natural_keys):
            diretorios_audios.append(os.path.join(path, name))
            nomes_audios.append('input/'+name)
    #        cont+=1
    #        if(cont==maximo):
    #             break
    #    if(cont==maximo):
    #         break
    # codigo do wit executado em multiprocessos
    if((int(opcao) == 1) or (int(opcao) == 3)):
        with Pool() as pool:
            transcricoes_wit = list(tqdm.tqdm(pool.imap(transcricaoWit, diretorios_audios), total=len(diretorios_audios)))
           
    # codigo do vosk executado em multiprocessos
    if((int(opcao) == 2) or (int(opcao) == 3)):
        with Pool() as pool:
            transcricoes_vosk = list(tqdm.tqdm(pool.imap(startTranscricaoVosk, diretorios_audios), total=len(diretorios_audios)))
        
    fim = time.time()
    print("")
    print("Tempo de processamento (seg):", fim - inicio)