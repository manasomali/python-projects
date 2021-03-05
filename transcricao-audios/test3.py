from dotenv import load_dotenv
import os
import speech_recognition as sr
from os import path
import asyncio
import websockets
import ssl
import nest_asyncio
import json
from multiprocessing import Pool, freeze_support
import time
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == '__main__':

        #limpa arquivos txt
    with open("output/transcricoes_wit.txt", "w") as txt_file:
        txt_file.write(str(""))
    with open("output/transcricoes_vosk.txt", "w") as txt_file:
        txt_file.write(str(""))
    
    # nomes dos arquivos
    inputdirectory = path.dirname(path.realpath(__file__)) + '\input'
    diretorios_audios = []
    nomes_audios = []
    for path, subdirs, files in os.walk(inputdirectory):
        for name in  sorted(files, key=natural_keys):
            diretorios_audios.append(os.path.join(path, name))
            
    for diretorio_audio in diretorios_audios:
        nomes_audios.append(diretorio_audio.split("\\").pop().replace(".wav", ""))
        
        
        