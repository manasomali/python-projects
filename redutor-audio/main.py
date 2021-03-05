from pydub import AudioSegment
from os import path
import os
from scipy.io.wavfile import read
import pathlib
import glob
from tqdm import tqdm

# pegando caminho e nomes dos audios - entrada
directorywav = path.dirname(path.realpath(__file__)) + '\input\*.wav'
diretorios_audios = (glob.glob(directorywav))

inputdirectory = path.dirname(path.realpath(__file__)) + '\input'
outputdirectory = path.dirname(path.realpath(__file__)) + '\output'

diretorio_atual = pathlib.Path().parent.absolute()
nomes_audios = []
for nome in os.listdir(inputdirectory):
    if ".wav" in nome:
        nomes_audios.append(nome)

# se audio > x segundos segundo corta ele
corte = input("Informe em que momento será o corte dos áudios em segundos: ")
for nome_audio in tqdm(nomes_audios):
    samplerate, data = read(path.join(inputdirectory, nome_audio))
    duration = len(data)/samplerate
    if duration > int(corte):
        song = AudioSegment.from_wav(path.join(inputdirectory, nome_audio))
        cut_seconds = int(corte) * 1000
        first_x_seconds = song[:cut_seconds]
        first_x_seconds.export(outputdirectory+"/"+nome_audio.replace('\*.wav', ''), format="wav")
