from pydub import AudioSegment
from os import path
import os
from auditok import split
from scipy.io.wavfile import read
import pathlib
import glob

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
corte = input("Informe até quantos segundo ocorrerá a separação dos áudios (segundos): ")
for nome_audio in nomes_audios:
    samplerate, data = read(path.join(inputdirectory, nome_audio))
    duration = len(data)/samplerate
    if duration > int(corte):
        song = AudioSegment.from_wav(path.join(inputdirectory, nome_audio))
        cut_seconds = int(corte) * 1000
        first_x_seconds = song[:cut_seconds]
        first_x_seconds.export(inputdirectory+"/"+nome_audio.replace('\*.wav', ''), format="wav")

# divide os audios em regioes e salva cada regiao de cada audio para uma pasta separada
names_file_saved = []
for nome_audio in nomes_audios:
    cont=0
    audio_regioes = split(path.join(inputdirectory, nome_audio))
    os.makedirs(path.join(outputdirectory,nome_audio.replace('.wav', '')), exist_ok=True)
    for region in audio_regioes:
        region.save(path.join(outputdirectory,nome_audio.replace('.wav', ''))+'/'+nome_audio.replace('.wav', '_')+str(cont)+".wav")
        cont=cont+1

# pegando caminho e nomes dos audios - saida
diretorios_audios_saida = []
for path, subdirs, files in os.walk(outputdirectory):
    for name in files:
        diretorios_audios_saida.append(os.path.join(path, name))
        
# adiciona 1s de silencio no inicio dos audios para melhor processamento do wit
silent_segment = AudioSegment.silent(duration=1000)
cont=0
for audio_saida in diretorios_audios_saida:
    audio = AudioSegment.from_wav(audio_saida)
    final_audio = silent_segment + audio + silent_segment
    final_audio.export(audio_saida, format="wav")

