from os import path
import numpy as np
import os
from natsort import natsorted
# pegando caminho e nomes
inputdirectory = path.dirname(path.realpath(__file__)) + '\input'

arquivos_txt = []
for path, subdirs, files in os.walk(inputdirectory):
    for name in files:
        arquivos_txt.append(name)

# input do usuario
print("Opções: " + str(arquivos_txt))
arquivo_txt = input("Informe o nome (com extensão) do arquivo a ser organizado: ")

f = open(inputdirectory+"/"+arquivo_txt, "r")
resultados = (f.readlines())
f.close()
    
# pega as transcricoes
transcricoes = []
for linha in resultados:
    if('_' in linha):
        nome_transcricao=linha.split('\t')
        numero = nome_transcricao[0].split('_').pop()
        if(nome_transcricao[1]!="\n"):
            transcricoes.append([nome_transcricao[0],nome_transcricao[1].replace("\n", "")])

# cria um csv com as transcricoes em ordem
transcricoes = natsorted(transcricoes)
np_transcricoes = np.asarray(transcricoes)
np.savetxt("output/"+arquivo_txt.replace(".txt", "")+".csv", np_transcricoes, delimiter="_", fmt='%s')

# agrupa as transcricoes
transcricoes_agrupadas=[]
transcricao_agrupada = ""
nome_temp=transcricoes[0][0].rsplit("_", 1).pop(0)
cont=1
for transcricao in np_transcricoes:
    nome=transcricao[0].rsplit("_", 1).pop(0)
    if(nome!=nome_temp):
        transcricoes_agrupadas.append([nome_temp,transcricao_agrupada])
        transcricao_agrupada = transcricao[1]
    else:
        transcricao_agrupada=transcricao_agrupada+" "+transcricao[1]
    if(cont== len(np_transcricoes)):
        transcricoes_agrupadas.append([nome_temp,transcricao_agrupada])
        
    cont+=1    
    nome_temp=nome

np_transcricoes_agrupadas = np.asarray(transcricoes_agrupadas)
np.savetxt("output/"+arquivo_txt.replace(".txt", "")+"_agrupadas"+".csv", np_transcricoes_agrupadas, delimiter="_", fmt='%s')

