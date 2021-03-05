from os import path
import glob
import re
import shutil
import sys
from natsort import natsorted

def IsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# pega diretorios e nomes dos audios
directory = path.dirname(path.realpath(__file__))
directoryinput = path.dirname(path.realpath(__file__)) + '\input\*.wav'
caminhos = (glob.glob(directoryinput))
caminhos = natsorted(caminhos)
nomes = []
for caminho in caminhos:
    nomes.append(caminho.replace(directoryinput.replace("*.wav",""),""))
    
nomes = natsorted(nomes)

# quebra nomes dos audios em uma lista de listas
nomes_em_partes = []
for nome in nomes:
    nome = nome.replace(".wav", "")
    nomes_em_partes.append(re.split("_|-",nome))

# pega dadas no fomato aaaammdd
datas = []
for nome_em_partes in nomes_em_partes:
    for parte_do_nome in nome_em_partes:
        if((IsInt(str(parte_do_nome)))and(len(parte_do_nome)==int(8))):
            datas.append(parte_do_nome)
            break

# verifica se ta tudo ok com as datas
if(len(datas)==len(nomes)):
    print("Datas extraidas com sucesso")
else:
    print("Erro na extração das dadas")
    sys.exit()
    
# pega horas no fomato aaaammdd
horas = []
for nome_em_partes in nomes_em_partes:
    for parte_do_nome in nome_em_partes:
        if((IsInt(str(parte_do_nome)))and(len(parte_do_nome)==int(6))):
            horas.append(parte_do_nome)
            break
            
# verifica se ta tudo ok com as horas
if(len(horas)==len(nomes)):
    print("Horas extraidas com sucesso")
else:
    print("Erro na extração das horas")
    sys.exit() 

# pega agente
agentes = []
for nome_em_partes in nomes_em_partes:
    agente=""
    for parte_do_nome in nome_em_partes:
        if((len(parte_do_nome)>=2)and(not IsInt(str(parte_do_nome)))and(str(parte_do_nome)!="Trans1")and(str(parte_do_nome)!="Trans2")and(str(parte_do_nome)!="L1")and(str(parte_do_nome)!="L2")):
           parte_do_nome=re.sub('[0-9]', '', parte_do_nome)
           agente=agente+parte_do_nome.upper()

    agentes.append(agente)

# verifica se ta tudo ok com os agentes
if(len(agentes)==len(nomes)):
    print("Agentes extraidos com sucesso")
else:
    print("Erro na extração dos agentes")
    sys.exit()

# copia os audios da pasta input para a pasta output (cont garante que não sobreescreva caso nome seja igual)
cont = 1
for (data, hora, agente, caminho, nome) in zip(datas, horas, agentes, caminhos, nomes):
    shutil.copy2(caminho, directory+"/output/"+str(cont)+"_"+data+"_"+hora+"_"+agente+".wav")
    cont+=1

# cria um txt com os dados dos audios
cont = 1
with open("output/dados.txt", "w") as txt_file:
    for (data, hora, agente, caminho, nome) in zip(datas, horas, agentes, caminhos, nomes):
        
        ano = data[0]+data[1]+data[2]+data[3]
        mes = data[4]+data[5]
        dia = data[6]+data[7]
        data=ano+"/"+mes+"/"+dia
        
        h = hora[0]+hora[1]
        m = hora[2]+hora[3]
        s = hora[4]+hora[5]
        hora=h+":"+m+":"+s
        
        txt_file.write(str(cont) + "\t" + data + "\t" + hora + "\t" + agente + "\n")
        cont+=1