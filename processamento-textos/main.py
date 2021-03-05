import csv
import re
import nltk
import string
import unicodedata
from tqdm import tqdm
from lexemas_raizes import substituicoes

def removeRepetidos(lista):
    l = []
    for i in lista:
        if i not in l:
            l.append(i)
    return l

def corrigePalavras(lista):
    text_new = []
    for word in lista:
        try:
            if substituicoes[word]:
                text_new.append(substituicoes[word])
        except:
            text_new.append(word)
    return text_new

def removePlural(lista):
    clean_text = []
    for word in lista:
        last_char = word[-1]
        if last_char == 's':
            word = word[:-1]
            
        clean_text.append(word)
    return clean_text

def removeInho(lista):
    clean_text = []
    for word in lista:
        if len(word)>=4:
            last_char = word[-4:]
            if last_char == 'inho':
                word = word[:-4]+'o'
            if last_char == 'inha':
                word = word[:-4]+'a'
                
        clean_text.append(word)
    return clean_text

def removeAcento(lista):
    clean_text = []
    for word in lista:
        nfkd = unicodedata.normalize('NFKD', word)
        palavras_sem_acento = u''.join([c for c in nfkd if not unicodedata.combining(c)])
        q = re.sub('[^a-zA-Z0-9 \\\]', ' ', palavras_sem_acento)
        clean_text.append(q.lower().strip())
    return clean_text

def removePontuacao(lista):
    clean_text = []
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    words = lista.split()
    for word in words:
        new_token = regex.sub(u'', word)
        if not new_token == u'':
            clean_text.append(new_token)
    return clean_text

def removePequenas(lista):
    clean_text = []
    for word in lista:
        if len(word)<=2:
            if word == 'kv' or word == 'mw' or word == 'sg':
                clean_text.append(word)
        
        else:
            clean_text.append(word)
                
    return clean_text

def getStopWords():
    # remove stop words
    #stop_words = nltk.corpus.stopwords.words('portuguese')
    stop_words = ['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'ate', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'eramos', 'essa', 'essas', 'esse', 'esses', 'esta', 'estamos', 'estao', 'estão', 'estas', 'estava', 'estavam', 'estavamos', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estiveramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivessemos', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora', 'foram', 'foramos', 'forem', 'formos', 'fosse', 'fossem', 'fossemos', 'fôssemos', 'fui', 'ha', 'há', 'haja', 'hajam', 'hajamos', 'hao', 'havemos', 'havia', 'hei', 'houve', 'houvemos', 'houver', 'houvera', 'houveram', 'houveramos', 'houverao', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houveriamos', 'houvermos', 'houvesse', 'houvessem', 'houvessemos', 'isso', 'isto', 'ja', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nao', 'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'sao', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'sera', 'serao', 'serei', 'seremos', 'seria', 'seriam', 'seriamos', 'seríamos', 'seu', 'seus', 'so', 'somos', 'sou', 'sua', 'suas', 'tambem', 'te', 'tem', 'temos', 'tenha', 'tenham', 'tenhamos', 'tenho', 'ter', 'tera', 'terao', 'terei', 'teremos', 'teria', 'teriam', 'teriamos', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tinhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tiveramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivessemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'voce', 'voces', 'vos']
    with open('stopwords.txt', 'r') as f:
        stop_words_personal = f.readlines()
    
    stop_words_personal = [x.strip() for x in stop_words_personal] 
    stop_words.extend(stop_words_personal)
    
    with open('nomes.txt', 'r') as f:
        stop_words_nomes = f.readlines()
    
    stop_words_nomes = [x.strip() for x in stop_words_nomes]
    stop_words.extend(stop_words_nomes)
    
    with open('agentes.txt', 'r') as f:
        stop_words_agentes = f.readlines()
    
    stop_words_agentes = [x.strip() for x in stop_words_agentes]
    stop_words.extend(stop_words_agentes)
    
    stop_words = removePlural(stop_words)
    stop_words = removeRepetidos(stop_words)
    stop_words = removeInho(stop_words)
    return stop_words
    
def processamentoTexto(text, stop_words):
    # remove pontuação
    text_sem_pontuacao = removePontuacao(text)
    # remove acento
    text_sem_acento = removeAcento(text_sem_pontuacao)
    # remove plural
    text_sem_plural = removePlural(text_sem_acento)
    # remove inho
    text_sem_inho = removeInho(text_sem_plural)
    # remove numeros
    text_sem_numero = [a for a in text_sem_inho if not a.isdigit()]
    # remove palavras pequenas
    text_sem_pequenas = removePequenas(text_sem_numero)
    # subistitui palavras transcritas erradas
    text_corrigido = corrigePalavras(text_sem_pequenas)
    # remove stop words
    tokens = [w for w in text_corrigido if w.lower().strip() not in stop_words]
    # teste se ta vazio
    text_sem_vazio = ["transcricaovazia"] if len(tokens)<3 else tokens
    # junta tokens
    frase = ' '.join(text_sem_vazio)
    return frase

new_file = csv.reader(open('input/transcricoes_wit.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
doc_words=[]
# prepara lista de stop words
stop_words = getStopWords()
for row in tqdm(list(new_file)):
    limpo = processamentoTexto(row[4], stop_words)
    doc_words.append([row[0], row[1], row[2], row[3], limpo])
    list_docs.append(limpo)

with open("output/transcricoes_tratadas.csv", "w") as txt_file:
    for line in tqdm(doc_words):
        txt_file.write(str(line[0])+"_"+str(line[1])+"_"+str(line[2])+"_"+str(line[3])+"_"+str(line[4])+"\n")

palavras = []
for line in tqdm(doc_words):
    words = line[4].split()
    cont=1
    for word in words:
        palavras.append([str(line[0])+"_"+str(line[1])+"_"+str(line[2])+"_"+str(line[3])+"_"+str(cont),word])
        cont+=1

with open("output/transcricoes_palavras.csv", "w") as txt_file:
    for word in tqdm(palavras):
        txt_file.write(str(word[0])+"_"+str(word[1])+"\n")
