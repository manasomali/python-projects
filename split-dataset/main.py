import numpy as np
from sklearn.model_selection import train_test_split
import csv
import os 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

    
filename = 'transcricoes_comnum_comsubes'
new_file = csv.reader(open('input/'+str(filename)+'.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in list(new_file):
    list_docs.append(row[4])
    list_labels.append(row[5])

x_train, x_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=0.2, random_state = 0, shuffle=True, stratify=list_labels)

createFolder('output/'+str(filename))
createFolder('output/'+str(filename)+'/train')
createFolder('output/'+str(filename)+'/test')
categories = ['carga',
'comprovacao_de_disponibilidade',
'controle_de_geracao',
'controle_de_tensao',
'controle_de_transmissao',
'conversora',
'falha_de_supervisao',
'hidrologia',
'horario',
'sem_informacao',
'sgi',
'teste_de_comunicacao']
for cat in categories:
    createFolder('output/'+str(filename)+'/train/'+str(cat))
    createFolder('output/'+str(filename)+'/test/'+str(cat))

train=[]
cont=0
for x, y in zip(x_train, y_train):
    text = str(x)+"_"+str(y)
    train.append(text)
    cat=y.replace(' ','_')
    np.savetxt('output/'+str(filename)+'/train/'+str(cat)+'/'+str(cont)+'.txt', [x], fmt='%s')
    cont+=1
 
test=[]
for x, y in zip(x_test, y_test):
    text = str(x)+"_"+str(y)
    test.append(text)
    cat=y.replace(' ','_')
    np.savetxt('output/'+str(filename)+'/test/'+str(cat)+'/'+str(cont)+'.txt', [x], fmt='%s')
    cont+=1

np.savetxt('output/'+str(filename)+'_train.csv', train, delimiter="_", fmt='%s')
np.savetxt('output/'+str(filename)+'_test.csv', test, delimiter="_", fmt='%s')



