import numpy as np
from sklearn.model_selection import train_test_split
import csv
import os 
import matplotlib.pyplot as plt


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


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


filename = 'transcricoes_comnum_comsubes'
new_file = csv.reader(open('input/'+str(filename)+'.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
cont_classes_geral = np.zeros(len(categories))

for row in list(new_file):
    list_docs.append(row[4])
    list_labels.append(row[5])
    if row[5] == categories[0].replace('_',' '):
        cont_classes_geral[0] += 1
    if row[5] == categories[1].replace('_',' '):
        cont_classes_geral[1] += 1
    if row[5] == categories[2].replace('_',' '):
        cont_classes_geral[2] += 1
    if row[5] == categories[3].replace('_',' '):
        cont_classes_geral[3] += 1
    if row[5] == categories[4].replace('_',' '):
        cont_classes_geral[4] += 1
    if row[5] == categories[5].replace('_',' '):
        cont_classes_geral[5] += 1
    if row[5] == categories[6].replace('_',' '):
        cont_classes_geral[6] += 1
    if row[5] == categories[7].replace('_',' '):
        cont_classes_geral[7] += 1
    if row[5] == categories[8].replace('_',' '):
        cont_classes_geral[8] += 1
    if row[5] == categories[9].replace('_',' '):
        cont_classes_geral[9] += 1
    if row[5] == categories[10].replace('_',' '):
        cont_classes_geral[10] += 1
    if row[5] == categories[11].replace('_',' '):
        cont_classes_geral[11] += 1
        
x_train, x_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=0.2, random_state = 0, shuffle=True, stratify=list_labels)

createFolder('output/'+str(filename))
createFolder('output/'+str(filename)+'/train')
createFolder('output/'+str(filename)+'/test')


for cat in categories:
    createFolder('output/'+str(filename)+'/train/'+str(cat))
    createFolder('output/'+str(filename)+'/test/'+str(cat))

train=[]
cont=0
cont_classes_train = np.zeros(len(categories))
for x, y in zip(x_train, y_train):
    text = str(x)+"_"+str(y)
    train.append(text)
    cat=y.replace(' ','_')
    np.savetxt('output/'+str(filename)+'/train/'+str(cat)+'/'+str(cont)+'.txt', [x], newline='', fmt='%s')
    if y == categories[0].replace('_',' '):
        cont_classes_train[0] += 1
    if y == categories[1].replace('_',' '):
        cont_classes_train[1] += 1
    if y == categories[2].replace('_',' '):
        cont_classes_train[2] += 1
    if y == categories[3].replace('_',' '):
        cont_classes_train[3] += 1
    if y == categories[4].replace('_',' '):
        cont_classes_train[4] += 1
    if y == categories[5].replace('_',' '):
        cont_classes_train[5] += 1
    if y == categories[6].replace('_',' '):
        cont_classes_train[6] += 1
    if y == categories[7].replace('_',' '):
        cont_classes_train[7] += 1
    if y == categories[8].replace('_',' '):
        cont_classes_train[8] += 1
    if y == categories[9].replace('_',' '):
        cont_classes_train[9] += 1
    if y == categories[10].replace('_',' '):
        cont_classes_train[10] += 1
    if y == categories[11].replace('_',' '):
        cont_classes_train[11] += 1
        
    cont+=1
 
test=[]
cont_classes_test = np.zeros(len(categories))
for x, y in zip(x_test, y_test):
    text = str(x)+"_"+str(y)
    test.append(text)
    cat=y.replace(' ','_')
    np.savetxt('output/'+str(filename)+'/test/'+str(cat)+'/'+str(cont)+'.txt', [x], newline='', fmt='%s')
    if y == categories[0].replace('_',' '):
        cont_classes_test[0] += 1
    if y == categories[1].replace('_',' '):
        cont_classes_test[1] += 1
    if y == categories[2].replace('_',' '):
        cont_classes_test[2] += 1
    if y == categories[3].replace('_',' '):
        cont_classes_test[3] += 1
    if y == categories[4].replace('_',' '):
        cont_classes_test[4] += 1
    if y == categories[5].replace('_',' '):
        cont_classes_test[5] += 1
    if y == categories[6].replace('_',' '):
        cont_classes_test[6] += 1
    if y == categories[7].replace('_',' '):
        cont_classes_test[7] += 1
    if y == categories[8].replace('_',' '):
        cont_classes_test[8] += 1
    if y == categories[9].replace('_',' '):
        cont_classes_test[9] += 1
    if y == categories[10].replace('_',' '):
        cont_classes_test[10] += 1
    if y == categories[11].replace('_',' '):
        cont_classes_test[11] += 1
        
    cont+=1

np.savetxt('output/'+str(filename)+'_train.csv', train, delimiter="_", fmt='%s')
np.savetxt('output/'+str(filename)+'_test.csv', test, delimiter="_", fmt='%s')


plt.figure()
obj = categories
y = np.arange(len(obj))
x = cont_classes_geral
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.title("Quantidade de amostras por classe geral")
plt.xlabel("Classe")
plt.ylabel("Quantidade")

plt.figure()
obj = categories
y = np.arange(len(obj))
x = cont_classes_train
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.title("Quantidade de amostras por classe treino")
plt.xlabel("Classe")
plt.ylabel("Quantidade")

plt.figure()
obj = categories
y = np.arange(len(obj))
x = cont_classes_test
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.title("Quantidade de amostras por classe teste")
plt.xlabel("Classe")
plt.ylabel("Quantidade")

print(cont_classes_geral)
print(cont_classes_train)
print(cont_classes_test)