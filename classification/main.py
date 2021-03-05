from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import csv
from tqdm import tqdm
from prettytable import PrettyTable

categories = ['controle de tensao',
            'controle de geracao',
            'conversora',
            'teste de comunicacao',
            'sgi',
            'controle de transmissao',
            'hidrologia',
            'horario',
            'carga',
            'sem informacao',
            'falha de supervisao',
            'comprovacao de disponibilidade']
categories_acertos = {'controle de tensao':0,
            'controle de geracao':0,
            'conversora':0,
            'teste de comunicacao':0,
            'sgi':0,
            'controle de transmissao':0,
            'hidrologia':0,
            'horario':0,
            'carga':0,
            'sem informacao':0,
            'falha de supervisao':0,
            'comprovacao de disponibilidade':0}
categories_erros = {'controle de tensao':0,
            'controle de geracao':0,
            'conversora':0,
            'teste de comunicacao':0,
            'sgi':0,
            'controle de transmissao':0,
            'hidrologia':0,
            'horario':0,
            'carga':0,
            'sem informacao':0,
            'falha de supervisao':0,
            'comprovacao de disponibilidade':0}


new_file = csv.reader(open('input/dataset_1500.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in tqdm(list(new_file)):
    list_docs.append(row[0])
    list_labels.append(row[1])

X_train, X_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=0.2)

choose=input("1 - MultinomialNB \n2 - LinearSVC \n-> ")
if choose==str(1):
    # treinando modelo
    text_clf = Pipeline([('vect', TfidfVectorizer()), 
                          ('clf', MultinomialNB(alpha=0.01) )])
    text_clf.fit(list(X_train), list(y_train))
    # predicao
    predicted = text_clf.predict(list(X_test))

if choose==str(2):
    # treinando modelo
    text_clf = Pipeline([('vect', TfidfVectorizer()), 
                          ('clf', LinearSVC(penalty='l1', loss='squared_hinge', dual=False, random_state=0, tol=1e-5)) ])
    text_clf.fit(list(X_train), list(y_train))
    # predicao
    predicted = text_clf.predict(list(X_test))
    
acertos = 0
erros = 0
for (predic,correto) in zip(predicted,y_test):
    if predic==correto:
        acertos+=1
        categories_acertos[predic]+=1
    else:
        erros+=1
        categories_erros[predic]+=1

print("\n")
t = PrettyTable(['Erros', 'Acertos', 'Percentual'])
t.add_row([str(erros), str(acertos),"{:.3f}".format(text_clf.score(list(X_test), list(y_test)))])
print(t)

t = PrettyTable(['Categoria', 'Acertos', 'Erros', 'Percentual'])
for cat in categories:
    t.add_row([cat, str(categories_acertos[cat]), str(categories_erros[cat]),"{:.3f}".format((categories_acertos[cat])/(categories_erros[cat]+categories_acertos[cat]))])

print(t)

with open("output/resultado.csv", "w") as txt_file:
    for (doc,predic,correto) in zip(X_test,predicted,y_test):
        txt_file.write(str(doc) + "_" + str(predic) + "_" + str(correto) + "\n")