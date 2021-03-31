from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import csv
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

categories = ['carga',
'comprovacao de disponibilidade',
'controle de geracao',
'controle de tensao',
'controle de transmissao',
'conversora',
'falha de supervisao',
'hidrologia',
'horario',
'sem informacao',
'sgi',
'teste de comunicacao']
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
categories_conut_train = {'controle de tensao':0,
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
categories_conut_test = {'controle de tensao':0,
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

new_file = csv.reader(open('input/dataset_1500_1.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in tqdm(list(new_file)):
    list_docs.append(row[0])
    list_labels.append(row[1])
print("\n")
print("Divisão Dataset Treino/Teste:")
choose=input("1 - Aleatorio \n2 - Controlado \n-> ")
X_train = []
X_test = []
y_train = []
y_test = []
if choose==str(1):
    # divisão em treio e teste aleatório
    #split=input("Porcentagem para teste (0 a 1): ")
    split=0.2
    X_train, X_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=float(split))
if choose==str(2):
    # divisão em treio e teste controlada
    split=input("Máximo de docs para treino e teste: ")
    for (doc, label) in zip(list_docs, list_labels):
        if categories_conut_train[label]<int(split):
            X_train.append(doc)
            y_train.append(label)
            categories_conut_train[label]+=1
        if ((categories_conut_test[label]<int(split))and(categories_conut_train[label]>=int(split))):
            X_test.append(doc)
            y_test.append(label)
            categories_conut_test[label]+=1
            
choose=input("1 - MultinomialNB \n2 - LinearSVC \n3 - SGDClassifier\n4 - ComplementNB\n-> ")
tfidf_vec=TfidfVectorizer(smooth_idf=True, min_df=0.0005, ngram_range=(1, 2))
if choose==str(1):
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', MultinomialNB(alpha=0.01) )])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(2):
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', loss='hinge', random_state=0, tol=1e-5))) ])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(3):
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', SGDClassifier(alpha=0.0001, loss='hinge', penalty='l2', random_state=0, max_iter=5, tol=None)) ])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(4):
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', ComplementNB(alpha=0.01) )])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))
    
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
t.add_row([str(erros), str(acertos),"{:.3f}".format(pipeline.score(list(X_test), list(y_test)))])
print(t)

t = PrettyTable(['Categoria', 'Acertos', 'Erros', 'Percentual'])
for cat in categories:
    t.add_row([cat, str(categories_acertos[cat]), str(categories_erros[cat]),"{:.3f}".format((categories_acertos[cat])/(categories_erros[cat]+categories_acertos[cat]))])

print(t)


with open("output/resultado.csv", "w") as txt_file:
    for (doc,predic,correto) in zip(X_test,predicted,y_test):
        txt_file.write(str(doc) + "_" + str(predic) + "_" + str(correto) + "\n")

if False:    
    print("\n")        
    print("classification report:")
    print(metrics.classification_report(y_test, predicted))

if True:
    feature_names = np.asarray(tfidf_vec.get_feature_names())
    print(get_top_n_words(feature_names,10))

if True:
    feature_names = np.asarray(tfidf_vec.get_feature_names())
    print("top 10 keywords per class:")
    for i, label in enumerate(categories):
        top10 = np.argsort(pipeline['clf'].coef_[i])[-10:]
        print("%s: %s" % (label, " ".join(feature_names[top10])))

import seaborn as sns
import pandas as pd
cm =confusion_matrix(y_test, predicted)  
index = categories
columns = categories
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True)


#dados_x =  [train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std]
dado_nb = np.array([[ 120,  390,  660,  930, 1200], [0.99583333, 0.9925641 , 0.99090909, 0.99043011, 0.98941667], [0.00853913, 0.00492548, 0.00310514, 0.00162718, 0.00149304], [0.715, 0.793, 0.82 , 0.831, 0.84 ], [0.03930649, 0.01206004, 0.02357023, 0.013     , 0.01807392]])
dado_svc = np.array([[ 120,  390,  660,  930, 1200], [0.90333333, 0.92974359, 0.93530303, 0.9388172 , 0.94083333], [0.0346009 , 0.01107027, 0.00873153, 0.00373878, 0.00355512], [0.65766667, 0.80266667, 0.829     , 0.84233333, 0.859     ], [0.04751725, 0.02602563, 0.01955335, 0.01782944, 0.01453349]])
dado_sgd = np.array([[ 120,  390,  660,  930, 1200], [0.99583333, 0.99410256, 0.9930303 , 0.99322581, 0.993     ], [0.00853913, 0.00444855, 0.00226767, 0.00096774, 0.001     ], [0.69833333, 0.80633333, 0.83466667, 0.85466667, 0.86533333], [0.03304038, 0.01929018, 0.02513077, 0.01431394, 0.0116619 ]])

fig0, axes0 = plt.subplots(dpi=300)
axes0.grid()

axes0.fill_between(dado_svc[0], dado_svc[1] - dado_svc[2],
                     dado_svc[1] + dado_svc[2], alpha=0.1,
                     color="tab:purple")
axes0.fill_between(dado_svc[0], dado_svc[3] - dado_svc[4],
                     dado_svc[3] + dado_svc[4], alpha=0.1,
                     color="b")
axes0.plot(dado_svc[0], dado_svc[1], 'v-', color="tab:purple",linestyle='dashed',
             label="Training score LinearSVC")
axes0.plot(dado_svc[0], dado_svc[3], 'v-', color="b",
             label="Cross-validation score LinearSVC")


axes0.fill_between(dado_nb[0], dado_nb[1] - dado_nb[2],
                     dado_nb[1] + dado_nb[2], alpha=0.1,
                     color="k")
axes0.fill_between(dado_nb[0], dado_nb[3] - dado_nb[4],
                     dado_nb[3] + dado_nb[4], alpha=0.1,
                     color="g")
axes0.plot(dado_nb[0], dado_nb[1], 'o-', color="k",linestyle='dashed',
             label="Training score ComplementNB")
axes0.plot(dado_nb[0], dado_nb[3], 'o-', color="g",
             label="Cross-validation score ComplementNB")


axes0.fill_between(dado_sgd[0], dado_sgd[1] - dado_sgd[2],
                     dado_sgd[1] + dado_sgd[2], alpha=0.1,
                     color="c")
axes0.fill_between(dado_sgd[0], dado_sgd[3] - dado_sgd[4],
                     dado_sgd[3] + dado_sgd[4], alpha=0.1,
                     color="r")
axes0.plot(dado_sgd[0], dado_sgd[1], 's-', color="c",linestyle='dashed',
             label="Training score SGDClassifier")
axes0.plot(dado_sgd[0], dado_sgd[3], 's-', color="r",
             label="Cross-validation score SGDClassifier")

axes0.legend(loc="best")
axes0.set_xlabel("Training Samples")
axes0.set_ylabel("Score")
plt.savefig('plot1.png', dpi=300, format='png')
