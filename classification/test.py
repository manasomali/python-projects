from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import csv
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    print("\n")
    t = PrettyTable(['Scores','Values'])
    t.add_row(['train_scores_mean', train_scores_mean])
    t.add_row(['train_scores_std', train_scores_std])
    t.add_row(['test_scores_mean', test_scores_mean])
    t.add_row(['test_scores_std', test_scores_std])
    print(t)

    # Plot learning curve
    fig0, axes0 = plt.subplots(dpi=300)
    axes0.grid()
    axes0.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes0.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes0.plot(train_sizes, train_scores_mean, 'o-', color="r",linestyle='dashed',
                 label="Training score")
    axes0.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes0.legend(loc="best")
    axes0.set_xlabel("Training examples")
    axes0.set_ylabel("Score")
    plt.savefig('plot1.png', dpi=300, format='png')
    
    # Plot n_samples vs fit_times
    fig1, axes1 = plt.subplots(dpi=300)
    axes1.grid()
    axes1.plot(train_sizes, fit_times_mean, 'o-')
    axes1.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes1.set_xlabel("Training examples")
    axes1.set_ylabel("fit_times")
    axes1.set_title("Scalability of the model")
    plt.savefig('plot2.png', dpi=300, format='png')
    
    # Plot fit_time vs score
    fig2, axes2 = plt.subplots(dpi=300)
    axes2.grid()
    axes2.plot(fit_times_mean, test_scores_mean, 'o-')
    axes2.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes2.set_xlabel("fit_times")
    axes2.set_ylabel("Score")
    axes2.set_title("Performance of the model")
    plt.savefig('plot3.png', dpi=300, format='png')
    
    return [train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std]
    


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
choose_dataset=input("1 - Aleatorio\n2 - Controlado\n3 - Plots Learning Curve\n-> ")
X_train = []
X_test = []
y_train = []
y_test = []
if (choose_dataset==str(1))or(choose_dataset==str(3)):
    # divisão em treio e teste aleatório
    split=input("Porcentagem para teste (0 a 1): ")
    #split=0.2
    X_train, X_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=float(split))
if choose_dataset==str(2):
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
    
choose=input("1 - MultinomialNB \n1.1 - ComplementNB\n2 - SVC \n2.1 - LinearSVC\n3 - SGDClassifier\n-> ")
tfidf_vec=TfidfVectorizer(ngram_range=(1, 2))
if choose==str(1):
    if choose_dataset==str(3):
        X = tfidf_vec.fit_transform(list_docs)
        y = list_labels
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        dados=plot_learning_curve(MultinomialNB(alpha=0.01), "Learning Curves MultinomialNB", X, y, cv=cv, n_jobs=4)
      
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', MultinomialNB(alpha=0.1) )])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(2.1):
    if choose_dataset==str(3):
        X = tfidf_vec.fit_transform(list_docs)
        y = list_labels
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        dados=plot_learning_curve(LinearSVC(penalty='l1', dual=False), "Learning Curves", X, y, cv=cv, n_jobs=4)
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', LinearSVC(penalty='l1', dual=False)) ])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(2):
    if choose_dataset==str(3):
        X = tfidf_vec.fit_transform(list_docs)
        y = list_labels
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        dados=plot_learning_curve(SVC(kernel='linear', tol=1e-5), "Learning Curves", X, y, cv=cv, n_jobs=4)
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', SVC(kernel='linear', tol=1e-5)) ])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(3):
    if choose_dataset==str(3):
        X = tfidf_vec.fit_transform(list_docs)
        y = list_labels
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        dados=plot_learning_curve(SGDClassifier(loss='hinge', penalty='l1', max_iter=50, tol=1e-5, shuffle=False), "Learning Curves", X, y, cv=cv, n_jobs=4)
    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec),
                          ('clf', SGDClassifier(loss='hinge', penalty='l1', max_iter=50, tol=1e-5, shuffle=False)) ])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))

if choose==str(1.1):
    if choose_dataset==str(3):
        X = tfidf_vec.fit_transform(list_docs)
        y = list_labels
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        dados=plot_learning_curve(ComplementNB(alpha=0.1), "Learning Curves", X, y, cv=cv, n_jobs=4)

    # treinando modelo
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', ComplementNB(alpha=0.1) )])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))
    
if choose==str(5):
    pipeline = Pipeline([('vect', tfidf_vec), 
                          ('clf', KNeighborsClassifier(n_neighbors=5) )])
    pipeline.fit(list(X_train), list(y_train))
    # predicao
    predicted = pipeline.predict(list(X_test))
    predicted_proba = pipeline.predict_proba(list(X_test))
    for i in range(len(predicted_proba)):
        for j in range(len(predicted_proba[i])):
            print(str(i) +","+ str(j), end=' -> ')
            print(predicted_proba[i][j], end='\n')
        
    
acertos = 0
erros = 0
for (predic,correto) in zip(predicted,y_test):
    if predic==correto:
        acertos+=1
        categories_acertos[predic]+=1
    else:
        erros+=1
        categories_erros[predic]+=1

if True:
    print("\n")
    t = PrettyTable(['Erros', 'Acertos', 'Mean Accuracy'])
    t.add_row([str(erros), str(acertos),"{:.3f}".format(pipeline.score(list(X_test), list(y_test)))])
    print(t)
    
    t = PrettyTable(['Categoria', 'Acertos', 'Erros', 'Percentual'])
    for cat in categories:
        t.add_row([cat, str(categories_acertos[cat]), str(categories_erros[cat]),"{:.3f}".format((categories_acertos[cat])/(categories_erros[cat]+categories_acertos[cat]))])
    
    print(t)


with open("output/resultado.csv", "w") as txt_file:
    for (doc,predic,correto) in zip(X_test,predicted,y_test):
        txt_file.write(str(doc) + "_" + str(predic) + "_" + str(correto) + "\n")

if True:    
    print("\n")        
    print("classification report:")
    print(metrics.classification_report(y_test, predicted,target_names=categories,digits=3))

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
plt.savefig('plot4.png', dpi=300, format='png')