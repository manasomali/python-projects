from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

new_file = csv.reader(open('input/transcricoes_comnum_comsubes.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in tqdm(list(new_file)):
    list_docs.append(row[4])
    list_labels.append(row[5])

X_train = []
X_test = []
y_train = []
y_test = []

split=0.2
X_train, X_test, y_train, y_test = train_test_split(list_docs, list_labels, test_size=float(split), random_state = 42, shuffle=True, stratify=list_labels)
sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
meus_scores = {'accuracy' :make_scorer(accuracy_score),
               'recall'   :make_scorer(recall_score,average='micro'),
               'precision':make_scorer(precision_score,average='micro'),
               'f1'       :make_scorer(fbeta_score,average='micro',beta = 1)}
choose=input("-1 - Usar modelos já treinados\n0 - Todos \n1 - MultinomialNB \n2 - ComplementNB \n3 - LinearSVC\n4 - SGDClassifier\n5 - KNeighborsClassifier\n6 - MLPClassifier\n7 - RandomForestClassifier\n8 - DecisionTreeClassifier\n9 - AdaBoostClassifier\n-> ")
if (choose==str(1))or(choose==str(0)):
    # treinando modelo
    print("\nMultinomialNB")
    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB() )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
               'tfidf__use_idf': [True, False],
               'tfidf__norm': ['l1', 'l2'],
               'tfidf__smooth_idf': [True, False],
               'tfidf__sublinear_tf': [True, False],
               'clf__alpha': [0.001,0.01,0.1,1,0.5,7],
               'clf__fit_prior': [True, False],}
        
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #0.8367
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\MultinomialNB.xlsx',sheet_name= 'MultinomialNB')
    #{'clf__alpha': 0.01, 'clf__fit_prior': True, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': False, 'vect__binary': False, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    #evaluateModel(predicted, y_test, recall, precision)
    with open('models\MultinomialNB', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(2))or(choose==str(0)):
    # treinando modelo
    print("\nComplementNB")
    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                          ('clf', ComplementNB() )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
               'tfidf__use_idf': [True, False],
               'tfidf__norm': ['l1', 'l2'],
               'tfidf__smooth_idf': [True, False],
               'tfidf__sublinear_tf': [True, False],
               'clf__alpha': [0.001,0.01,0.1,1,0.5,7],
               'clf__fit_prior': [True],
               'clf__norm': [False],}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #0.855
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\ComplementNB.xlsx',sheet_name= 'ComplementNB')
    #{'clf__alpha': 0.1, 'clf__fit_prior': True, 'clf__norm': False, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': False, 'vect__binary': True, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\ComplementNB', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
        
if (choose==str(3))or(choose==str(0)):
    # treinando modelo
    print("\nLinearSVC")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LinearSVC()) ])
    
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1', 'l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__penalty': ['l1', 'l2'],
                'clf__loss': ['hinge','squared_hinge'],
                'clf__dual': [True, False],
                'clf__C': [0.1,0.5,1,5,7,10],
                'clf__multi_class': ['ovr','crammer_singer'],
                'clf__fit_intercept': [True, False],
                'clf__class_weight': ['balanced',None],
                'clf__max_iter': [1,5,10,50,1000],}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #0.8783
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\LinearSVC.xlsx',sheet_name= 'LinearSVC')
    #{'clf__C': 1, 'clf__class_weight': 'balanced', 'clf__dual': True, 'clf__fit_intercept': True, 'clf__loss': 'squared_hinge', 'clf__max_iter': 10, 'clf__multi_class': 'ovr', 'clf__penalty': 'l2', 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': True, 'vect__binary': True, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\LinearSVC', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(4))or(choose==str(0)):
    # treinando modelo
    print("\nSGDClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier()) ])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1', 'l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__loss': ['hinge','log','modified_huber','squared_hinge','perceptron','squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'clf__penalty': ['l1','l2','elasticnet'],
                'clf__alpha': [0.0001,0.001,0.01,0.1,1,0.5,7],
                'clf__fit_intercept': [True, False],
                'clf__max_iter': [10,100,1000],
                'clf__n_jobs': [-1],
                'clf__learning_rate': ['optimal'],
                'clf__class_weight': ['balanced',None],}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) # 0.8858
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\SGDClassifier.xlsx',sheet_name= 'SGDClassifier')
    #{'clf__alpha': 0.001, 'clf__class_weight': 'balanced', 'clf__early_stopping': True, 'clf__fit_intercept': True, 'clf__learning_rate': 'optimal', 'clf__loss': 'modified_huber', 'clf__max_iter': 10, 'clf__n_iter_no_change': 5, 'clf__n_jobs': -1, 'clf__penalty': 'elasticnet', 'clf__validation_fraction': 0.1, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': False, 'tfidf__use_idf': True, 'vect__binary': True, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\SGDClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(5))or(choose==str(0)):
    # treinando modelo
    print("\nKNeighborsClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', KNeighborsClassifier() )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1','l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__n_neighbors': [5,10,20,25,30],
                'clf__weights': ['distance'],
                'clf__algorithm': ['auto'],
                'clf__p': [2],
                'clf__n_jobs': [-1]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #0.8175
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\KNeighborsClassifier.xlsx',sheet_name= 'KNeighborsClassifier')
    #{'clf__algorithm': 'auto', 'clf__n_jobs': -1, 'clf__n_neighbors': 25, 'clf__p': 2, 'clf__weights': 'distance', 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': True, 'vect__binary': True, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\KNeighborsClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    

if (choose==str(6))or(choose==str(0)):
    # treinando modelo
    print("\nMLPClassifier")
    #The number of hidden neurons should be between the size of the input layer and the size of the output layer.
    # entre len(input) e len(output)
    #The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
    # ( (2/3) * len(input) ) + len(output)
    #The number of hidden neurons should be less than twice the size of the input layer
    # number of hidden neurons < 2*len(input)
    pipeline = Pipeline([('vect', CountVectorizer(max_features=1000)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MLPClassifier(verbose=True) )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [False],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [True],
                #'clf__alpha': [0.0001,0.001,0.01,0.1,0.25,1,0.5,7],
                'clf__alpha': [7],
                'clf__max_iter': [1000],
                'clf__solver': ['lbfgs'],
                'clf__activation': ['identity'],
                'clf__hidden_layer_sizes': [(3000,1000)]}
                #'clf__hidden_layer_sizes': [(1000,1000)]}
                #'clf__hidden_layer_sizes': [(1000,1000,1000),(2000,2000,2000),(2500,2500,2500)]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=2, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\MLPClassifier.xlsx',sheet_name= 'MLPClassifier')
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\MLPClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)

if (choose==str(7))or(choose==str(0)):
    # treinando modelo
    print("\nRandomForestClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RandomForestClassifier() )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [True, False],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1','l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__n_estimators': [100,200,500,1000],
                'clf__criterion': ['gini','entropy'],
                'clf__class_weight': ['balanced','balanced_subsample'],
                'clf__verbose': [4],
                'clf__n_jobs': [-1]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=4)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #0.8175
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\RandomForestClassifier.xlsx',sheet_name= 'RandomForestClassifier')
    #{'clf__algorithm': 'auto', 'clf__n_jobs': -1, 'clf__n_neighbors': 25, 'clf__p': 2, 'clf__weights': 'distance', 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': True, 'vect__binary': True, 'vect__ngram_range': (1, 2)}
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\RandomForestClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(8))or(choose==str(0)):
    # treinando modelo
    print("\nDecisionTreeClassifier")

    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', DecisionTreeClassifier() )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [False],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [True],
                'clf__criterion': ['gini','entropy'],
                'clf__splitter': ['best','random'],
                'clf__max_features': ['auto','sqrt','log2',None],
                'clf__class_weight': ['balanced']}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=100)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\DecisionTreeClassifier.xlsx',sheet_name= 'AdaBoostClassifier')
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\DecisionTreeClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)  

if (choose==str(9))or(choose==str(0)):
    # treinando modelo
    print("\nAdaBoostClassifier")

    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', AdaBoostClassifier(random_state=0) )])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__binary': [False],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [True],
                'clf__base_estimator': [DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2)],
                'clf__n_estimators': [600,800,1000],
                'clf__learning_rate': [1,1.5,2],
                'clf__algorithm': ['SAMME.R','SAMME'] }
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=sss, scoring = meus_scores, refit = 'f1', n_jobs=-1, verbose=100)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_) #
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)[['params',
                              'mean_test_recall',
                              'mean_test_precision',
                              'mean_test_f1']]
    results.to_excel( r'E:\python-projects\classification\docs\AdaBoostClassifier.xlsx',sheet_name= 'AdaBoostClassifier')
    # predicao
    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted))
    with open('models\AdaBoostClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)        
   

if (choose==str(-1)):
    print("\nMultinomialNB")
    with open('models\MultinomialNB', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
         
    print("\nComplementNB")
    with open('models\ComplementNB', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
         
    print("\nLinearSVC")
    with open('models\LinearSVC', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
         
    print("\nSGDClassifier")
    with open('models\SGDClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
         
    print("\nKNeighborsClassifier")
    with open('models\KNeighborsClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
    
    print("\nRandomForestClassifier")
    with open('models\RandomForestClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))
         
    
    print("\nMLPClassifier")
    with open('models\MLPClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print(model.best_score_)
         print(model.best_params_)
         print(confusion_matrix(list(y_test), predicted))
         print(classification_report(y_test, predicted))


#teste
# with open('MLPClassifier', 'rb') as training_model:
#     model = pickle.load(training_model)

# print(model.predict(['por favor abaixar vinte kv na barra']))

