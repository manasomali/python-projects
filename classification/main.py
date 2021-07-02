from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from time import time
from tpot import TPOTClassifier

import variables
from utils import reportEvaluation
from utils import compileModels
from utils import setupTraining



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
skf = StratifiedKFold(n_splits = 5, shuffle = False)

choose=input("-1 - Usar modelos já treinados\n0 - Todos \n2 - ComplementNB \n3 - LinearSVC\n4 - SGDClassifier\n5 - KNeighborsClassifier\n6 - MLPClassifier\n7 - RandomForestClassifier\n8 - TPOT\n-> ")

if (choose==str(2))or(choose==str(0)):
    print("\nComplementNB")
    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                          ('clf', ComplementNB() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True, False],
                  'vect__max_features': [1000, 2000, 3000],
               'tfidf__use_idf': [True, False],
               'tfidf__norm': ['l1', 'l2'],
               'tfidf__smooth_idf': [True, False],
               'tfidf__sublinear_tf': [True, False],
               'clf__alpha': [0.1,0.5,1],
               'clf__fit_prior': [True],
               'clf__norm': [False]}
    
    ComplementNB = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='ComplementNB')
    
    model = ComplementNB.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_macro'],
                                      refit='f1_macro')
    
    ComplementNB.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model, cv=skf)
    ComplementNB.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ComplementNB = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='ComplementNB')
         
    ComplementNB.print_basic_metrics()
    ComplementNB.print_classification_report()
    ComplementNB.plot_error_matrix(labels=variables.vecclassesnamesen)
    ComplementNB.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
   
        
if (choose==str(3))or(choose==str(0)):
    print("\nLinearSVC")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LinearSVC()) ])
    
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True, False],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l1', 'l2'],
                'tfidf__smooth_idf': [True],
                'tfidf__sublinear_tf': [True],
                'clf__penalty': ['l2'],
                'clf__loss': ['squared_hinge'],
                'clf__dual': [True, False],
                'clf__C': [0.001,0.01,0.1,1],
                'clf__multi_class': ['ovr'],
                'clf__fit_intercept': [True, False],
                'clf__class_weight': ['balanced'],
                'clf__max_iter': [10,500,1000]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\LinearSVC.xlsx',sheet_name= 'LinearSVC')

    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    with open('models\LinearSVC', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(4))or(choose==str(0)):
    print("\nSGDClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier()) ])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [False],
                'clf__loss': ['log'],
                'clf__penalty': ['l1','l2'],
                'clf__alpha': [0.00001,0.0001,0.001],
                'clf__fit_intercept': [True],
                'clf__max_iter': [5],
                'clf__n_jobs': [-1],
                'clf__learning_rate': ['optimal'],
                'clf__class_weight': ['balanced']}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\SGDClassifier.xlsx',sheet_name= 'SGDClassifier')

    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    with open('models\SGDClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    
if (choose==str(5))or(choose==str(0)):
    print("\nKNeighborsClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', KNeighborsClassifier() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True, False],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1','l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__n_neighbors': [5,10,20,30,40],
                'clf__weights': ['uniform','distance'],
                'clf__p': [1,2],
                'clf__n_jobs': [-1]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\KNeighborsClassifier.xlsx',sheet_name= 'KNeighborsClassifier')

    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    with open('models\KNeighborsClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
    

if (choose==str(6))or(choose==str(0)):
    print("\nMLPClassifier")

    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MLPClassifier() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l1'],
                'tfidf__smooth_idf': [True],
                'tfidf__sublinear_tf': [False],
                'clf__alpha': [0.1],
                'clf__early_stopping':[True],
                'clf__verbose':[True],
                'clf__max_iter': [1000],
                'clf__solver': ['lbfgs'],
                'clf__activation': ['identity'],
                'clf__hidden_layer_sizes': [(1000,),(1500,),(2000,)]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5)
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\MLPClassifier.xlsx',sheet_name= 'MLPClassifier')

    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    with open('models\MLPClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)

if (choose==str(7))or(choose==str(0)):
    print("\nRandomForestClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RandomForestClassifier() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True, False],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1','l2'],
                'tfidf__smooth_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'clf__n_estimators': [1000,2000,3000,4000],
                'clf__criterion': ['gini','entropy'],
                'clf__class_weight': ['balanced','balanced_subsample'],
                'clf__verbose': [4],
                'clf__n_jobs': [-1]}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5, return_train_score=True)
    
    gs_clf = gs_clf.fit(list(X_train), list(y_train))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\RandomForestClassifier.xlsx',sheet_name= 'RandomForestClassifier')

    predicted = gs_clf.predict(list(X_test))
    print(confusion_matrix(list(y_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    with open('models\RandomForestClassifier', 'wb') as picklefile:
        pickle.dump(gs_clf,picklefile)
        
if (choose==str(8)):
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, config_dict='TPOT sparse') )])
    
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True, False],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True, False],
                'tfidf__norm': ['l1','l2']}
    
    gs_clf = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=5, return_train_score=True)
    
    y_train_num=[variables.dicclasstonum[a] for a in y_train]
    y_test_num=[variables.dicclasstonum[a] for a in y_test]
    gs_clf = gs_clf.fit(np.array(X_train), np.array(y_train_num))
    
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    results = pd.DataFrame(gs_clf.cv_results_)
    results.to_excel( r'E:\python-projects\classification\docs\TPOTClassifier.xlsx',sheet_name= 'TPOTClassifier')
    
    predicted = gs_clf.predict(np.array(X_test))
    print(confusion_matrix(np.array(X_test), predicted))
    print(classification_report(y_test, predicted, digits=4))
    
        
if (choose==str(-1)):
    
    AllData = compileModels(X_test=X_test, y_test=y_test)
    
    print("\nComplementNB")
    with open('models\ComplementNB', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         ComplementNB = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='ComplementNB')
         
         ComplementNB.print_basic_metrics()
         ComplementNB.print_classification_report()
         ComplementNB.plot_error_matrix(labels=variables.vecclassesnamesen)
         ComplementNB.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         ComplementNB.set_verbose_false()
         AllData.add_basic_metrics(ComplementNB.print_basic_metrics())
         AllData.add_classification_report(ComplementNB.print_classification_report())
         AllData.add_confusion_matrix(ComplementNB.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(ComplementNB.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
         
    print("\nLinearSVC")
    with open('models\LinearSVC', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         LinearSVC = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='LinearSVC')
         
         LinearSVC.print_basic_metrics()
         LinearSVC.print_classification_report()
         LinearSVC.plot_error_matrix(labels=variables.vecclassesnamesen)
         LinearSVC.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         LinearSVC.set_verbose_false()
         AllData.add_basic_metrics(LinearSVC.print_basic_metrics())
         AllData.add_classification_report(LinearSVC.print_classification_report())
         AllData.add_confusion_matrix(LinearSVC.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(LinearSVC.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
         
    print("\nSGDClassifier")
    with open('models\SGDClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         SGDClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='SGDClassifier')
         
         SGDClassifier.print_basic_metrics()
         SGDClassifier.print_classification_report()
         SGDClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
         SGDClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         SGDClassifier.set_verbose_false()
         AllData.add_basic_metrics(SGDClassifier.print_basic_metrics())
         AllData.add_classification_report(SGDClassifier.print_classification_report())
         AllData.add_confusion_matrix(SGDClassifier.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(SGDClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
         
    print("\nKNeighborsClassifier")
    with open('models\KNeighborsClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         KNeighborsClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='KNeighborsClassifier')
         
         KNeighborsClassifier.print_basic_metrics()
         KNeighborsClassifier.print_classification_report()
         KNeighborsClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
         KNeighborsClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         KNeighborsClassifier.set_verbose_false()
         AllData.add_basic_metrics(KNeighborsClassifier.print_basic_metrics())
         AllData.add_classification_report(KNeighborsClassifier.print_classification_report())
         AllData.add_confusion_matrix(KNeighborsClassifier.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(KNeighborsClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
    
    print("\nMLPClassifier")
    with open('models\MLPClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         MLPClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='MLPClassifier')
         
         MLPClassifier.print_basic_metrics()
         MLPClassifier.print_classification_report()
         MLPClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
         MLPClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         MLPClassifier.set_verbose_false()
         AllData.add_basic_metrics(MLPClassifier.print_basic_metrics())
         AllData.add_classification_report(MLPClassifier.print_classification_report())
         AllData.add_confusion_matrix(MLPClassifier.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(MLPClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)

    print("\nRandomForestClassifier")
    with open('models\RandomForestClassifier', 'rb') as training_model:
         model = pickle.load(training_model)
         print('Model Best Score: {:.4f} \n'.format(model.best_score_))
         print('Model Best Params: {} \n'.format(model.best_params_))
         t0 = time()
         predicted = model.predict(list(X_test))
         test_time = time() - t0
         print('Test Time (s): {:.4f} \n'.format(test_time))
         
         RandomForestClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='RandomForestClassifier')
         
         RandomForestClassifier.print_basic_metrics()
         RandomForestClassifier.print_classification_report()
         RandomForestClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
         RandomForestClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         RandomForestClassifier.set_verbose_false()
         AllData.add_basic_metrics(RandomForestClassifier.print_basic_metrics())
         AllData.add_classification_report(RandomForestClassifier.print_classification_report())
         AllData.add_confusion_matrix(RandomForestClassifier.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(RandomForestClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
         
    alldata=AllData.return_all_data()
    AllData.export_all_predictions()
