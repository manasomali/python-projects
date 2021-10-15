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
from sklearn.model_selection import GridSearchCV
import csv
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from time import time
from tpot import TPOTClassifier
import os
import variables
from utils import reportEvaluation
from utils import compileModels
from utils import setupTraining

import winsound
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest, chi2

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV


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

choose=input("-1 - Usar modelos já treinados\n0 - Todos \n2 - ComplementNB \n3 - LinearSVC\n4 - SGDClassifier\n5 - KNeighborsClassifier\n6 - MLPClassifier\n7 - RandomForestClassifier\n8 - RandomForestClassifierOpt\n9 - TPOT\n-> ")

if (choose==str(1))or(choose==str(0)):
    print("Corpus")
    count_vectorizer = CountVectorizer(ngram_range=(1,1))
    doc_vec = count_vectorizer.fit_transform(list_docs)
    print('ngram_range=(1,1):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,2))
    doc_vec = count_vectorizer.fit_transform(list_docs)
    print('ngram_range=(1,2):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,3))
    doc_vec = count_vectorizer.fit_transform(list_docs)
    print('ngram_range=(1,3):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,4))
    doc_vec = count_vectorizer.fit_transform(list_docs)
    print('ngram_range=(1,4):'+str(doc_vec.shape))
    print("Train")
    count_vectorizer = CountVectorizer(ngram_range=(1,1))
    doc_vec = count_vectorizer.fit_transform(X_train)
    print('ngram_range=(1,1):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,2))
    doc_vec = count_vectorizer.fit_transform(X_train)
    print('ngram_range=(1,2):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,3))
    doc_vec = count_vectorizer.fit_transform(X_train)
    print('ngram_range=(1,3):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,4))
    doc_vec = count_vectorizer.fit_transform(X_train)
    print('ngram_range=(1,4):'+str(doc_vec.shape))
    print("Test")
    count_vectorizer = CountVectorizer(ngram_range=(1,1))
    doc_vec = count_vectorizer.fit_transform(X_test)
    print('ngram_range=(1,1):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,2))
    doc_vec = count_vectorizer.fit_transform(X_test)
    print('ngram_range=(1,2):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,3))
    doc_vec = count_vectorizer.fit_transform(X_test)
    print('ngram_range=(1,3):'+str(doc_vec.shape))
    count_vectorizer = CountVectorizer(ngram_range=(1,4))
    doc_vec = count_vectorizer.fit_transform(X_test)
    print('ngram_range=(1,4):'+str(doc_vec.shape))
    
if (choose==str(2))or(choose==str(0)):
    print("\nComplementNB")
    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                          ('clf', ComplementNB() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3),(1, 4)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
               'tfidf__use_idf': [False],
               'tfidf__norm': ['l2'],
               'tfidf__smooth_idf': [True],
               'tfidf__sublinear_tf': [True],
               'clf__alpha': [0.1,0.5,1],
               'clf__fit_prior': [True],
               'clf__norm': [False]}
    
    SetupComplementNB = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='ComplementNB')
    
    model = SetupComplementNB.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted')
    
    SetupComplementNB.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupComplementNB.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportComplementNB = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='ComplementNB')
    
    ReportComplementNB.generate_gridsearch_results('clf__alpha', left_xlim=0.1, right_xlim=0.1, is_log=False)
    ReportComplementNB.print_basic_metrics()
    ReportComplementNB.print_classification_report()
    #ReportComplementNB.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportComplementNB.plot_error_matrix()
    #ReportComplementNB.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportComplementNB.plot_multiclass_roc()
    ReportComplementNB.export_predictions()
        
if (choose==str(3))or(choose==str(0)):
    print("\nLinearSVC")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LinearSVC()) ])
    
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3),(1, 4)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [True],
                'tfidf__sublinear_tf': [True],
                'clf__penalty': ['l2'],
                'clf__loss': ['squared_hinge'],
                'clf__dual': [True],
                'clf__C': [0.01,0.1,1],
                'clf__multi_class': ['ovr'],
                'clf__fit_intercept': [True],
                'clf__random_state': [42],
                'clf__class_weight': ['balanced'],
                'clf__max_iter': [10]}
    
    SetupLinearSVC = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='LinearSVC')
    
    model = SetupLinearSVC.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted')
    
    SetupLinearSVC.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupLinearSVC.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportLinearSVC = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='LinearSVC')
         
    ReportLinearSVC.generate_gridsearch_results('clf__C', left_xlim=0.005, right_xlim=1, is_log=True)
    ReportLinearSVC.print_basic_metrics()
    ReportLinearSVC.print_classification_report()
    #ReportLinearSVC.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportLinearSVC.plot_error_matrix()
    #ReportLinearSVC.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportLinearSVC.plot_multiclass_roc()
    ReportLinearSVC.export_predictions()
   
if (choose==str(4))or(choose==str(0)):
    print("\nSGDClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier()) ])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3),(1, 4)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [True],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [False],
                'clf__loss': ['log'],
                'clf__penalty': ['l1'],
                'clf__alpha': [0.00001,0.0001,0.001],
                'clf__fit_intercept': [True],
                'clf__max_iter': [5],
                'clf__n_jobs': [-1],
                'clf__learning_rate': ['optimal'],
                'clf__class_weight': ['balanced']}
    
    SetupSGDClassifier = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='SGDClassifier')
    
    model = SetupSGDClassifier.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted')
    
    SetupSGDClassifier.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupSGDClassifier.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportSGDClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='SGDClassifier')
    
    ReportSGDClassifier.generate_gridsearch_results('clf__alpha', left_xlim=0.000005, right_xlim=0.001, is_log=True)
    a=ReportSGDClassifier.print_basic_metrics()
    ReportSGDClassifier.print_classification_report()
    #ReportSGDClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportSGDClassifier.plot_error_matrix()
    #ReportSGDClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportSGDClassifier.plot_multiclass_roc()
    ReportSGDClassifier.export_predictions()
   
if (choose==str(5))or(choose==str(0)):
    print("\nKNeighborsClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', KNeighborsClassifier() )])
    parameters = {'vect__ngram_range': [(1, 2),(1, 3),(1, 4)],
                  'vect__binary': [True],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [False],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [True],
                'tfidf__sublinear_tf': [True],
                'clf__n_neighbors': [5,10,15],
                'clf__weights': ['distance'],
                'clf__p': [2],
                'clf__n_jobs': [-1]}
    
    SetupKNeighborsClassifier = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='KNeighborsClassifier')
    
    model = SetupKNeighborsClassifier.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted')
    
    SetupKNeighborsClassifier.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupKNeighborsClassifier.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportKNeighborsClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='KNeighborsClassifier')
    
    ReportKNeighborsClassifier.generate_gridsearch_results('clf__n_neighbors', left_xlim=3, right_xlim=3, is_log=False)
    ReportKNeighborsClassifier.print_basic_metrics()
    ReportKNeighborsClassifier.print_classification_report()
    #ReportKNeighborsClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportKNeighborsClassifier.plot_error_matrix()
    #ReportKNeighborsClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportKNeighborsClassifier.plot_multiclass_roc()
    ReportKNeighborsClassifier.export_predictions()
   
if (choose==str(6))or(choose==str(0)):
    print("\nMLPClassifier")

    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MLPClassifier() )])
    parameters = {#'vect__ngram_range': [(1, 1)], # ok
                  #'vect__ngram_range': [(1, 2)], # ok
                  #'vect__ngram_range': [(1, 3)],  # ok
                  'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [True],
                  #'vect__max_features': [1000],   # ok
                  #'vect__max_features': [2000],  # ok
                  #'vect__max_features': [3000],  # ok
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
                #'clf__hidden_layer_sizes': [(1000,)]}
                #'clf__hidden_layer_sizes': [(1500,)]}
                #'clf__hidden_layer_sizes': [(2500,)]}
                'clf__hidden_layer_sizes': [(1500,),(2000,),(2500,)]}
    
    # 1 -> (1, 1) [1000] ok
    # 2 -> (1, 1) [2000] ok
    # 3 -> (1, 1) [3000] ok
    # 4 -> (1, 2) [1000] ok
    # 5 -> (1, 2) [2000] ok
    # 6 -> (1, 2) [3000] ok
    # 7 -> (1, 3) [1000] ok
    # 8 -> (1, 3) [2000] ok
    # 9 -> (1, 3) [3000] ok
    # 10 ->(1, 1),(1, 2),(1, 3) e [1000, 2000, 3000]
    
    SetupMLPClassifier = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='MLPClassifier')
    
    model = SetupMLPClassifier.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted',
                                      jobs=None)
    
    SetupMLPClassifier.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupMLPClassifier.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportMLPClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='MLPClassifier')
    
    ReportMLPClassifier.generate_gridsearch_results('clf__hidden_layer_sizes', left_xlim=200, right_xlim=200, is_log=False)
    ReportMLPClassifier.print_basic_metrics()
    ReportMLPClassifier.print_classification_report()
    #ReportMLPClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportMLPClassifier.plot_error_matrix()
    #ReportMLPClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportMLPClassifier.plot_multiclass_roc()
    ReportMLPClassifier.export_predictions()
    
if (choose==str(7))or(choose==str(0)):
    print("\nRandomForestClassifier")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RandomForestClassifier() )])
    parameters = {'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
                  'vect__binary': [False],
                  'vect__max_features': [1000, 2000, 3000],
                'tfidf__use_idf': [False],
                'tfidf__norm': ['l2'],
                'tfidf__smooth_idf': [False],
                'tfidf__sublinear_tf': [True],
                'clf__max_depth': [50],
                'clf__max_features': ['sqrt'],
                'clf__min_samples_split': [25],
                'clf__min_samples_leaf': [1],
                'clf__bootstrap': [True],
                'clf__oob_score': [True],
                'clf__n_estimators': [1000, 1500, 2000],
                'clf__criterion': ['gini'],
                'clf__class_weight': ['balanced_subsample'],
                'clf__verbose': [0],
                'clf__random_state': [42],
                'clf__n_jobs': [-1]}


    SetupRandomForestClassifier = setupTraining(pipeline=pipeline,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 titulo='RandomForestClassifier')
    
    model = SetupRandomForestClassifier.perform_gridsearchcv(searchparameters=parameters,
                                      cv=skf,
                                      scoring=['accuracy','f1_weighted'],
                                      refit='f1_weighted')

    SetupRandomForestClassifier.plot_learning_curve(x=list_docs, 
                                     y=list_labels,
                                     model=model,
                                     scoring='f1_weighted',
                                     cv=skf)
    SetupRandomForestClassifier.save_model(model=model)
    
    predicted = model.predict(list(X_test))
    
    ReportRandomForestClassifier = reportEvaluation(
             clf=model, 
             X_test=X_test, 
             y_test=y_test, 
             predicted=predicted, 
             n_classes=12, 
             titulo='RandomForestClassifier')
    
    ReportRandomForestClassifier.generate_gridsearch_results('clf__n_estimators', left_xlim=500, right_xlim=500, is_log=False)
    ReportRandomForestClassifier.print_basic_metrics()
    ReportRandomForestClassifier.print_classification_report()
    #ReportRandomForestClassifier.plot_error_matrix(labels=variables.vecclassesnamesen)
    ReportRandomForestClassifier.plot_error_matrix()
    #ReportRandomForestClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
    ReportRandomForestClassifier.plot_multiclass_roc()
    ReportRandomForestClassifier.export_predictions()

if (choose==str(8)):
    print("\nRandomForestClassifierOpt")
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RandomForestClassifier() )])
    # define search space 
    search = {
        'vect__ngram_range': [(1, 1),(1, 2),(1, 3)],
        'vect__binary': [False],
        'vect__max_features': [1000, 2000, 3000],
        'tfidf__use_idf': [False],
        'tfidf__norm': ['l2'],
        'tfidf__smooth_idf': [False],
        'tfidf__sublinear_tf': [True],
        'clf__max_depth': Integer(10, 50),
        'clf__max_features': Categorical(['sqrt','auto']),
        'clf__min_samples_split': Integer(1, 50),
        'clf__min_samples_leaf': Integer(1, 50),
        'clf__bootstrap': [True],
        'clf__oob_score': [True],
        'clf__n_estimators': Integer(500, 1500),
        'clf__criterion': ['gini'],
        'clf__class_weight': ['balanced_subsample'],
        'clf__verbose': [0],
        'clf__random_state': [42],
        'clf__n_jobs': [-1]
    }
    opt = BayesSearchCV(
        pipeline,
        [(search, 16)],
        cv=skf,
        scoring='f1_weighted',
        iid=True
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    print("best params: %s" % str(opt.best_params_))

if (choose==str(9)):
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
         ComplementNB.plot_error_matrix()
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
         LinearSVC.plot_error_matrix()
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
         SGDClassifier.plot_error_matrix()
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
         KNeighborsClassifier.plot_error_matrix()
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
         RandomForestClassifier.plot_error_matrix()
         RandomForestClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen)
         
         RandomForestClassifier.set_verbose_false()
         AllData.add_basic_metrics(RandomForestClassifier.print_basic_metrics())
         AllData.add_classification_report(RandomForestClassifier.print_classification_report())
         AllData.add_confusion_matrix(RandomForestClassifier.plot_error_matrix(labels=variables.vecclassesnamesen))
         AllData.add_multiclass_roc(RandomForestClassifier.plot_multiclass_roc(name_classes=variables.vecclassesnamesen))
         AllData.add_predictions(predicted)
         
    alldata=AllData.return_all_data()
    AllData.export_all_predictions()


duration = 1500
freq = 440
winsound.Beep(freq, duration)

#os.system("shutdown /s /t 1") 