# -*- coding: utf-8 -*-
import variables

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import os
import pathlib

class setupTraining:
    def __init__(self, pipeline, X_train, X_test, y_train, y_test, titulo, verbose=True):
        self.pipeline = pipeline
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.titulo = titulo
        self.verbose = verbose
    
    def perform_gridsearchcv(self, searchparameters, cv, scoring, refit, jobs=-1):
        gs_clf = GridSearchCV(self.pipeline, searchparameters, cv=cv, scoring=scoring, refit=refit, return_train_score=True, n_jobs=jobs, verbose=5)
        gs_clf = gs_clf.fit(list(self.X_train), list(self.y_train))

        results = pd.DataFrame(gs_clf.cv_results_)
        results.to_excel(os.path.join(pathlib.Path().absolute(), 'docs\\' + str(self.titulo) + '.xlsx'), sheet_name = str(self.titulo))

        
        if self.verbose==True:
            print('Best Score ('+str(refit)+'): ' + str(gs_clf.best_score_))
            print('Best Params: ' + str(gs_clf.best_params_))
            
        return gs_clf 
    
    def plot_learning_curve(self, x, y, model, cv, scoring, train_sizes=np.linspace(.1, 1.0, 5)):
        fig, ax = plt.subplots()

        ax.set_title('Learning Curve '+str(self.titulo))
        ax.set_xlabel("Training samples")
        ax.set_ylabel("F1 Weighted")
    
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(model, x, y, cv=cv, scoring=scoring, n_jobs=-1,
                           train_sizes=train_sizes,
                           return_times=True, random_state = 42)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
    
        # Plot learning curve
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        ax.legend(loc="best")
    
        
        plt.savefig('learningcurve\\' +self.titulo+ ".tiff", format="tiff", dpi=300, bbox_inches='tight')

        return 
    
    def save_model(self, model):
        with open('models\\'+str(self.titulo), 'wb') as picklefile:
            pickle.dump(model,picklefile)
    
class reportEvaluation:
    def __init__(self, clf, X_test, y_test, predicted, n_classes, titulo, verbose=True):
        self.clf = clf
        self.X_test = X_test
        self.y_test = y_test
        self.predicted = predicted
        self.n_classes = n_classes
        self.titulo = titulo
        self.verbose = verbose
        
    def set_verbose_false(self):
        self.verbose = False
        
    def plot_multiclass_roc(self, name_classes=variables.vecclassesnamespt):
        try:
            y_score = self.clf.predict_proba(self.X_test)
        except:
            y_score = self.clf.decision_function(self.X_test)
            
        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # calculate dummies once
        y_test_dummies = pd.get_dummies(self.y_test, drop_first=False).values
        for i in range(self.n_classes):
            fpr[i], tpr[i], thresholds  = roc_curve(y_test_dummies[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        
        y_test_roc = label_binarize(self.y_test, classes=variables.vecclassesnamespt)
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_test_roc).ravel(), np.array(y_score).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
        if self.verbose==True:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic ' + str(self.titulo))
            for i in range(self.n_classes):
                ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], name_classes[i]))
            
            plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
            
            plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
            
            ax.legend(loc="lower right")
            ax.grid(alpha=.4)
            sns.despine()
            plt.savefig('roc\\' +self.titulo+ ".tiff", format="tiff", dpi=300, bbox_inches='tight')
            plt.show()
            
        return {'model':self.titulo, 'fpr-micro':fpr["micro"], 'fpr-macro':fpr["macro"], 'tpr-micro':tpr["micro"], 'tpr-macro':tpr["macro"] , 'roc_auc-micro':roc_auc["micro"], 'roc_auc-macro':roc_auc["macro"]}

    def print_basic_metrics(self):
        if self.verbose==True:
            print('Accuracy Score: {:.4f}'.format(accuracy_score(list(self.y_test), self.predicted)))
            print('Precision Score Micro: {:.4f}'.format(precision_score(list(self.y_test), self.predicted, average='micro')))
            print('Precision Score Macro: {:.4f}'.format(precision_score(list(self.y_test), self.predicted, average='macro')))
            print('Precision Score Weighted: {:.4f}'.format(precision_score(list(self.y_test), self.predicted, average='weighted')))
            print('Recall Score Micro: {:.4f}'.format(recall_score(list(self.y_test), self.predicted, average='micro')))
            print('Recall Score Macro: {:.4f}'.format(recall_score(list(self.y_test), self.predicted, average='macro')))
            print('Recall Score Weighted: {:.4f}'.format(recall_score(list(self.y_test), self.predicted, average='weighted')))
            print('F1 Score Micro: {:.4f}'.format(f1_score(list(self.y_test), self.predicted, average='micro')))
            print('F1 Score Macro: {:.4f}'.format(f1_score(list(self.y_test), self.predicted, average='macro')))
            print('F1 Score Weighted: {:.4f}'.format(f1_score(list(self.y_test), self.predicted, average='weighted')))
            print('Matthews Corrcoef: {:.4f}'.format(matthews_corrcoef(list(self.y_test), self.predicted)))
            
        return {'model':self.titulo,
                'accuracy_score':accuracy_score(list(self.y_test), self.predicted), 
                'precision_score_micro':precision_score(list(self.y_test), self.predicted, average='micro'), 
                'precision_score_macro':precision_score(list(self.y_test), self.predicted, average='macro'), 
                'precision_score_weighted':precision_score(list(self.y_test), self.predicted, average='weighted'),
                'recall_score_micro':recall_score(list(self.y_test), self.predicted, average='micro'),
                'recall_score_macro':recall_score(list(self.y_test), self.predicted, average='macro'),
                'recall_score_weighted':recall_score(list(self.y_test), self.predicted, average='weighted'),
                'f1_score_micro':f1_score(list(self.y_test), self.predicted, average='micro'),
                'f1_score_macro':f1_score(list(self.y_test), self.predicted, average='macro'),
                'f1_score_weighted':f1_score(list(self.y_test), self.predicted, average='weighted'),
                'matthews_corrcoef':matthews_corrcoef(list(self.y_test), self.predicted)}

    
    def print_classification_report(self):
        if self.verbose==True:
            print(classification_report(self.y_test, self.predicted, digits=4, output_dict=False))

        return {'model':self.titulo,
                'classification_report':classification_report(self.y_test, self.predicted, digits=4, output_dict=True)}
    
    def plot_error_matrix(self, color=plt.cm.Blues, labels=variables.vecclassesnamespt):
        if self.verbose==True:
            cm = plot_confusion_matrix(self.clf, list(self.X_test),list(self.y_test),
                                     display_labels=labels,
                                     cmap=color,
                                     xticks_rotation='vertical')
            cm.ax_.set_title('Confusion Matrix ' + str(self.titulo))
            plt.savefig('confusionmatrix\\' +self.titulo+ ".tiff", format="tiff", dpi=300, bbox_inches='tight')

            print(cm.confusion_matrix)
            
        return {'model':self.titulo,
                 'confusion_matrix':confusion_matrix(self.y_test, self.predicted)}
    
    def generate_gridsearch_results(self, param_x, left_xlim, right_xlim,  top_ylim=0.005, bottom_ylim=0.005, is_log=False):
        #titulo='RandomForestClassifier'
        #param_x='clf__n_estimators'
        #left_xlim=500
        #right_xlim=500
        #top_ylim=0.005
        #bottom_ylim=0.005
        #is_log=False
        dados = pd.read_excel(os.path.join(pathlib.Path().absolute(), 'docs\\' + str(self.titulo) + '.xlsx'))
        
        dados["param_vect__max_features"].replace({1000:'tab:blue',2000:'tab:red',3000:'tab:green'}, inplace=True)
        dados["param_vect__ngram_range"].replace({'(1, 1)':100,'(1, 2)':400,'(1, 3)':1600,'(1, 4)':3200,'(1, 5)':6400}, inplace=True)
        
        if param_x=="clf__hidden_layer_sizes":
            dados["param_clf__hidden_layer_sizes"].replace({'(2500,)':2500,'(2000,)':2000,'(1500,)':1500}, inplace=True)

        
        xis = list(dados['param_'+str(param_x)])
        
        test_scores_mean = list(dados['mean_test_f1_weighted'])
        
        sizes = list(dados['param_vect__ngram_range'])
        
        colors = list(dados['param_vect__max_features'])
        
        ymax = max(test_scores_mean)
        xpos = test_scores_mean.index(ymax)
        xmax = xis[xpos]
        
        plt.figure()
        plt.title(self.titulo, fontsize=12)
        plt.xlabel(param_x, fontsize=12)
        plt.ylabel('Mean F1 Weighted', fontsize=12)
        
        for x, test_score_mean, size, col in zip(xis, test_scores_mean, sizes, colors):
            plt.scatter(x, test_score_mean, s=size, c=col, marker="_", linewidths=2)
        
        plt.annotate("{0:.4f}".format(ymax), xy = (xmax, ymax+0.001), ha='center')
        
        if(is_log):
            plt.gca().set_xscale('log')
            
        plt.grid(True, alpha=0.5)
        plt.xticks(list(set(xis)))
        plt.xlim(min(xis)-left_xlim,max(xis)+right_xlim)
        plt.ylim(min(test_scores_mean)-bottom_ylim,max(test_scores_mean)+top_ylim)
        plt.savefig('gridsearch\\' +self.titulo+ ".tiff", format="tiff", dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_predictions(self):
        predictions = pd.DataFrame(
            {'Amostra': self.X_test,
             'Verdadeira': self.y_test,
             'Predição': list(self.predicted)
            })
        predictions.to_excel(os.path.join(pathlib.Path().absolute(), "predictions\\"+str(self.titulo)+".xlsx"),sheet_name=self.titulo)

class compileModels:
    def __init__(self, X_test, y_test):
        self.basic_metrics = []
        self.classification_report = []
        self.confusion_matrix = []
        self.multiclass_roc = []
        
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.allpredictions = np.zeros(shape=(len(self.X_test),0))
        np_X_test=np.asarray(X_test).reshape(len(self.X_test),1)
        self.allpredictions = np.append(self.allpredictions, np_X_test, axis=1)
        np_y_test=np.asarray(self.y_test).reshape(len(self.y_test),1)
        self.allpredictions = np.append(self.allpredictions, np_y_test, axis=1)

        
    def add_basic_metrics(self, newdic):
        self.basic_metrics.append(newdic)
        
    def add_classification_report(self, newdic):
        self.classification_report.append(newdic)
        
    def add_confusion_matrix(self, newdic):
        self.confusion_matrix.append(newdic)
        
    def add_multiclass_roc(self, newdic):
        self.multiclass_roc.append(newdic)
        
    def add_predictions(self, predicted):
        self.allpredictions = np.append(self.allpredictions, np.asarray(predicted).reshape(len(predicted),1), axis=1)
        
    def return_all_data(self):
        return{'basic_metrics':self.basic_metrics,
              'classification_report':self.classification_report,
              'confusion_matrix':self.confusion_matrix,
              'multiclass_roc':self.multiclass_roc}
    
    def export_all_predictions(self):
        results = pd.DataFrame(self.allpredictions,columns=['Amostra','Verdadeira']+variables.vecmodelsnames)
        results.to_excel(os.path.join(pathlib.Path().absolute(), 'docs\All_Predictions.xlsx'),sheet_name='Predictions')