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
    
    def perform_gridsearchcv(self, searchparameters, cv, scoring, refit):
        gs_clf = GridSearchCV(self.pipeline, searchparameters, cv=cv, scoring=scoring, refit=refit, return_train_score=True, n_jobs=-1, verbose=5)
        gs_clf = gs_clf.fit(list(self.X_train), list(self.y_train))

        results = pd.DataFrame(gs_clf.cv_results_)
        results.to_excel(os.path.join(pathlib.Path().absolute(), 'docs\\' + str(self.titulo) + '.xlsx'), sheet_name = str(self.titulo))

        
        if self.verbose==True:
            print('Best Score: ' + str(gs_clf.best_score_))
            print('Best Params: ' + str(gs_clf.best_params_))
            
        return gs_clf    
    
    def plot_learning_curve(self, x, y, model, cv, train_sizes=np.linspace(.1, 1.0, 5)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(self.titulo)
        axes[0].set_xlabel("Training samples")
        axes[0].set_ylabel("Score")
    
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(model, x, y, cv=cv, n_jobs=-1,
                           train_sizes=train_sizes,
                           return_times=True, random_state = 42)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    
        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
    
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")
    
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

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
            print(cm.confusion_matrix)
            
        return {'model':self.titulo,
                 'confusion_matrix':confusion_matrix(self.y_test, self.predicted)}

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