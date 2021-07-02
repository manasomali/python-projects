import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tpot.export_utils import set_param_recursive
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def class_to_num(classe):
    classes_to_num={'carga':0,
                    'comprovacao de disponibilidade':1,
                    'controle de geracao':2,
                    'controle de tensao':3,
                    'controle de transmissao':4,
                    'conversora':5,
                    'falha de supervisao':6,
                    'hidrologia':7,
                    'horario':8,
                    'sem informacao':9,
                    'sgi':10,
                    'teste de comunicacao':11}
    return classes_to_num[classe]

def num_to_class(num):
    num_to_classes={0:'carga',
                    1:'comprovacao de disponibilidade',
                    2:'controle de geracao',
                    3:'controle de tensao',
                    4:'controle de transmissao',
                    5:'conversora',
                    6:'falha de supervisao',
                    7:'hidrologia',
                    8:'horario',
                    9:'sem informacao',
                    10:'sgi',
                    11:'teste de comunicacao'}
    return num_to_classes[num]

new_file = csv.reader(open('input/transcricoes_comnum_comsubes.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in list(new_file):
    list_docs.append(row[4])
    list_labels.append(row[5])

X_train = []
X_test = []
y_train = []
y_test = []

list_labels_num = [class_to_num(a) for a in list_labels]
split=0.2
training_features, testing_features, training_target, testing_target = train_test_split(list_docs, list_labels_num, test_size=float(split), random_state = 42, shuffle=True, stratify=list_labels)

# Average CV score on the training set was: 0.8558333333333333
exported_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1,3),max_features=2000),
    SelectPercentile(score_func=f_classif, percentile=85),
    LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2", tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(confusion_matrix(testing_target, results))
print(classification_report(testing_target, results, digits=4))
