import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import csv

new_file = csv.reader(open('input/transcricoes_comnum_comsubes_train.csv', 'r', encoding='utf-8'),delimiter='_')
train_docs = np.array([])
train_labels = np.array([])
for row in list(new_file):
    train_docs = np.append(train_docs, np.array([row[0]]), axis=0)
    train_labels = np.append(train_labels, np.array([row[1]]), axis=0)
    
new_file = csv.reader(open('input/transcricoes_comnum_comsubes_test.csv', 'r', encoding='utf-8'),delimiter='_')
test_docs = np.array([])
test_labels = np.array([])
for row in list(new_file):
    test_docs = np.append(test_docs, np.array([row[0]]), axis=0)
    test_labels = np.append(test_docs, np.array([row[1]]), axis=0)


data_docs = np.append(train_docs, test_docs, axis=0)
vectorizer = TextVectorization(output_mode="tf-idf", ngrams=2)

vectorizer.adapt(data_docs)
vec_docs = vectorizer(data_docs)
print(data_docs[:10])
print(vec_docs[:10])
