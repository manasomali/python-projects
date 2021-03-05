from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import csv
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 1)
    
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=1).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('Sum of Squared Error (SSE) by Cluster Center Plot')

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=1500, replace=True)
    
    pca = PCA(n_components=12).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=1500).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=150, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 1)
    ax.scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax.set_title('PCA Cluster Plot')
    
    f, ax = plt.subplots(1, 1)
    ax.scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax.set_title('TSNE Cluster Plot')
    

labels = ['controle de tensao',
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

choose=input("1 - K-means \n-> ")
if choose==str(1):
    # treinando modelo
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list_docs)
    km = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=1)
    km.fit(X)
    print("Achar melhor n de clusters")
    find_optimal_clusters(X, 15)
    clusters = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=1).fit_predict(X)
    plot_tsne_pca(X, clusters)



print()
print("Para 12 clusters")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(list_labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(list_labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(list_labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(list_labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(12):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

print(km.score(X))

