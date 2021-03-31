from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
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

def find_optimal_clusters_kmeans(data, max_k):
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
    ax.set_title('Sum of Squared Error (SSE) by Cluster Center Plot KMeans')

def find_optimal_clusters_spectralclustering(data, max_k):
    iters = range(2, max_k+1, 1)
    sse = []
    for k in iters:
        sse.append(SpectralClustering(n_clusters=k, affinity='nearest_neighbors').fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('Sum of Squared Error (SSE) by Cluster Center Plot SpectralClustering')
    
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
    
def top_terms(word_list,top_n):
    word_counter = {}
    for word in word_list:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    
    popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
    return popular_words[:top_n]

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


new_file = csv.reader(open('input/dataset_1500_3.csv', 'r', encoding='utf-8'),delimiter='_')
list_docs=[]
list_labels=[]
for row in tqdm(list(new_file)):
    list_docs.append(row[0])
    list_labels.append(row[1])

choose=input("1 - K-means \n2 - SpectralClustering \n3 - MeanShift\n-> ")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_docs)
if choose==str(1):
    clusters = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=1)
    clusters.fit(X)
    print("Achar melhor n de clusters...")
    find_optimal_clusters_kmeans(X, 15)
    print("Plot tsne pca...")
    clusters_labels = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=1).fit_predict(X)
    plot_tsne_pca(X, clusters_labels)
    print()
    print("Para 12 clusters KMeans")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(list_labels, clusters.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(list_labels, clusters.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(list_labels, clusters.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(list_labels, clusters.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, clusters.labels_, sample_size=1000))
    print("Clusters Score: %0.3f"
          % clusters.score(X))
    print()
    print("Top terms per cluster:")
    order_centroids = clusters.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(12):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
    
if choose==str(2):
    clusters = SpectralClustering(n_clusters=12, affinity='nearest_neighbors')
    terms = vectorizer.get_feature_names()
    clusters.fit(X)
    print("Achar melhor n de clusters...")
    #find_optimal_clusters_spectralclustering(X, 15)
    print("Plot tsne pca...")
    clusters_labels = SpectralClustering(n_clusters=12, affinity='nearest_neighbors').fit_predict(X)
    plot_tsne_pca(X, clusters_labels)
    print()
    print("Para 12 clusters SpectralClustering")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(list_labels, clusters.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(list_labels, clusters.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(list_labels, clusters.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(list_labels, clusters.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, clusters.labels_, sample_size=1000))
    print()
    
    print("Terms per cluster:")
    for i in range(12):
        termos=[]
        print("Cluster %d:" % i, end='')
        T=X[clusters.labels_==i].indices
        for ind in T:
            termos.append(terms[ind])
        print(top_terms(termos,10))

if choose==str(3):
    # tem q reduzir para uma dimensao, ta errado
    X=X.toarray()
    bandwidth = estimate_bandwidth(X)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)