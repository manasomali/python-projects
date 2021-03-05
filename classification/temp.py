categories = ['tensao', 'geracao','comunicacao', 'conversora']

docs_train = ['duzento trinta kv dezenove kv duzento dezenove',
        'eevalr geracao',
        'bloqueado ribeira hora doze minuto bloqueando conversor rivera',
        'teste de comunicacao']

docs_test = ['tensao podemo considerar partir cinco kv quarenta oito deixar anotado',
            'central elevar quatro kv tensao barra usina conectada quinhento vinte cinco quatro luquinha',
            'gerar outro grupo deixar total usina seiscento mw maquina total geracao seiscento positivo']


from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
count_vec.fit(docs_train)

print('Vocabulario: '+ str(count_vec.vocabulary_)+'\n\n')

print('Names: '+str(count_vec.get_feature_names())+'\n\n')

count = count_vec.transform(docs_train)
print('Shape: '+str(count.shape)+'\n\n')

print('Count: \n'+str(count.toarray()))



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(docs_train)
print('Aprendendo a frequencia de todas as features: \n'+str(tfidf_vec.idf_)+'\n\n')

freq = tfidf_vec.transform(docs_train)
print('Transformando a matrix baseado no aprendizado da frequencia ou peso: \n'+str(freq.toarray())+'\n\n')
clf = MultinomialNB().fit(X, news_train.target)


# predição
from sklearn.naive_bayes import MultinomialNB

x_new_counts = count_vec.transform(docs_new)
x_new_tfidf = tfidf_vec.transform(x_new_counts)
predicted = clf.predicted(x_new_tfidf)

for predict in predicted:
    print(predict)