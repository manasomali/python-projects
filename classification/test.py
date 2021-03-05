from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

categories = ['controle de tensao', 'controle de geracao', 'conversora', 'teste comunicacao']

docs_train = ['duzento trinta kv dezenove kv duzento dezenove',
            'elevar geracao',
            'bloqueado ribeira hora doze minuto bloqueando conversor rivera',
            'teste de comunicacao']

docs_test = ['tensao podemo considerar partir cinco kv quarenta oito deixar anotado',
            'central quatro kv tensao barra usina conectada quinhento vinte cinco quatro luquinha',
            'gerar outro grupo deixar total usina seiscento mw maquina total geracao seiscento positivo']


text_clf = Pipeline([('vect', TfidfVectorizer()), 
                      ('clf', MultinomialNB()) ])

# treinando modelo
text_clf.fit(list(docs_train), list(categories))
# predicao
predicted = text_clf.predict(list(docs_test))

for predi in predicted:
    print(predi)