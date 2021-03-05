import scattertext as st
import spacy
from pprint import pprint
convention_df = st.SampleCorpora.ConventionData2012.get_data()
convention_df.iloc[0]
nlp = spacy.load("en_core_web_sm")
corpus = st.CorpusFromPandas(convention_df, category_col='party', text_col='text',nlp=nlp).build()
html = st.produce_scattertext_explorer(corpus,
          category='democrat',
          category_name='Democratic',
          not_category_name='Republican',
          width_in_pixels=1000,
          metadata=convention_df['speaker'])
open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))
