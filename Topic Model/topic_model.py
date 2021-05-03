from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import spacy
import datetime
import numpy as np
import pandas as pd
import gensim
import pickle
from pprint import pprint



#vectorizer = CountVectorizer(analyzer='word',       
 #                               min_df=10,                        # minimum reqd occurences of a word 
  #                              stop_words='english',             # remove stop words
   #                            lowercase=True,                   # convert all words to lowercase
    #                            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
     #                           # max_features=50000,             # max number of uniq words
      #                          )


nlp = spacy.load('en', disable=['parser', 'ner'])

def best_LDA( data_vectorized, n_components, learning_decay ):

    
    
    #GridSearch the best LDA model
    # Define Search Param
    search_params = {'n_components': n_components, 'learning_decay': learning_decay}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

    return best_lda_model

def entrenar(data_vectorized, num_topics):
    
    lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                                        evaluate_every=-1, learning_decay=0.7,
                                        learning_method='online', learning_offset=10.0,
                                        max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                                        n_components=num_topics, n_jobs=-1, perp_tol=0.1,
                                        random_state=100, topic_word_prior=None,
                                        total_samples=1000000.0, verbose=0)
    lda_model.fit_transform(data_vectorized)

    #print(lda_model)  # Model attributes

    return lda_model

def save_model(lda_model, ruta="lda_model.pk"):

    pickle.dump(lda_model, open(ruta, 'wb'))
    print("Done")

def load_model(ruta="lda_model.pk"):

    lda_model = pickle.load(open(ruta, 'rb'))

    return lda_model

def topic_names(lda_model):
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    return topicnames

def document_names(doc_names):
    docnames = ["Doc" + str(i) for i in range(len(doc_names))]
    return docnames

## Funcion para crear Document - Topic Matrix. LA QUE USARIA PARA EL SISTEMA DE RECOMENDACION.
def document_topic_matrix(lda_model,data_vectorized,doc_names):
    # Create Document - Topic Matrix
    lda_output = lda_model.transform(data_vectorized)

    # column names
    #topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    topicnames = topic_names(lda_model)

    # index names
    #docnames = ["Doc" + str(i) for i in range(len(doc_names))]
    docnames = document_names(doc_names)
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    return df_document_topic


# Styling
#def color_green(val):
#    color = 'green' if val > .1 else 'black'
#    return 'color: {col}'.format(col=color)

#def make_bold(val):
#    weight = 700 if val > .1 else 400
#    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
#df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)

#asd = document_topic_matrix(data_vectorized)

#print(asd)
def weight_matrix_keywords(lda_model,vectorized,topicnames):

    
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorized.get_feature_names()

    
    df_topic_keywords.index = topicnames

    # View
    return df_topic_keywords

#asd = weight_matrix_keywords(lda_model)
#print(asd)

def Review_topics_distribution(lda_model,data_vectorized,doc_names):

    df_document_topic =  document_topic_matrix(lda_model,data_vectorized,doc_names)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    
    return df_topic_distribution

#asd = Review_topics_distribution(data_vectorized)

#pprint(asd)



def topic_keyword_matrix(lda_model,vectorizer, topicnames):

    
                                
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    return df_topic_keywords

#asd = topic_leyword_matrix(lda_model)
#pprint(asd)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def show_topics(lda_model,vectorizer, n_words):
    
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    
    return topic_keywords

def top_n_keywords_topic(lda_model,vectorizer, n_words):

    # Show top n keywords for each topic
    #

    topic_keywords = show_topics(lda_model,vectorizer, n_words)        

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    
    return df_topic_keywords


#asd = top_n_keywords_topic(vectorizer,best_lda_model, 15)
#print(asd)

#def predict_topic(vectorizer,best_lda_model, text, nlp = nlp):
    

    # Step 1: Clean with simple_preprocess
 #   mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
  #  mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
                                    
   # mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    #topic_probability_scores = best_lda_model.transform(mytext_4)

    #df_topic_keywords = top_n_keywords_topic(vectorizer,best_lda_model, 15)

    #topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    #return topic, topic_probability_scores


#mytext = ["Some text about Physics an mathematics"]


#topic, prob_scores = predict_topic(text = mytext)

#print(topic)

 