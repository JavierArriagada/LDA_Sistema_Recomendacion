from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import re, nltk, spacy, gensim


### Modulo para aplicar tecnicas de limpieza de datos. Lematizar los datos

## posible mejora en proceso de limpiar palabras. Relacionado con la naturaleza del corpus


#vectorizer = CountVectorizer(analyzer='word',       
 #                               min_df=10,                        # minimum reqd occurences of a word 
  #                              stop_words='english',             # remove stop words
   #                            lowercase=True,                   # convert all words to lowercase
    #                            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
     #                           # max_features=50000,             # max number of uniq words
 #                               )
                                
nlp = spacy.load('en', disable=['parser', 'ner'])

nlp.max_length = 2000000

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



#print(data_words[:1])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


## Funcion que limpia el corpus de los documentos y nos entrega solo las palabras mas significativas para los topicos

def lematizar(data):

    #data = data
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    
    #return data
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    #nlp = spacy.load('en', disable=['parser', 'ner'])

    data_words = list(sent_to_words(data))
    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    
    
    ###Create the Document-Word matrix

    #Cada documento del corpus  es representado como un vector de palabras
    #Cada fila de la matriz representa una vector de cada documento del corpus
    #Cada columna representa las palabras del diccionario del corpus de documentos 
    #  
    #Se puede re definir las stop_words
    
    

    

    return data_lemmatized

#def vectorizar(data_lemmatized):
 #   
  #  data_vectorized = vectorizer.fit_transform(data_lemmatized)
#
 #   return data_vectorized



import re
import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

#nlp = spacy.load('en', disable=['parser', 'ner'])
custom_stopwords = ['researchgate', 'www', 'author']

# remove --
def remove_hypens(book_text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', book_text)

# tokenize text
def tokenize_text(book_text):
    TOKEN_PATTERN = r'\s+'
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=True)
    word_tokens = regex_wt.tokenize(book_text)
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_stopwords(tokens, custom_stopwords):
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list += custom_stopwords
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def get_lemma(tokens):
    lemmas = []
    for word in tokens:
        lemma = wn.morphy(word)
        if lemma is None:
            lemmas.append(word)
        else:
            lemmas.append(lemma)
    return lemmas

def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]

def keep_only_words_in_wordnet(tokens):
    return [token for token in tokens if wn.synsets(token)]

def apply_lemmatize(tokens, wnl=WordNetLemmatizer()):
    return [wnl.lemmatize(token) for token in tokens]

def list_to_string(lst): 
      
    return ' '.join(lst)

def cleanText(doc_texts):
    clean_doc = []
    for doc in doc_texts:
        doc = remove_hypens(doc)
        doc_i = tokenize_text(doc)
        doc_i = remove_characters_after_tokenization(doc_i)
        doc_i = convert_to_lowercase(doc_i)
        doc_i = remove_stopwords(doc_i, custom_stopwords)
        doc_i = get_lemma(doc_i)
        doc_i = remove_short_tokens(doc_i)
        doc_i = keep_only_words_in_wordnet(doc_i)
        doc_i = apply_lemmatize(doc_i)
        doc_i = list_to_string(doc_i)
        clean_doc.append(doc_i)
    

    return clean_doc

def Convert_to_list(doc_texts):
    
    doc_text_words = list(sent_to_words(doc_texts))

    return doc_text_words