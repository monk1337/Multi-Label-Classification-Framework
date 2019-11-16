import pickle as pk
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re



def lower_case(sentences):
    return [sentence.lower() for sentence in sentences]

def keep(sentences,labels,ratio):
    data_size = int(len(sentences) * ratio)
    sentences = sentences[:data_size]
    labels    = labels[:data_size]
    return sentences,labels

def split_data(sentences,labels):
    X_train, X_val, y_train, y_val = train_test_split(sentences, labels, train_size=0.8)
    return X_train, X_val, y_train, y_val


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent



stop_words = set(stopwords.words('english'))
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)



stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def tf_idf(train_sentences,test_sentences):
    
    #error while passing list 
    #solution : converting into pandas frame

    pd_train_frame = pd.DataFrame({'text': train_sentences})
    pd_test_frame =  pd.DataFrame({'text': test_sentences})
    pd_train_frame = pd_train_frame['text']
    pd_test_frame  = pd_test_frame['text']
    
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(pd_train_frame)
    vectorizer.fit(pd_test_frame)
    x_train = vectorizer.transform(pd_train_frame)
    x_test = vectorizer.transform(pd_test_frame)
    
    return x_train,x_test


#demo data
def get_keras_data():
    
    with open('r_steamed_st.pkl','rb') as f:
        sentences = pk.load(f)
    with open('r_steamed_lb.pkl','rb') as f:
        labels = pk.load(f)
        
    return lower_case(sentences), labels

def get_retu_data(keep_ratio):
    
    with open('reuters_sentences_st.pkl','rb') as f:
        sentences = pk.load(f)
    
    with open('all_raw_labels_keras.pkl','rb') as f:
        labels = pk.load(f)

    with open('label_order_keras.pkl','rb') as f:
        columns = pk.load(f)

    with open('label_encoding_keras.pkl','rb') as f:
        label_embedding = pk.load(f)

    with open('one_hot_encoding_keras.pkl','rb') as f:
        one_hot = pk.load(f)

    with open('adj_matrix_keras.pkl','rb') as f:
        adj_matrix = pk.load(f)
        
    
        
    if keep_ratio:
        sentences, labels = keep(sentences,labels,keep_ratio)
        
        
    return lower_case(sentences),labels,columns,label_embedding,one_hot,adj_matrix

from sklearn.preprocessing import MultiLabelBinarizer


def labels_to_dataframe(sentences,labels):
    mlb              = MultiLabelBinarizer()
    labels_on        = mlb.fit_transform(labels)
    pd_data          = pd.DataFrame(labels_on)
    pd_data.columns  = mlb.classes_
    pd_data['text']  = sentences
    return pd_data

def one_hot_to_dataframe(sentences,labels,columns_name):

    pd_data          = pd.DataFrame(labels)
    pd_data.columns  = columns_name
    pd_data['text']  = sentences
    return pd_data

def dataframe_columns(df,columns_list):
    text_col = df['text']
    df = df.drop('text', 1)
    if len(df.columns) == len(columns_list):
        df = df.reindex(columns = columns_list)
        df['text'] = text_col
        return df
    else:
        return 'column list is not equal to data_frame columns'
    
        
def keep_labels(df, keep_ratio = False, freq_Value = False):
    
    text_col = df['text']
    df_ = df.drop('text', 1)
    
    get_frequency = {}
    
    for column in df_.columns:
        get_frequency[column]= (df_[column]==1).sum()
    sorted_long   = sorted(get_frequency.items(), key=operator.itemgetter(1),reverse=True)
    raw_frequency = sorted_long
    
    if freq_Value:
        sorted_long = [col for col in sorted_long if int(col[1])>= freq_Value]
        
    if keep_ratio:
        keep_ratio   = int(len(sorted_long)* keep_ratio)
        sorted_long  = sorted_long[:keep_ratio]
        
    keep_columns = [col[0] for col in sorted_long]
    df_          = df_[keep_columns]
    df_['text']  = text_col
    return df_, raw_frequency

def adj_matrix(df):
    u = np.diag(np.ones(df.shape[1], dtype=bool))
    return df.T.dot(df) * (~u)
def get_raw_data():
    
    documents = reuters.fileids()
    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))
    X_train = [(reuters.raw(doc_id)) for doc_id in train_docs_id]
    X_test = [(reuters.raw(doc_id)) for doc_id in test_docs_id]


    mlb = MultiLabelBinarizer()
    y_train = [reuters.categories(doc_id)
                                 for doc_id in train_docs_id]
    y_test = [reuters.categories(doc_id)
                            for doc_id in test_docs_id]

    all_dataa     =    X_train +  X_test
    all_lavelsa   =    y_train +  y_test



    mlb = MultiLabelBinarizer()
    datas_y = mlb.fit_transform(all_lavelsa)
    
    return all_dataa, datas_y, mlb.classes_
