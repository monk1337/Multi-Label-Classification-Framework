# # Convert text to lowercase – This is to avoid distinguish between words simply on case.

# # Remove Number – Numbers may or may not be relevant to our analyses. Usually it does not carry any importance in sentiment analysis

# # Remove Punctuation – Punctuation can provide grammatical context which supports understanding. For bag of words based sentiment analysis punctuation does not add value.

# # Remove English stop words – Stop words are common words found in a language. Words like for, of, are etc are common stop words.

# # Remove Own stop words(if required) – Along with English stop words, we could instead or in addition remove our own stop words. The choice of own stop word might depend on the domain of discourse, and might not become apparent until we’ve done some analysis.

# # Strip white space – Eliminate extra white spaces.

# # Stemming – Transforms to root word. Stemming uses an algorithm that removes common word endings for English words, such as “es”, “ed” and “’s”. For example i.e., 1) “computer” & “computers” become “comput”

# # Lemmatisation – transform to dictionary base form i.e., “produce” & “produced” become “produce”

# # Sparse terms – We are often not interested in infrequent terms in our documents. Such “sparse” terms should be removed from the document term matrix.


import pickle as pk
import sys
import warnings
import operator
import pandas as pd
import string
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import nltk
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re

from sklearn.preprocessing import MultiLabelBinarizer
import os


# from nlpre import titlecaps, dedash, identify_parenthetical_phrases
# from nlpre import replace_acronyms, replace_from_dictionary

# def nlpre_pipeline(sentences):
#     ABBR = identify_parenthetical_phrases()(sentences)
#     parsers = [dedash(), titlecaps(), replace_acronyms(ABBR),
#            replace_from_dictionary(prefix="MeSH_")]
#     for f in parsers:
#         sentences = f(sentences)
        
#     return sentences


def lower_case(sentences):
    return [sentence.lower() for sentence in sentences]

def keep(sentences,labels,ratio):
    data_size = int(len(sentences) * ratio)
    sentences = sentences[:data_size]
    labels    = labels[:data_size]
    return sentences,labels

def split_data(sentences,labels, train_ration = 0.8):
    X_train, X_val, y_train, y_val = train_test_split(sentences, labels, train_size = train_ration)
    return X_train, X_val, y_train, y_val


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def preproces(sentences):
    all_data = []
    for sentence in tqdm(sentences):
        all_data.append(keepAlpha(cleanPunc(cleanHtml(sentence.lower()))))
        
    return all_data

        
        


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    return " ".join(sentence.translate(str.maketrans('', '', string.punctuation)).split())


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent



stop_words    = set(stopwords.words('english'))
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

def steamit(sentences):
    all_data = []
    for sentence in tqdm(sentences):
        all_data.append(stemming(sentence))
    return all_data

def stopit_dude(sentences):
    all_data = []
    for sentence in tqdm(sentences):
        all_data.append(removeStopWords(sentence))
    return all_data


def tf_idf(train_sentences,test_sentences):
    
    #error while passing list to tfidf vectors
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
    
    return x_train, x_test


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
        sorted_long = sorted([col for col in sorted_long if int(col[1])>= freq_Value])
        
    if keep_ratio:
        keep_ratio   = int(len(sorted_long)* keep_ratio)
        sorted_long  = sorted(sorted_long[:keep_ratio])
    
    keep_columns = [col[0] for col in sorted_long]
    
    
    df_          = df_[sorted(keep_columns)]
    df_['text']  = text_col
    
    #remove all rows where all labels are zeros
    df_          = df_[(df_.loc[:, df_.columns != 'text'].T != 0).any()]
    df_          = df_.reset_index(drop=True)

    return df_, sorted_long

def adj_matrix(df):
    
    df = df.drop('text', 1)
    u = np.diag(np.ones(df.shape[1], dtype=bool))
    return df.T.dot(df) * (~u)


def final_adj_matrix(adj_m, freq_list):
    
    all_num  = []
    all_rela = []
    
    for value in freq_list:
        all_num.append(value[1])
        all_rela.append(value[0])
        
        
    final_matrix = {}

    final_matrix['nums']   = np.array(list(all_num))
    final_matrix['adj']    = np.array(adj_m,np.float32)
    final_matrix['labels'] = list(adj_m.columns)
    
    return final_matrix



def get_raw_data():
    
    nltk.download("reuters")
    from nltk.corpus import reuters
    
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
    
    return all_dataa,all_lavelsa

def get_sentence_length(sentences):
    
    lenths = []
    
    for sentence in tqdm(sentences):
        if isinstance(sentence,list):
            lenths.append(len(sentence))
        else:
            lenths.append(len(sentence.split()))
    return "max {} min {} average {}".format(np.max(np.array(lenths)), 
                                             np.min(np.array(lenths)), 
                                             np.average(np.array(lenths)))


def chunk(sentences,value):
    
    return [" ".join(sentence.split()[:value]) for sentence in sentences]



def loadGloveModel(gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model
    
    
def get_label_dict():
    
    return {'cocoa': 'cocoa', 'acq': ['corporate', 'acquisitions'], 'money-supply': ['money', 'supply'], 'corn': 'corn', 'earn': 'earn', 'trade': 'trade', 'crude': 'crude', 'nat-gas': ['natural', 'gas'], 'coffee': 'coffee', 'sugar': 'sugar', 'veg-oil': ['vegetable', 'oil'], 'gas': 'gas', 'iron-steel': ['iron', 'steel'], 'ship': 'ship', 'money-fx': 'money', 'cotton': 'cotton', 'dlr': 'dlr', 'interest': 'interest', 'grain': 'grain', 'wheat': 'wheat', 'carcass': 'carcass', 'livestock': 'livestock', 'gnp': 'gnp', 'jobs': 'jobs', 'strategic-metal': ['strategic', 'metal'], 'oilseed': 'oilseed', 'soybean': 'soybean', 'barley': 'barley', 'meal-feed': ['meal', 'feed'], 'sorghum': 'sorghum', 'soy-oil': ['soy', 'oil'], 'gold': 'gold', 'lei': 'lei', 'ipi': 'ipi', 'alum': 'alum', 'cpi': 'cpi', 'reserves': 'reserves', 'tea': 'tea', 'bop': 'bop', 'tin': 'tin', 'housing': 'housing', 'yen': 'yen', 'lead': 'lead', 'silver': 'silver', 'zinc': 'zinc', 'rice': 'rice', 'heat': 'heat', 'pet-chem': ['pet', 'chem'], 'income': 'income', 'rubber': 'rubber', 'dmk': 'dmk', 'rapeseed': 'rapeseed', 'sunseed': 'sunseed', 'hog': 'hog', 'fuel': 'fuel', 'orange': 'orange', 'copper': 'orange', 'lumber': 'lumber', 'palm-oil': ['palm', 'oil'], 'soy-meal': ['soy', 'meal'], 'wpi': 'wpi', 'oat': 'oat', 'retail': 'retail', 'platinum': 'platinum'}


#there are two word_to_vec function 

#this is for multilabel and generate label embedding
def label_correlation_matrix(labels_list):
    
    load_embeddings = loadGloveModel('glove.6B.300d.txt')
    
    labels_embeddings = []

    for i in tqdm(labels_list):
        if isinstance(i,list):
            dta_both = []
            for k in i:
                dta_both.append(load_embeddings[k])
            labels_embeddings.append(np.average(dta_both,axis=0))
        else:
            labels_embeddings.append(load_embeddings[i])
        
    return np.array(labels_embeddings)


def vocab_freq(sentences, 
               keep_ratio = False, 
               freq_Value = False, 
               custom_value = False):
    
    vocab_frequency = []
    for sentence in tqdm(sentences):
        if isinstance(sentence,list):
            vocab_frequency.extend(sentence)
        else:
            vocab_frequency.extend(sentence.split())
    freq = Counter(vocab_frequency)
    
    sorted_long   = sorted(freq.items(), key=operator.itemgetter(1),reverse=True)
    
    if freq_Value:
        sorted_long = sorted([col for col in sorted_long if int(col[1])>= freq_Value])
        
    if keep_ratio:
        keep_ratio   = int(len(sorted_long)* keep_ratio)
        sorted_long  = sorted(sorted_long[:keep_ratio])
        
    if custom_value:
        sorted_long = sorted(sorted_long[:custom_value])
        
    freq_num        = set([k[1] for k in sorted_long])
    
    word_to_int = [(n[0],s) for s,n in enumerate(sorted_long,2)]
    word_to_int.extend([('unk',1),('pad',0)])
    
    
    int_to_word = [(n,m) for m,n in word_to_int]
        
    return sorted_long, freq_num,word_to_int,int_to_word


def encoder(sentences, vocab_dict):
    vocab_dict = dict(vocab_dict)
    all_sentences = []
    
    for sentence in tqdm(sentences):
        token = nltk.word_tokenize(sentence)
        encoded_token = []
        for k in token:
            if k in vocab_dict:
                
                encoded_token.append(vocab_dict[k])
            else:
                encoded_token.append(vocab_dict['unk'])
        all_sentences.append(encoded_token)
    
    return all_sentences, vocab_dict


#there are two word_to_vec function 

#this is for vocab embedding

def vocab_embedding(vocab):
    
    vocab   = sorted(dict(vocab).items(), key=operator.itemgetter(1))
        
    encoded_vocab = []
    not_in_embedding = []
    vocas = [token[0].lower() for token in vocab]
    load_embeddings = loadGloveModel('glove.6B.300d.txt')
    
    for token in tqdm(vocas):
        if token in load_embeddings:
            encoded_vocab.append(load_embeddings[token])
        else:
            not_in_embedding.append(token)
            encoded_vocab.append(load_embeddings['unk'])
            
    return np.array(encoded_vocab), not_in_embedding

# how many rows you want
def data_volume(dataframe, volume):
    
    df = dataframe.sample(frac=1).reset_index(drop=True)
    return df.head(int(volume))


def get_quick_data(freq_label,how_much, pickle_ = True, baseline_classical = True, deep_learning = True):
    
    x,y = get_raw_data()
    x = lower_case(x)
    x = chunk(x,150)
    x = preproces(x)
    df = labels_to_dataframe(x,y)
    ration_data, freq_list = keep_labels(df, keep_ratio = freq_label, freq_Value = False)
    df = data_volume(ration_data,how_much)
    ration_data_text = list(df['text'])
    labels           = np.array(df.drop('text', 1))
            
    if baseline_classical:
        X_train, X_val, y_train, y_val = split_data(ration_data_text,labels)
        train_x, test_x = tf_idf(X_train,X_val)
        
        with open('baseline_classical_train_x.pkl','wb') as f:
            pk.dump(train_x,f)
        
        with open('baseline_classical_y_train.pkl','wb') as f:
            pk.dump(y_train,f)
            
        with open('baseline_classical_test_x.pkl','wb') as f:
            pk.dump(test_x,f)
            
        with open('baseline_classical_y_val.pkl','wb') as f:
            pk.dump(y_val,f)
        
        
    if deep_learning:
        sorted_long, freq_num,word_to_int,int_to_word = vocab_freq(ration_data_text)
        all_sentences, vocab_dict = encoder(ration_data_text,word_to_int)
        dgh = adj_matrix(ration_data)
        final_ = final_adj_matrix(dgh,freq_list)
        label_dict = get_label_dict()
        #map it 

        final_labels = []
        for labels_a in freq_list:
            final_labels.append(label_dict[labels_a[0]])
            
        embeddin = label_correlation_matrix(final_labels)
        
        with open('deep_learning_all_sentences.pkl','wb') as f:
            pk.dump(all_sentences,f)
            
        with open('deep_learning_all_labels.pkl','wb') as f:
            pk.dump(labels,f)
            
        with open('deep_leaning_adj_matrix.pkl','wb') as f:
            pk.dump(final_,f)
            
        with open('deep_leaning_label_embedding.pkl','wb') as f:
            pk.dump(embeddin,f)
            
    return len(word_to_int), labels.shape

# print(get_quick_data(0.25, 4000 , pickle_ = True))
