import configparser
import pickle as pk
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange

from hm import hamming_score
from sklearn.metrics import f1_score
from Lstm_simple import Base_model


import random
import time
import os 
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split

#load configuation

config = configparser.RawConfigParser()
config.read('config.properties')
parameter_dict = dict(config.items('BiLstm network'))
boolean_dict = {'None': None, 'True': True, 'False': False}

#load data files 
with open(parameter_dict['sentence_path'] ,'rb') as f:
    X_data = pk.load(f)

with open(parameter_dict['labels_path']   ,'rb') as f:
    y_data = pk.load(f)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=0.8)




def get_train_data(batch_size, slice_no):


    batch_data_j = np.array(X_train[slice_no * batch_size:(slice_no + 1) * batch_size])
    batch_labels = np.array(y_train[slice_no * batch_size:(slice_no + 1) * batch_size])
    
    max_sequence = max(list(map(len, batch_data_j)))

    # getting Max length of sequence
    padded_sequence = [i + [0] * (max_sequence - len(i)) if len(i) < max_sequence else i for i in batch_data_j]
    
    return {'sentenc': padded_sequence, 'labels': batch_labels}


def get_test_data(batch_size,slice_no):


    batch_data_j = np.array(X_val[slice_no * batch_size:(slice_no + 1) * batch_size])
    batch_labels = np.array(y_val[slice_no * batch_size:(slice_no + 1) * batch_size])
    
    max_sequence = max(list(map(len, batch_data_j)))
    
    padded_sequence = [i + [0] * (max_sequence - len(i)) if len(i) < max_sequence else i for i in batch_data_j]
    
    return {'sentenc': padded_sequence, 'labels': batch_labels}



def evaluate_(model, epoch_, batch_size = 120):

    sess = tf.get_default_session()
    iteration = len(X_val) // batch_size

    sub_accuracy    = []
    hamming_score_a = []
    hamming_loss_   = []

    micr_ac = []
    weight_ac = []

    for i in range(iteration):
        
        data_g = get_test_data(batch_size,i)
        
        sentences_data = data_g['sentenc']
        labels_data    = data_g['labels']

        network_out, targe = sess.run([model.predictions,model.targets], feed_dict={model.placeholders['sentence']: sentences_data,
                                                                                    model.placeholders['labels']: labels_data, 
                                                                                    model.placeholders['dropout']: 0.0})

        h_s     = hamming_score(targe, network_out)

        ham_sco = h_s['hamming_score']
        sub_acc = h_s['subset_accuracy']
        ham_los = h_s['hamming_loss']

        sub_accuracy.append(sub_acc)
        hamming_score_a.append(ham_sco)
        hamming_loss_.append(ham_los)



        micr_ac.append(f1_score(targe, network_out, average='micro'))
        weight_ac.append(f1_score(targe, network_out, average='weighted'))

    return {  'subset_accuracy' : np.mean(np.array(sub_accuracy)) , 
              'hamming_score'   : np.mean(np.array(hamming_score_a)) , 
              'hamming_loss'    : np.mean(np.array(hamming_loss_)), 
               'micro_ac'       : np.mean(np.array(micr_ac)), 
               'weight_ac'      : np.mean(np.array(weight_ac)) , 'epoch': epoch_ }





def train_model(model, batch_size = int(parameter_dict['batch_size']), epoch = int(parameter_dict['epoch'])):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        iteration = len(X_train) // batch_size


        for i in range(epoch):
            t = trange(iteration, desc='Bar desc', leave=True)

            for j in t:



                data_g = get_train_data(batch_size,j)
                sentences_data = data_g['sentenc']
                labels_data    = data_g['labels']



                network_out, train, targe, losss  = sess.run([model.predictions, model.optimizer, model.targets,model.loss],
                                          feed_dict={model.placeholders['sentence']: sentences_data,
                                                     model.placeholders['labels']: labels_data,
                                                     model.placeholders['dropout']: 0.0})

                t.set_description("epoch {},  iteration {},  F1_score {},  loss {}".format(i,
                                                                                       j,
                                                                                       f1_score(targe, 
                                                                                                network_out, 
                                                                                                average='micro'), 
                                                                                       losss))
                t.refresh() # to show immediately the update


            val_data = evaluate_(model, i, batch_size = 100)
            print("validation_acc",val_data)
            with open('./result/iterres.txt', 'a') as f:
                f.write(str({'test_accuracy':  val_data}) + '\n')
                
                
if __name__ == "__main__":
    
    model = Base_model(vocab_size                  =   int(parameter_dict['vocab_size']),
                       rnn_units                   =   int(parameter_dict['rnn_units']), 
                       word_embedding_dim          =   int(parameter_dict['word_embedding_dim']),  
                       no_of_labels                =   int(parameter_dict['no_of_labels']), 
                       learning_rate               =   float(parameter_dict['learning_rate']),   
                       trained_embedding           =   boolean_dict[parameter_dict['trained_embedding']], 
                       train_embedding             =   boolean_dict[parameter_dict['train_embedding']],
                       model_output                =   boolean_dict[parameter_dict['model_output']])
    
    train_model(model)