
# coding: utf-8

# In[1]:


from data import *
x,y = get_raw_data()
x = lower_case(x)
x = chunk(x,150)
x = preproces(x)
x = steamit(x)
x = stopit_dude(x)


# In[2]:


df = labels_to_dataframe(x,y)
ration_data, freq_list = keep_labels(df, keep_ratio = 0.25, freq_Value = False)


# In[3]:


#train test split
import numpy as np


# In[6]:


#tf-idf vectors

ration_data_text = list(ration_data['text'])
labels           = np.array(ration_data.drop('text', 1))


# In[7]:


#split the data

X_train, X_val, y_train, y_val = split_data(ration_data_text,labels)


# In[8]:


train_x, test_x = tf_idf(X_train,X_val)


# In[10]:


#train on baseline models

import baseline

models = baseline.Base_models(train_x, y_train, test_x, y_val)


# In[12]:


try:
    result_a = models.BinaryRe()
    with open('result.txt','a') as f:
        
        f.write(str({'model'  : 'BinaryRe', 'result' :  result_a}) + '\n')
except Exception as e:
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'BinaryRe', 'result' :  e}) + '\n')
    
try:
    result_a = models.powerset()
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'powerset', 'result' :  result_a}) + '\n')
except Exception as e:
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'powerset', 'result' :  e})  + '\n')
        
try:
    result_a = models.mlknn()
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'mlknn', 'result' :  result_a}) + '\n')
except Exception as e:
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'mlknn', 'result' :  e}) + '\n')
        
try:
    result_a = models.classfier_chain()
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'classfier_chain', 'result' :  result_a}) + '\n')
except Exception as e:
    with open('result.txt','a') as f:
        f.write(str({'model'  : 'classfier_chain', 'result' :  e}) + '\n')

