from Baseline_models import *
import pickle as pk



with open('../data/baseline_classical_train_x.pkl','rb') as f:
    train_x = pk.load(f)

with open('../data/baseline_classical_y_train.pkl','rb') as f:
    y_train = pk.load(f)
            
with open('../data/baseline_classical_test_x.pkl','rb') as f:
    test_x  = pk.load(f)
            
with open('../data/baseline_classical_y_val.pkl','rb') as f:
    y_val  = pk.load(f)
    
    
models = Base_models(train_x, y_train, test_x, y_val)


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