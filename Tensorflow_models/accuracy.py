import ast
import numpy as np

def get_accuracy(file):
    accuracy = []
    with open(file,'r') as f:
        read = f.readlines()
        
    dict_data = {}

        
    for k in read:
        accuracy.append(float(ast.literal_eval(k)['test_accuracy']['micro_ac']))
        
    accu = np.array(accuracy)
    
    
    return file, np.max(accu)

import os
all_files = os.listdir()

final_dict = {}

for i in all_files:
    if '.txt' in i:
        fil, dat = get_accuracy(i)
        final_dict[fil] = dat

print(final_dict)
