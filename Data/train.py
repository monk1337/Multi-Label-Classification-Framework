import data
import baseline
import numpy as np
x_data, y_data = data.get_keras_data()
x_data = [" ".join(x.split()[:100]) for x in x_data]


X_train, X_val, y_train, y_val = data.split_data(x_data,y_data)
X_train, X_val = data.tf_idf(X_train,X_val)
#get baseline accuracy
models = baseline.Base_models(X_train, y_train, X_val, y_val)
print(models.BinaryRe())
print(powerset())
print(mlknn())
