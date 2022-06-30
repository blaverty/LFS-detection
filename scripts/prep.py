import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# prepare data
# import data
data_all = pd.read_csv("../../final_model_input.csv")
y = data_all[["TP53"]]
data = data_all.drop(labels='TP53', axis=1)

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.30, stratify=y, shuffle=True)
folds = StratifiedKFold(n_splits=5, shuffle=False) # use same folds for each estimator, allows best model for each estimator so can calculate comparision stats and figures

# save test sets 
dill.dump(x_train, file = open("output_f1/x_train", "wb"))
dill.dump(y_train, file = open("output_f1/y_train", "wb"))
dill.dump(x_test, file = open("output_f1/x_test", "wb"))
dill.dump(y_test, file = open("output_f1/y_test", "wb"))

# save folds
dill.dump(folds, file = open("output_f1/folds", "wb"))

