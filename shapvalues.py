import dill
import pandas as pd
import numpy as np
import shap

# load data
model = dill.load(open("output_larger_f1/model_log", "rb"))
test_x = dill.load(open("output_larger_f1/x_test", "rb"))
test_y = dill.load(open("output_larger_f1/y_test", "rb"))
train_x = dill.load(open("output_larger_f1/x_train", "rb"))
train_y = dill.load(open("output_larger_f1/y_train", "rb"))

# remove sample column
train_x = train_x.drop(labels='sample', axis=1)
test_x = test_x.drop(labels='sample', axis=1)

# explain predictions of the model
explainer = shap.KernelExplainer(model.predict_proba, train_x) #shap.sample(train_x, 50)) 
#test_sample = test_x.iloc[:250,:]
shap_values = explainer.shap_values(test_x)

dill.dump(shap_values, file = open("shapvalues_editted", "wb"))
dill.dump(explainer, file = open("shapexplainer_editted", "wb"))
