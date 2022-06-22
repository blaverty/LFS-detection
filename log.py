import pandas as pd
import numpy as np
import dill
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import umap 

# import sets and folds
x_train = dill.load(open("output_f1_umap/x_train", "rb"))
y_train = dill.load(open("output_f1_umap/y_train", "rb"))
x_test = dill.load(open("output_f1_umap/x_test", "rb"))
y_test = dill.load(open("output_f1_umap/y_test", "rb"))
folds = dill.load(open("output_f1_umap/folds", "rb"))

# remove sample column
x_train = x_train.drop(labels='sample', axis=1)
x_test = x_test.drop(labels='sample', axis=1)

# initialze the estimators
print("Staring LR")
log = LogisticRegression()

# pipeline setup
pipeline = Pipeline([('normalization', RobustScaler()), # reassigned in param dict
                     ('dimensionality_reduction', PCA()), # reassigned in param dict
                     ('sampling', KMeansSMOTE()), # reassigned in param dict
                     ('classifier', log)]) # classifier reassigned in param dict

# initiaze the hyperparameters for each dictionary
# preprocessing
scalers = [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileTransformer()]
dim_red = [umap.umap_.UMAP(n_components=5)] # SVD(), KernelPCA()
#number_comp = [0.8, 0.85, 0.9, 0.95, 2, 5]
sampling_strat = [SMOTE(), KMeansSMOTE(), SVMSMOTE(), BorderlineSMOTE()] # can use all for now because not categorical data, SMOTENC(cat_features), 

# log reg
param_log = {}
param_log['normalization']= scalers
param_log['dimensionality_reduction']= dim_red
#param_log['dimensionality_reduction__n_components']= number_comp
param_log['sampling']= sampling_strat
param_log['classifier__C'] = [0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10, 50, 100]
param_log['classifier__penalty'] = ['l1', 'l2', 'elasticnet']
param_log['classifier__solver'] = ['saga', 'newton-cg', 'lbfgs', 'liblinear', 'sag'] # some will give errors because not compatible with penalty
param_log['classifier__max_iter'] = [1500]
param_log['classifier'] = [log]

print(param_log)

# log reg
gs_log = GridSearchCV(pipeline, param_log, cv=folds, n_jobs=-1, scoring='f1', refit='f1').fit(x_train, y_train.values.ravel())
print("Done LG")

print(gs_log.best_score_)
#print("")
#print(gs_log.best_estimator_.get_params())

dill.dump(gs_log, file = open("output_f1_umap/model_log2", "wb"))
