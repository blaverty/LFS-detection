import numpy as np
import pandas as pd
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
print("Starting RF")
rf = RandomForestClassifier()

# pipeline setup
pipeline = Pipeline([('normalization', RobustScaler()), # reassigned in param dict
                     ('dimensionality_reduction', PCA()), # reassigned in param dict
                     ('sampling', KMeansSMOTE()), # reassigned in param dict
                     ('classifier', rf)]) # classifier reassigned in param dict

# initiaze the hyperparameters for each dictionary
# preprocessing
scalers = [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileTransformer()]
dim_red = [PCA(n_components=0.9), umap.umap_.UMAP(n_components=5)] # SVD(), KernelPCA()
#number_comp = [0.8, 0.85, 0.9, 0.95, 2, 5]
sampling_strat = [SMOTE(), KMeansSMOTE(), SVMSMOTE(), BorderlineSMOTE()] # can use all for now because not categorical data, SMOTENC(cat_features),

# rf
param_rf = {}
param_rf['normalization']= scalers
param_rf['dimensionality_reduction']= dim_red
#param_rf['dimensionality_reduction__n_components']= number_comp
param_rf['sampling']= sampling_strat
param_rf['classifier__n_estimators'] = [1000, 1500, 2000, 2500] 
param_rf['classifier__max_features'] = [8, 10, 12, 14, 16, 18, 20, 22]
param_rf['classifier__min_samples_leaf'] = [1, 3, 5, 7, 9]
param_rf['classifier__min_samples_split'] = [2, 4, 6, 8]
param_rf['classifier'] = [rf]

print(param_rf)

# train the grid search model
# rf
gs_rf = GridSearchCV(pipeline, param_rf, cv=folds, n_jobs=-1, scoring='f1', refit='f1').fit(x_train, y_train.values.ravel())
print("Done RF")

print(gs_rf.best_score_)
#print(gs_rf.best_estimator_.get_params())

dill.dump(gs_rf, file = open("output_f1_umap/model_rf", "wb"))
