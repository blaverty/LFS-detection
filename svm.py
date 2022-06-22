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
print("Starting SVM")
svm = SVC(probability=True)

# pipeline setup
pipeline = Pipeline([('normalization', RobustScaler()), # reassigned in param dict
                     ('dimensionality_reduction', PCA()), # reassigned in param dict
                     ('sampling', KMeansSMOTE()), # reassigned in param dict
                     ('classifier', svm)]) # classifier reassigned in param dict

# initiaze the hyperparameters for each dictionary
# preprocessing
scalers = [StandardScaler()]
dim_red = [PCA(n_components=0.9), umap.umap_.UMAP(n_components=5)] # SVD(), KernelPCA()
#number_comp = [0.9, 2]
sampling_strat = [SMOTE()]

# svm
param_svm = {}
param_svm['normalization']= scalers
param_svm['dimensionality_reduction']= dim_red
#param_svm['dimensionality_reduction__n_components']= number_comp
param_svm['sampling']= sampling_strat
param_svm['classifier__C'] = [0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
param_svm['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid'] # if poly works well then tune degree 
param_svm['classifier__degree'] = [2, 3, 4, 5, 6] # degree for poly
param_svm['classifier__gamma'] = [0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
param_svm['classifier'] = [svm]

print(param_svm)

# svm
gs_svm = GridSearchCV(pipeline, param_svm, cv=folds, n_jobs=-1, scoring='f1', refit='f1').fit(x_train, y_train.values.ravel())
print("Done SVM")

print(gs_svm.best_score_)
#print("")
#print(gs_svm.best_estimator_.get_params())

dill.dump(gs_svm, file = open("output_f1_umap/model_svm", "wb"))
