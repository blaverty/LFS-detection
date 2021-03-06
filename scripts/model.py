import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
import shap
import umap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings

class Model:
	''' 
	Train a model to classify LFS patients 

	Arguments
	datafile: feature matrix with labels, includes header
	model_type: string specifying one of rf, gbt, svm, log
	base: string specifying output directory
	'''
	def __init__(self, datafile, base):
		'''read in dataframe and initialize model'''
		self.df = pd.read_csv(datafile) # read in data
		self.base = base+"/"

	def split(self):
		'''split data into training and testing'''
		y = self.df[["TP53"]] # labels
		data = self.df.drop(labels="TP53", axis=1) # remove labels from data
		self.folds = StratifiedKFold(n_splits=5, shuffle=False) # use same folds for each estimator
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, y, test_size=0.30, stratify=y, shuffle=True) # split into training and testing sets

	def save_splits(self):
		''' save files ''' 
		dill.dump(self.x_train, file = open(self.base+"x_train", "wb")) # save x_train file
		dill.dump(self.y_train, file = open(self.base+"y_train", "wb"))
		dill.dump(self.x_test, file = open(self.base+"x_test", "wb"))
		dill.dump(self.y_test, file = open(self.base+"y_test", "wb"))
		dill.dump(self.folds, file = open(self.base+"folds", "wb"))
		self.x_train = self.x_train.drop(labels='sample', axis=1) # remove sample column after saving
		self.x_test = self.x_test.drop(labels='sample', axis=1)

	def init_pipeline(self, pipeline_options): 
		''' 
		initiate pipeline object

		Arguments
		pipeline_options: dictionary specifying which pipeline steps to include
		'''
		d = dict()
		d['StandardScaler()'] = StandardScaler()
		d['PCA()'] = PCA()
		d['SMOTE()'] = SMOTE()
		d['RandomForestClassifier()'] = RandomForestClassifier()
		pipe = list() # list to hold pipeline tuples
		for k, v in pipeline_options.items(): 
			if v != 'None': # remove steps set to None
				v2 = d[v] # change pipeline options from strings to functions
				k = (k, v2) # make tuple of each step from dictionary
				pipe.append(k) # append step to list
		self.pipeline = Pipeline(pipe) # initialize pipeline, reassigned in param dict

	def parameters_preprocessing(self):
		''' add all preprocessing options to pipeline '''
		k = self.pipeline.get_params().keys() # steps in pipeline
		self.param = {} # dictionary for parameter options 
		if 'normalization' in k:
			warnings.filterwarnings("ignore", message="n_quantiles is greater than the total number of samples. n_quantiles is set to n_samples.")
			self.param['normalization'] = [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileTransformer()] # normalization options
		if 'dimensionality_reduction' in k:
			self.param['dimensionality_reduction'] = [PCA()] # dimensionatliy reduction options
			self.param['dimensionality_reduction__n_components'] = [0.8, 0.85, 0.9, 0.95] # components for PCA
		if 'sampling' in k:
			self.param['sampling'] = [SMOTE(), KMeansSMOTE(), SVMSMOTE(), BorderlineSMOTE()]  # oversampling options

	def parameters_classifier(self, model_type):	
		''' 
		initiate parameter dictionary
		
		Arguments
		model_type: string specifing classifier type.  one of rf, svm, log, gbt
		 '''
		if model_type == "rf": # parameters specific to each model
			self.param['classifier'] = [RandomForestClassifier()]
			self.param['classifier__n_estimators'] = [1000, 1500, 2000, 2500] 
			self.param['classifier__max_features'] = [8, 10, 12, 14, 16, 18, 20, 22]
			self.param['classifier__min_samples_leaf'] = [1, 3, 5, 7, 9]
			self.param['classifier__min_samples_split'] = [2, 4, 6, 8]
		if model_type ==  "gbt":
			self.param['classifier'] = [GradientBoostingClassifier()]
			self.param['classifier__n_estimators'] = [1000, 1500, 2000, 2500] 
			self.param['classifier__learning_rate'] = [0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
			self.param['classifier__min_samples_leaf'] = [1, 3, 5, 7, 9, 11, 13]
			self.param['classifier__min_samples_split'] = [2, 4, 6, 8, 10]
		if model_type ==  "svm":
			self.param['classifier'] = [SVC(probability=True)]
			self.param['classifier__C'] = [0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
			self.param['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid'] # if poly works well then tune degree 
			self.param['classifier__degree'] = [2, 3, 4, 5, 6] # degree for poly
			self.param['classifier__gamma'] = [0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
		if model_type == "log":
			self.param['classifier'] = [LogisticRegression()]
			self.param['classifier__C'] = [0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10, 50, 100]
			self.param['classifier__penalty'] = ['l1', 'l2', 'elasticnet']
			self.param['classifier__solver'] = ['saga', 'newton-cg', 'lbfgs', 'liblinear', 'sag'] # some will give errors because not compatible with penalty
			self.param['classifier__max_iter'] = [50000]

	def fit(self, score, model_type):
		''' 
		fit model with grid search
		
		Arguments
		score: string specifying score to optimize during CV
		model_type: string specifying classifier.  one of rf, gbt, svm, log
		'''
		self.gs = GridSearchCV(self.pipeline, self.param, cv=self.folds, n_jobs=-1, scoring=score, refit=score).fit(self.x_train, self.y_train.values.ravel()) # train model using grid search 
		dill.dump(self.gs, file = open(self.base+"model_"+model_type, "wb"))
	def import_files(self, model_type):
		''' 
		import saved files for future shap, predict, and score functions

		Arguments
		model_type: string specifying classifier.  one of rf, gbt, svm, log
		'''

		self.gs = dill.load(open(self.base+"model_"+model_type, "rb"))
		self.x_train = dill.load(open(self.base+"x_train", "rb"))
		self.x_test = dill.load(open(self.base+"x_test", "rb"))
		self.y_test = dill.load(open(self.base+"y_test", "rb"))
		self.x_train = self.x_train.drop(labels='sample', axis=1) # remove sample column 
		self.x_test = self.x_test.drop(labels='sample', axis=1)

	def shap(self):
		''' find feature importance with shap '''
		explainer = shap.KernelExplainer(self.gs.predict_proba, self.x_train) #shap.sample(x_train, 50)) # explain predictions of the model
		#test_sample = x_test.iloc[:250,:]
		self.shap_values = explainer.shap_values(self.x_test)
		dill.dump(shap_values, file = open(self.base+"shapvalues", "wb"))

	def predict(self):
		self.test_pred = self.gs.best_estimator_.predict(self.x_test)
		self.test_prob = self.gs.best_estimator_.predict_proba(self.x_test)
		self.cm = confusion_matrix(y_true = self.y_test.values.ravel(), y_pred = self.test_pred)

	def acc(self): # accuracy
		return (accuracy_score(y_true=self.y_test, y_pred=self.test_pred))
	
	def prec(self): # ppv
		return (precision_score(y_true=self.y_test, y_pred=self.test_pred))

	def recall(self): # recall
		return (recall_score(y_true=self.y_test, y_pred=self.test_pred,))

	def npv(self): # npv 
  		return (self.cm[0,0]/(self.cm[0,0]+self.cm[1,0]))

	def spec(self): # specificity
  		return (self.cm[0,0]/(self.cm[0,0]+self.cm[0,1]))
	
	def f1(self): # f1 score
		return (f1_score(y_true=self.y_test, y_pred=self.test_pred))

	def calc_auc(self): # auroc
		fpr, tpr, _ = roc_curve(self.y_test, self.test_prob[:,1]) # fpr and tpr 
		return (auc(fpr, tpr)) 
	
	def calc_auprc(self): # auprc
		return (average_precision_score(self.y_test, self.test_prob[:, 1]))

	def score(self): 
		''' calculate test scores '''
		self.score = {}
		self.score["accuracy"] = self.acc() 
		self.score["precision"] = self.prec()
		self.score["recall"] = self.recall()
		self.score["npv"] = self.npv()
		self.score["specificity"] = self.spec()
		self.score["f1"] = self.f1()
		self.score["auc"] = self.calc_auc() 
		self.score["auprc"] = self.calc_auprc()
