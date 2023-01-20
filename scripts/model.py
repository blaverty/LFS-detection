import numpy as np
import pandas as pd
import dill
import multiprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from umap import UMAP
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN 
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix, make_scorer, recall_score, precision_score, average_precision_score, f1_score, roc_auc_score
from statistics import median
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings

class Model:
	''' 
	Train a model to classify LFS patients from tumour genomic features

	Arguments
	datafile: feature matrix with labels, includes header
	model_type: string specifying one of rf, gbt, svm, log
	base: string specifying output directory
	'''
	def __init__(self, datafile, base):
		'''read in dataframe and initialize model'''
		self.df = pd.read_csv(datafile) # read in data
		self.base = base+"/"

	def set_train(self):
		'''set training set when dont split into train and test '''
		self.y_train = self.df[["TP53"]] # labels
		self.x_train = self.df.drop(labels="TP53", axis=1) # remove labels from data
		#self.x_train = self.x_train.drop(labels='sample', axis=1) # remove sample column 
	
	def split(self):
		'''split data into training and testing'''
		y = self.df[["TP53"]] # labels
		data = self.df.drop(labels="TP53", axis=1) # remove labels from data
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, y, test_size=0.30, stratify=y, shuffle=True) # split into training and testing sets

	def save_splits(self, model_type):
		''' 
		save files 

		Arguments
		model_type: string specifing classifier type.  one of rf, svm, log, gbt
		'''
		dill.dump(self.x_train, file = open(self.base+"/"+model_type+"/x_train", "wb")) 
		dill.dump(self.y_train, file = open(self.base+"/"+model_type+"/y_train", "wb"))
		dill.dump(self.x_test, file = open(self.base+"/"+model_type+"/x_test", "wb"))
		dill.dump(self.y_test, file = open(self.base+"/"+model_type+"/y_test", "wb"))
		self.x_train = self.x_train.drop(labels='sample', axis=1) # remove sample column after saving
		self.x_test = self.x_test.drop(labels='sample', axis=1) # remove sample column

	def init_pipeline(self, pipeline_options): 
		''' 
		initiate pipeline object

		Arguments
		pipeline_options: dictionary specifying which pipeline steps to include
		'''
		d = dict()
		d['SMOTE()'] = SMOTE()
		d['StandardScaler()'] = StandardScaler()
		d['PCA()'] = PCA()
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
		if 'sampling' in k:
			self.param['sampling'] = [SMOTE(), ADASYN(), BorderlineSMOTE(), SVMSMOTE(), None] # SMOTE(), ADASYN(), BorderlineSMOTE(), SVMSMOTE()  oversampling options
		if 'normalization' in k:
			self.param['normalization'] = [QuantileTransformer(output_distribution="uniform", n_quantiles=25), QuantileTransformer(output_distribution="normal", n_quantiles=25), StandardScaler(), RobustScaler(), MinMaxScaler(), None] #[QuantileTransformer(output_distribution="uniform", n_quantiles=42), QuantileTransformer(output_distribution="normal", n_quantiles=42), StandardScaler(), RobustScaler(), MinMaxScaler()] 
		if 'dimensionality_reduction' in k:
			self.param['dimensionality_reduction'] = [None] # UMAP(n_components=5), FastICA(n_components=5, max_iter=1000, whiten='unit-variance'), None] # dimensionatliy reduction options
#			self.param['dimensionality_reduction__n_components'] = [2, 3, 5, 10, 15] # components for PCA

	def parameters_classifier(self, model_type):	
		''' 
		initiate parameter dictionary specific to each model
		
		Arguments
		model_type: string specifing classifier type.  one of rf, svm, log, gbt
		 '''
		if model_type == "rf": 
			self.param['classifier'] = [RandomForestClassifier()]
			self.param['classifier__n_estimators'] = [500] #, 1500, 2000, 2500] 
			self.param['classifier__max_features'] = [8, 10] #, 12, 14, 16, 18, 20, 22]
			self.param['classifier__min_samples_leaf'] = [1] #, 3, 5, 7, 9]
			self.param['classifier__min_samples_split'] = [2]#, 4, 6, 8]
		if model_type ==  "gbt":
			self.param['classifier'] = [GradientBoostingClassifier()]
			self.param['classifier__n_estimators'] = [500]#, 1500, 2000, 2500] 
			self.param['classifier__learning_rate'] = [1]#0.0001, 0.00015, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10]
			self.param['classifier__min_samples_leaf'] = [1, 2]#, 3, 5, 7, 9, 11, 13]
			self.param['classifier__min_samples_split'] = [2]#, 4, 6, 8, 10]
		if model_type ==  "svm":
			self.param['classifier'] = [SVC(probability=True)]
			self.param['classifier__C'] = [1]#0.0001, 0.001, 0.01, 0.1, 1, 10]
			self.param['classifier__kernel'] = ['linear']#, 'poly', 'rbf', 'sigmoid'] # if poly works well then tune degree 
			self.param['classifier__degree'] = [2, 3] # 4, 5 degree for poly
			self.param['classifier__gamma'] = [1]#0.0001, 0.001, 0.01, 0.1, 1, 10]
		if model_type == "log":
			self.param['classifier'] = [LogisticRegression()]
			self.param['classifier__C'] = [0.001, 0.01, 0.1, 1, 5, 10, 50] #[0.001, 0.0015, 0.01, 0.015, 0.1, 0.5, 1, 5, 10, 50, 100]
			self.param['classifier__penalty'] = ['l2'] # 'l1', 'l2', 'elasticnet'  elastric net needs l1 ratio specificed
			self.param['classifier__solver'] = ['lbfgs'] # 'saga', 'liblinear', 'newton-cg', 'lbfgs', 'sag'  some will give errors because not compatible with penalty
			self.param['classifier__class_weight'] = ['None', 'balanced']
			self.param['classifier__max_iter'] = [10000]

		
		dill.dump(self.param, file = open(self.base+"/"+model_type+"/param", "wb")) # save params for graphing change in performance by param 

	def fit(self, model_type):
		''' 
		fit model with grid search CV
		
		Arguments
		score: string specifying score to optimize during CV
		model_type: string specifying classifier.  one of rf, gbt, svm, log
		'''
	
		manager = multiprocessing.Manager() # parallelize grid search 
		y_predict = manager.list()

		def store_pred(y, y_pred): # custom scorer to save if predicted same as truth
			for n in range(len(y)):
				if y[n]==y_pred[n]:
					y_predict.append(True)
				else:
					y_predict.append(False)
			return(1)

		scores = {
			"auprc": make_scorer(average_precision_score), # optimized in grid seach
			"precision": make_scorer(precision_score, zero_division=1),
			"recall": make_scorer(recall_score),
			"f1": make_scorer(f1_score),
			"specificity": make_scorer(recall_score, pos_label=0),
			"npv": make_scorer(precision_score, pos_label=0, zero_division=1),
			"auc": make_scorer(roc_auc_score),
			"y_pred": make_scorer(store_pred, needs_proba = False) # custom scorer 
			} # metrics calculated during cv
		
		n_split = 10 
		n_repeat = 2 

		folds = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat) 
		self.gs = GridSearchCV(self.pipeline, self.param, cv=folds, scoring=scores, refit='auprc', return_train_score=True).fit(self.x_train.iloc[:,1:], self.y_train.values.ravel()) # train model using grid search, no n_jobs specified so custom scorer works, exclude sample column for x_train
		dill.dump(self.gs, file = open(self.base+"/"+model_type+"/model", "wb")) # save model

		y_prob = list(y_predict) # T/F if prediction same as truth for every fold and parameter combo 
		folds_idx = list(folds.split(self.x_train,self.y_train)) # indices of cv train and test sets 
		test_idx = [] # indices of training set used for cv test sets
		for n in range(len(folds_idx)):
			test_idx.extend(folds_idx[n][1]) # add test indices to list
		n_params = len(self.gs.cv_results_['params']) # number of paramer options
		test_idx.extend(n_params*test_idx) # extend test indices because m param options
		z = list(zip(test_idx, y_prob)) # zip test indices and y predictions together so know which prediction is which sample

		test_idx_2_pred = defaultdict(list) # dictionary with list values
		for k, v in z:
			test_idx_2_pred[k].append(v) # dictionary with key as test index and value as list of predictions from repeated cv with all param options
		index = self.gs.best_index_ # best param index
		test_idx_2_best_param_pred = defaultdict(list) # dictionary with key as testing index and value as  list of samples' predictions from repeated cv with best param option
		n_predictions = n_params*n_repeat-1 # number of predictions
		for k, v in test_idx_2_pred.items(): # iterate through each sample
			index = self.gs.best_index_ # best param index
			while index <= n_predictions: # stop when index out of range
				test_idx_2_best_param_pred[k].append(v[index]) # dictionary with key as testing index and value as  list of samples' predictions from repeated cv with best param option
				index += n_params # change index to repeated cv fold with best param

		samples = self.x_train.iloc[:,0] # samples in order of entire training set for cv, sample index is training set and cv test set index
		train_idx_2_sample = {} # dictionary with key as training set index and value as sample name
		for i in range(len(samples)): # iterate over samples
			train_idx_2_sample[i] = samples[i] # dictionary key is training index and value is sample name

		sample_2_best_param_pred = {} # dictionary with key as sample name and value as list of prediction with best param for all repeated folds 
		for k, v in test_idx_2_best_param_pred.items():
			sample_2_best_param_pred[train_idx_2_sample[k]] = v
#		open(self.base+"/"+model_type+"/classified","a").write(sample_2_best_param_pred)
		dill.dump(sample_2_best_param_pred, file = open(self.base+"/"+model_type+"/classified", "wb"))
		
		for k, v in sample_2_best_param_pred.items(): 
			n=v.count(False) # count number of times prediction is wrong
			if n > len(v)/2: # print sample if prediciton wrong more than 50% of the time
				open(self.base+"/"+model_type+"/half_misclassified","a").write(k + " misclassified " + str(n) + " times\n")

		for k, v in sample_2_best_param_pred.items(): 
			n=v.count(False) # count number of times prediction is wrong
			if n > 0:
				open(self.base+"/"+model_type+"/misclassified","a").write(k + " misclassified " + str(n) + " times\n")

	def import_files(self, model_type):
		''' 
		import saved files for future shap, predict, and score functions

		Arguments
		model_type: string specifying classifier.  one of rf, gbt, svm, log
		'''

		self.gs = dill.load(open(self.base+"/"+model_type+"/model", "rb"))
		self.x_train = dill.load(open(self.base+"/"+model_type+"/x_train", "rb"))
		self.x_test = dill.load(open(self.base+"/"+model_type+"/x_test", "rb"))
		self.y_test = dill.load(open(self.base+"/"+model_type+"/y_test", "rb"))

		self.x_train = self.x_train.drop(labels='sample', axis=1) # remove sample column 
		self.x_test = self.x_test.drop(labels='sample', axis=1) # drop sample column

	def shap(self):
		''' find feature importance with shap '''
		explainer = shap.KernelExplainer(self.gs.predict_proba, self.x_train) #shap.sample(x_train, 50)) # explain predictions of the model
		#test_sample = x_test.iloc[:250,:]
		self.shap_values = explainer.shap_values(self.x_test)
		dill.dump(shap_values, file = open(self.base+"/"+model_type+"/shapvalues", "wb"))

	def cv_performance(self, name):
		''' 
		calculate and print confidence intervals of repeated cv 

		Arguments
		name: string specifying metric 
		'''
		index = self.gs.best_index_
		mean = self.gs.cv_results_["mean_test_"+name][index]
		std = self.gs.cv_results_["std_test_"+name][index]

		upper = mean+2*std
		lower = mean-2*std
		return(name,'%.1f CI: %.1f%% - %.1f%%' % (mean, lower*100, min(upper*100, 100)))

#	def cv_performance(self):
#		''' return confidence intervals of repeated cv for all metrics '''
#		return(" ".join(self.calc_cv_performance("auprc")))
#		return(" ".join(self.calc_cv_performance("precision")))
#		return(" ".join(self.calc_cv_performance("recall")))
#		return(" ".join(self.calc_cv_performance("f1")))
#		return(" ".join(self.calc_cv_performance("specificity")))
#		return(" ".join(self.calc_cv_performance("npv")))
#		return(" ".join(self.calc_cv_performance("auc")))
	





#	def predict(self):
#		''' predict on test set '''
#		self.test_pred = self.gs.best_estimator_.predict(self.x_test)
#		self.test_prob = self.gs.best_estimator_.predict_proba(self.x_test)
#		self.cm = confusion_matrix(y_true = self.y_test.values.ravel(), y_pred = self.test_pred)
#
#	def acc(self): 
#		''' accuracy '''
#		return (accuracy_score(y_true=self.y_test, y_pred=self.test_pred))
#	
#	def prec(self): 
#		''' ppv '''
#		return (precision_score(y_true=self.y_test, y_pred=self.test_pred))
#
#	def recall(self): 
#		''' recall '''
#		return (recall_score(y_true=self.y_test, y_pred=self.test_pred,))
#
#	def npv(self): 
#		''' npv '''
#		return (self.cm[0,0]/(self.cm[0,0]+self.cm[1,0]))
#
#	def spec(self): 
#		''' specificity '''
#		return (self.cm[0,0]/(self.cm[0,0]+self.cm[0,1]))
#	
#	def f1(self): 
#		''' f1 score '''
#		return (f1_score(y_true=self.y_test, y_pred=self.test_pred))
#
#	def calc_auc(self): 
#		''' auroc '''
#		fpr, tpr, _ = roc_curve(self.y_test, self.test_prob[:,1]) # fpr and tpr 
#		return (auc(fpr, tpr)) 
#	
#	def calc_auprc(self): 
#		''' auprc '''
#		return (average_precision_score(self.y_test, self.test_prob[:, 1]))
#
#	def score(self): 
#		''' calculate performance on test set '''
#		self.accuracy = self.acc() 
#		self.precision = self.prec()
#		self.recall = self.recall()
#		self.npv = self.npv()
#		self.specificity = self.spec()
#		self.f1 = self.f1()
#		self.auc = self.calc_auc() 
#		self.auprc = self.calc_auprc()
#
#	def score_list(self, auprc, auc, prec, recall, f1, spec, npv):
#		''' add scores to lists to record each iteration 
#		
#		Arguments
#		auprc: auprc list
#		auc: auc list
#		prec: precision list
#		recall: recall list
#		f1: f1 list
#		spec: specificity list
#		npv: npv list
#		'''
#		auprc.append(self.auprc)
#		auc.append(self.auc)
#		prec.append(self.precision)
#		recall.append(self.recall)
#		f1.append(self.f1)
#		spec.append(self.specificity)
#		npv.append(self.npv)
#		return(auprc, auc, prec, recall, f1, spec, npv)
#
#	def save_score_list(self, model_type, auprc, auc, prec, recall, f1, spec, npv):
#		''' 	
#		save performance list for CI calc 
#		
#		Arguments
#		model_type: string specifing classifier type.  one of rf, svm, log, gbt 	
#		auprc: auprc list     	
#                auc: auc list
#                prec: precision list
#                recall: recall list
#                f1: f1 list
#                spec: specificity list
#                npv: npv list
#	        '''
#		open(self.base+"/"+model_type+"/auprc","a").write("\n".join(map(str, auprc)))
#		open(self.base+"/"+model_type+"/auc","a").write("\n".join(map(str, auc)))
#		open(self.base+"/"+model_type+"/precision","a").write("\n".join(map(str, prec)))
#		open(self.base+"/"+model_type+"/recall","a").write("\n".join(map(str, recall)))
#		open(self.base+"/"+model_type+"/f1","a").write("\n".join(map(str, f1)))
#		open(self.base+"/"+model_type+"/npv","a").write("\n".join(map(str, npv)))
#		open(self.base+"/"+model_type+"/specificity","a").write("\n".join(map(str, spec)))
#
#
#
