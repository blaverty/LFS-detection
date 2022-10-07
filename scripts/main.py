from ruamel import yaml
from model import Model # import my class
import dill

def main():
	# arguments
	stream = open('args.yaml', 'r') 
	args = yaml.load(stream, Loader=yaml.Loader) # load yaml file for arguments
	print("arguments loaded")
	datafile = args['datafile']
	model_type = args['model_type']
	base = args['base']
	pipeline_options = args['pipeline_options'] # loading order of dict perserved

	model = Model(datafile=datafile, base=base) # initiate
	#model.split() # training and test splits	
	#model.save_splits(model_type) # save splits
	model.set_train()
	model.init_pipeline(pipeline_options) # initiate pipeline
	model.parameters_preprocessing() # preprocessing parameters
	print(model.pipeline)
	model.parameters_classifier(model_type) # classifier parameters 
	model.fit(model_type=model_type) # grid search
	print(model.gs.best_estimator_.get_params())
	# print CI for cv metrics
	print(" ".join(model.cv_performance('auprc'))) 
	print(" ".join(model.cv_performance('precision')))
	print(" ".join(model.cv_performance('recall')))
	print(" ".join(model.cv_performance('f1')))
	print(" ".join(model.cv_performance('specificity')))
	print(" ".join(model.cv_performance('npv')))
	print(" ".join(model.cv_performance('auc')))

#	model.shap() # run shap 
#	model.import_files(model_type=model_type) # import files is dont run split 
#	model.predict()	# predict on test set	
#	model.score() # evaluate performance
#	auprc, auc, prec, recall, f1, spec, npv = model.score_list(auprc, auc, prec, recall, f1, spec, npv)  # add performance to list for CI
#	model.save_score_list(model_type, auprc, auc, prec, recall, f1, spec, npv) # save performance list 
#	model.conf_int(auprc, "auprc") # calculate CI 
#	model.conf_int(auc, "auc")
#	model.conf_int(prec, "precision")
#	model.conf_int(recall, "recall")
#	model.conf_int(f1, "f1")
#	model.conf_int(spec, "specificity")
#	model.conf_int(npv, "npv")

if __name__ == "__main__": # run function if script is main, if imported dont run
	main()


