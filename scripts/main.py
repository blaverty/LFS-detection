from ruamel import yaml
from model import Model # import my class
import dill

def main():
	stream = open('args.yaml', 'r')
	args = yaml.load(stream, Loader=yaml.Loader) # load yaml file for arguments
	print("arguments loaded")
	datafile = args['datafile']
	model_type = args['model_type']
	base = args['base']
	score = args['score']
	pipeline_options = args['pipeline_options'] # loading order of dict perserved

	auprc = []
	auc = []
	prec = []
	recall = []
	f1 = []
	spec = []
	npv = []

	for i in range(10):
		print(i)
		model = Model(datafile=datafile, base=base) # initiate
		model.split() # training and test splits	
		model.save_splits()
		model.init_pipeline(pipeline_options)
		model.parameters_preprocessing() # make parameters
		model.parameters_classifier(model_type)  
		model.fit(score=score, model_type=model_type) # grid search
# 		model.shap()
#		model.import_files(model_type=model_type)
#		print(model.gs.best_estimator_.get_params())
		model.predict()	# predict on test set	
		model.score() # calculate score
		print("here")
		auprc, auc, prec, recall, f1, spec, npv = model.score_list(auprc, auc, prec, recall, f1, spec, npv) # save score to list for CI
		print(auprc)

	model.save_score_list(model_type)
	model.conf_int(auprc, "auprc") # calculate confidence intervals after iterations 
	model.conf_int(auc, "auc")
	model.conf_int(prec, "precision")
	model.conf_int(recall, "recall")
	model.conf_int(f1, "f1")
	model.conf_int(spec, "specificity")
	model.conf_int(npv, "npv")

if __name__ == "__main__": # run function if script is main, if imported dont run
	main()


