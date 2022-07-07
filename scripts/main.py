from ruamel import yaml
from model import Model # import my class

def main():
	stream = open('args.yaml', 'r')
	args = yaml.load(stream, Loader=yaml.Loader) # load yaml file for arguments
	datafile = args['datafile']
	model_type = args['model_type']
	base = args['base']
	score = args['score']
	pipeline_options = args['pipeline_options'] # loading order of dict perserved

	model = Model(datafile=datafile, base=base)
	model.split()	
	model.save_splits()
	model.init_pipeline(pipeline_options)
#	print(model.pipeline)
	model.parameters_preprocessing()
	model.parameters_classifier(model_type) 
	print(model.param)
	model.fit(score=score, model_type=model_type)
# 	model.shap()
#	model.import_files(model_type=model_type)
	print(model.gs.best_estimator_.get_params())
	model.predict()			
	model.score()
	print(model.score)


if __name__ == "__main__": # run function if script is main, if imported dont run
	main()


