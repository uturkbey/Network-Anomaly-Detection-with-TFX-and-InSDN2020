	Project Title: End to end Machine Learning pipeline with Tensorflow Extended for network anomaly detection using a subset of InSDN dataset
	Version: 1.0
	Author: Utku Turkbey - turkbey.utku@gmail.com
	Date: 11 August 2021

	Contents:
		0) Warning
		1) Overview
		2) Pipeline
			2.1) Overview(Pipeline)
			2.2) raw_data
			2.3) data 
			2.4) tfx
			2.5) TFX_InSDN_pipeline_V1.0
			2.6) Python Scripts
			2.7) How to run
		3) Related Materials
			3.1) Overview(Related Materials)
			3.2) Textbook
			3.3) InSDN Pandas Profiling Report
			3.4) Reference Tutorial
			3.5) TFX Presenatation
			3.6) Useful links
		4) Warning Details
		5) Parts to be included and improved in further versions

	0)Warning:
	Data pipeline implemented in this project is unfortunately not completely functional. Some errors occur during the execution of the training part. For the functionalty status 
of the individual components and details of the error occuring in the pipeline, see seciton 4.   

	1) Overview:

	In this project, Tensorflow Extended Platform(TFX) components orchestrated by an interactive context are used to digest, discover, validate, preprocess, train, evaluate 
and push a network anomally detection Machine Learning(ML) model over a subset of InSDN 2020 dataset published in https://ieeexplore.ieee.org/ielx7/6287639/6514899/09187858.pdf and 
provided in http://aseados.ucd.ie/datasets/SDN/ . Most of the focus of this project is to construct an end to end(E2E) (as much as possible) but still manually executed 
ML pipeline using TFX components. Therefore, major concerns are including and integrating TFX components rather than high level of future engineering or training high accuracy 
models or digesting large amounts of data. Therefore a subset of original InSDN dataset constructed as a CSV file is used. 

	Folder named TFX_Pipeline_for_Network_Anomaly_Detection_using_InSDN2020Dataset includes not only functional parts(pipeline folder) related to the project but also informative materials
(related_materials folder) about the theoretical knowledge required to implement this project and learned during internship. All of the contents in this two seperate folders 
will be explained in detail in this manual. Please see the related sections 2) and 3) below for each folder before or during exploring these folders.

	2) Pipeline:
	
	2.1) Overview(Pipeline):
 
	Pipeline folder includes the functional parts of the project. Main code of the project is included in the Jupyter Notebook called TFX_InSDN_pipeline_V1.0. To execute the project, 
see section 2.7. Sample InSDN dataset is stored in the raw_data folder(see 2.2 for details). Subset of this raw_data is obtained by filtering according to notebook and stored in the data 
folder(see 2.3). tfx folder is the place where TFX metadata is stored(see 2.4). Python scripts are the Python files created automatically by the code in notebook.      
	
	2.2) raw_data:

	raw_data folder contains a CSV file named raw_data.csv. It is a subset of the original InsDN dataset. Sample set file includes 83 features and 7,856 rows. In contrast, orginial
InSDN set consists of 83 features and 343,889 samples. (http://aseados.ucd.ie/datasets/SDN/)
	
	This data file will be preprocessed initially to create a subset, with 20 features and a certain label range, of this sample InSDN set. You may use your own choice of dataset 
as long as the name of the file is raw_data, file format is CSV, feature include the desired ones stated in the section 2.3 and range of label is same as stated in section 2.3.

	2.3) data:

	data folder includes the preprocessed data saved to data.csv. This new dataset is obtained by both filtering 20 of the 83 original features, then changing feature names, and then
by changing string labels to indexes. 

	20 features to be used in pipeline:
		['Src Port', 'Dst Port', 'Flow Duration', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 'Bwd Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
		 'Fwd Pkts/s', 'Bwd Pkts/s', 'Flow IAT Mean', 'Flow IAT Min', 'Bwd IAT Tot', 'Bwd IAT Min', 'Fwd Header Len', 'Bwd Header Len',
		 'FIN Flag Cnt', 'SYN Flag Cnt', 'ACK Flag Cnt', 'Init Bwd Win Byts', 'Label']  	
	Encoding of label range: 
		'Normal' = 0
		'DoS/DDoS Attack' = 1
		'Malware Attack' = 2
		'Other Attack Types' = 3
		'Web Attack' = 4

	2.4) tfx:
	
	tfx folder includes the metadata of the pipeline to be used inside the pipeline. A local SQLite database is used implicitly by tfx to handle data acquisition.

	2.5) TFX_InSDN_pipeline_V1.0:

	This is the Jupyter notebook that contains all the necessary code to: 
		a) Create the data.csv out of raw_data.csv,
		b) Create the interactive context for orchesration
		c) Initialize all the necessary tfx components 
		d) And handle all the necessary pipeline opearations such as digest, discover, validate, preprocess, train, evaluate and push a network anomally model over 
	a subset of InSDN 2020 dataset.
	
	This notebook is supported by comments and is quiet self explanatory. Also it is important to note that, majority of the ideas and programming approaches represented 
here are referenced by the TFX tutorial provided by TensorFlow in https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/components.ipynb .  	
	
	2.6) Python Scripts:

	Python scripts called insdn_trainer.py and insdn_transform.py are python files created by the TFX_InSDN_pipeline_V1.0 notebook. These files are used during the execution 
of transformer and trainer components of TFX.
 
	2.7) How to run:

	Runnig the pipeline is very straightforward. First, you should copy TFX_Pipeline_for_Network_Anomaly_Detection_using_InSDN2020Dataset file to the environment you would like to run the pipeline.
Then you should place a dataset to the raw_data folder. You may either use the supplied raw_data.csv file(Pandas Profiling Report of this dataset might be found in related_materials)
or you may place your own data file to the raw_data folder in accordance with the format explained in section 2.2. After providing dataset, you should run the TFX_InSDN_pipeline_V1.0 
notebook step by step without skipping any cell. The code in the notebook also cleans the directories used previously, therefore you shouldn't worry about the files stored previousy 
inside tfx and data folders. CAUTION: This project has some components that lacks functionality, please see section 4 for Warning Details 

	3) Related Materials:
	
	3.1) Overview(Related Materials)
	
	Related_materials folder includes informative contents related to the project. These contents migh be increased and developed in the future but for now, they are limited with 
the textbook "Building Machine Learning Pipelines" by Hannes Hapke and Catherine Nelson, with the InSDN Pandas Profiling Report, and with the reference tutorial mentioned in the section 
2.5. Also you may find the links that might be useful in the study of TFX and building ML pipelines with TFX in secition 3.5.  

	3.2) Textbook

	For further readings about TFX and building ML pipelines with TFX you may refer to the book "Building Machine Learning Pipelines" by Hannes Hapke and Catherine Nelson.
	(https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)

	3.3) InSDN Pandas Profiling Report
	
	To gain more statistical insight about the sample dataset obtained from the original InSDN dataset, you may look at this report created by Merve Nur Yılmaz.  

	3.4) Reference Tutorial
	
	As stated in seciton 2.5, most of the ideas in this project are referenced by the the TFX tutorial provided by TensorFlow in 
https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/components.ipynb . You may find a copy of this Jupyter Notebook named as components.ipynb .

	3.5) TFX Presenatation:

	This a simple and introductory presentation about TFX.

	3.6) Useful links:

	A wide variety of online sources listed below:
	https://www.tensorflow.org/tfx/guide
	https://stackoverflow.blog/2020/10/12/how-to-put-machine-learning-models-into-production/
	https://medium.com/everything-full-stack/machine-learning-model-serving-overview-c01a6aa3e823
	https://github.com/kaiwaehner/kafka-streams-machine-learning-examples
	https://github.com/ksalama/tfx-workshop
	https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build
	https://blog.doit-intl.com/tensorflow-extended-101-literally-everthing-you-need-to-know-aeecc51e6832
	https://theaisummer.com/tfx/
	https://blog.tensorflow.org/2020/09/brief-history-of-tensorflow-extended-tfx.html
	https://blog.doit-intl.com/using-tensorflow-extended-tfx-to-build-machine-learning-pipelines-d04800bda1ec
	https://www.youtube.com/watch?v=VrBoQCchJQU
	https://www.youtube.com/watch?v=wPri78CFSEw
	https://ieeexplore.ieee.org/ielx7/6287639/6514899/09187858.pdf
	https://medium.com/acing-ai/understanding-tensorflow-serving-faca576b558c
	https://www.youtube.com/watch?v=7oW49Ulr4cY
	https://www.youtube.com/watch?v=RpWeVvAFzJE
	https://www.youtube.com/watch?v=YeuvR6m6ACQ&list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F
	https://www.youtube.com/watch?v=TA5kbFgeUlk&list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F&index=7
	https://dzlab.github.io/ml/2020/09/13/tfx-data-ingestion/
	https://colab.research.google.com/github/tensorflow/workshops/blob/master/tfx_colabs/TFX_Workshop_Colab.ipynb
	https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/components.ipynb
	http://aseados.ucd.ie/datasets/SDN/

	4) Warning Details

	As stated in the section 0, execution of the TFX_InSDN_pipeline_V1.0 notebook is not completely successful. As might be seen from the execution results, unfortunately, 
some errors occur during the execution of the trainer component. Error message for trainer states "ValueError: The corresponding Tensor of numerical column must be a Tensor. 
SparseTensor is not supported." This problem couldn't be solved.
	
	Furthermore, transform component lacks the preprocessing functionality and simply consists of a mere buffer which directly transfers data. All the preprocessing is hadled using
pandas over raw_data. Normally functions of tensorflow transform such as compute_and_apply_vocabulary or user defined functions compute_and_apply_vocabulary and fill_in_missing are very
usefull but some problems occure during the execution of transform component, therefore this section of code is simply commented and as stated above, pandas library is used to provide
these functionalities. 

	Components such as ExampleGen, StatisticsGen, SchemaGen and ExampleValidator are completely functional and you may see the outputs both inside the notebook and in the 
../Pipeline/tfx folder.

	Rest of the components, other than Trainer and Transform, are assumed to be working successfully but no experimental result can be provided due to the problems in the trainer part.    

	5) Parts to be included and improved in further versions:
	
	Even though this project includes an almost E2E ML pipeline, it is far away from being either automated or accurate. Therefore there are some aspects that should be 
included and improved in the future versions:
	a) Solving the problems stated in section 4.
	b) Integrating KAFKA to provide real-time data rather than using static data provided with csv files.
	c) Replacing Interactive Context with Apache AirBeam/AirFlow or KubeFlow to provide a almost fully automated pipeline orchestration.
	d) Using more features and samples from InSDN dataset.
	e) Using better designed ML models.
	f) Improving the future engineering and preprocessing aspects.

