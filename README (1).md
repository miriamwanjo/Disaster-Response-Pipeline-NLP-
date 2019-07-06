# Disaster Response Pipeline Project

### Project Overview
This project applied a broad range of data engineering skills to analyze disaster data provided by Figure Eight. The dataset contains real messages sent during a disaster event and the premise of the project was to build a model for an API that can classify the messages. Some of the notable skills applied include:
	1. creating ETL and ML pipelines
	2. Writing clear and organized code
	3. Building a web app that also displays visualizations

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

	- To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Web app Screenshots
	
	
