# Disaster Response Pipeline Project

### Project Overview
This project applied a broad range of data engineering skills to analyze disaster data provided by Figure Eight. The dataset contains real messages sent during a disaster event and the premise of the project was to build a model for an API that can classify the messages. Some of the notable skills applied include:

	- Creating ETL and ML pipelines
	- Writing clean and modular code
	- Building a web app that also displays visualizations
	- Use of Git and Github
### project Components

1. **ETL Pipeline**

	- Loads the datasets
	
	- Cleans the data
	
	- Stores in a SQLite database
	
2. **ML Pipeline**

	- Loads the data from the SQLite database
	
	- Splits into training and test data
	
	- Builds an NLP and machine learning pipeline
	
	- Tunes model using GridSearchCV
	
	- Outputs tesults on the test set
	
	- Exports final model as a pickle file
	
3. **Flask Web App**
 
 	- Adds data visualizations using Plotly
	
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
![alt text](https://github.com/miriamwanjo/Disaster-Response-Pipeline/blob/master/image.png/disaster%203.PNG)
![alt text](https://github.com/miriamwanjo/Disaster-Response-Pipeline/blob/master/image.png/Disaster%202.PNG)
	
![alt text](https://github.com/miriamwanjo/Disaster-Response-Pipeline/blob/master/image.png/Miriam%20Disaster%20response%20screenshot%201.PNG)
