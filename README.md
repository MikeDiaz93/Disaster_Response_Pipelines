# Disaster Response Pipelines


## Second Project of the Data Science Nanodegree Udacity

<a name="Introdutction"></a>
## Introduction 

This project is part of the Udacity's Data Scientist Nanodegre in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the 3 main sections:

Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
Machine Learning Pipeline to train a model able to classify text message in categories
Web App to show model results in real time.

<a name="Goal"></a>

## Goal 

The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

<a name="Dependencies"></a>

## Dependencies 

*Python 3.5+ 
*NumPy  
*Sciki-Learn
*SQLalchemy
*Pandas
*NLTK
*Plotly
*SciPy
*Flask
*Sys
*Pickle


<a name="Run the Web App"></a>

## Run the Web App 

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves it as a pickle file:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/


<a name="Files descriptions"></a>

## Files descriptions 


data/process_data.py: python script that reads two csv files (the messages and the categories files) and creates a sql alchemy database.

data/disaster_messages.csv: csv file with the messages data.

data/disaster_categories.csv: csv table with the categories data (for each message).

data/DisasterResponse.db: output of the process_data.py script

models/train_classifier.py: python script that reads the sql alchemy database and creates and trains a pkl file classifier and stores it in a pickle file.

app/run.py: python scripts that runs the entire app.

app/templates/* : html templates.
