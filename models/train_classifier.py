#import libraries 
import numpy as np
import pandas as pd
import sys
import pickle
import re
import os
import nltk
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import nltk
import ssl
nltk.download(['averaged_perceptron_tagger', 'wordnet', 'punkt'])




def load_data(database_filepath):
    """
    Loading Data function  
    this function load the dataset and create variables for the model
    Input:
         database_filepath -> filepath 
    Output:
        X, y, category names -> X variable, y variable, category list names 
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponse' , engine)  
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre', 'categories', 'child_alone'],             axis=1)
    category_names = y.columns
    
    return X, y, category_names
   

def tokenize(text):
    """
    Tokenization fucntion 
    This function tokenize the data 
    Input:
        Text -> Text messages
    Output:
        Clean_tokens -> Tokens extracted from the provided texts messages
    """
    #Replace the urls with a urlplaceholder string
    url_rgx = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #Extract the urls from the provided text 
    detected_urls = re.findall(url_rgx, text)
    
    #Replace url with a url placeholder string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    #Clean tokens
    clean_tkns = []
    for tkn in tokens:
        clean_tkn = lemmatizer.lemmatize(tkn).lower().strip()
        clean_tkns.append(clean_tkn)
        
    return clean_tkns

def build_pipeline():
    """
    Build machine learning pipeline function 
    This function create the machine learning pipeline and apply gridsearch 
    Output:
    model -> Train and gridsearch 
    """
    #build machine learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))      
        ])),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ]) 
   
    return pipeline 

def evaluate_model(model_pipeline, X_test, y_test, category_names):
    """
    Evaluate model function
    This function evaluate the model with classification reports 
    Input: 
    Model, X_test, y_test, category_names -> model, X and y data for test, category list names
    Output:
    Classification report -> classification report tables 
    """
    #predict on the test data
    y_preds_tst = model_pipeline.predict(X_test)

    #classification report table
    for i, col in enumerate(y_test.columns): 
            print('-------------------------------------------------------')
            print("-->", col)
            print(classification_report(y_test.iloc[:,i], y_preds_tst[:,i]))
            

def save_model(model_pipeline, model_filepath):
    """
    Save Pipeline function
    This function saves trained model as pickle file, to be loaded later.
    Input:
        model -> GridSearchCV object
        model_filepath -> destination path to save .pkl file
    """
    #saved the model
    
    pickle.dump(model_pipeline, open(model_filepath, 'wb'))
   

def main():
    """
    Train Classifier Main function
    This function applies the Machine learning pipeline 
    1) Extract data con Sqlite db
    2) Train model 
    3) Estimate model perfromance on test data 
    4) Save trained model 
    """
def main():
        
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n DATABASE: {}'.format(database_filepath))
        #split data 
      
        X, y, category_names = load_data(database_filepath)
        print("sali")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building the machine learning pipeline ...')
        model_pipeline = build_pipeline()
        
        print('Training the machine learning pipeline ...')
        model_pipeline.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model_pipeline, X_test, y_test, category_names)
        
        print('Saving machine learning pipeline to {} ...'.format(model_filepath))
        save_model(model_pipeline, model_filepath)
        print('Saved!')
        
        #print an the error messagge 
    else:
        print("Please provide arguments correctly:\nFilepath of the disaster messages database as the first \n\
    and the filepath of the pickle file to save the model to as the second.\n\n\
    Arguments: \n\
    1) Path to Sqlite destination database \n\
    2) Path to pickle file name (where ML model needs to be saved)")
        
if __name__ == '__main__':
    main()
