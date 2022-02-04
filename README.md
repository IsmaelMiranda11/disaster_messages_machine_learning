# Disaster Response Pipeline Project

## Content
1. [Disasters are hard situations](#1)
2. [Help from Machine Learning](#2)
3. [The flow](#3)
4. [Resources Used](#4)
5. [Instructions](#5)
6. [About Model](#7)
7. [Licensing, Authors, and Acknowledgements](#6)

\* direct to [try yourself](#action)

## Disasters are hard situations <a name="1"></a>

In time of disasters, various situations happen in the region affected.  
There would be floods, fire, cave-ins, lost people, isolation, and other diverse situations in which people need help.  
In our times, the social media could reflect tense moments. Lot of people post their hard moments in twitter, for example.  
But the most important players are the organizations, like civil defense and firefights that need to focus the energy on the right place. In days of disasters, they received a lot of contacts via messages.  

## Help from Machine Learning <a name="2"></a>

Machine learning can help!  

Messages in moments of disaster could be help requests, others not.  
What about an algorithm to process a lot of people messages and guide organizations to act properly?  

That is it!  

Using Natural Language Processing, messages get into a model, and it categorized within common categories, like medical help, food, search, and rescue. The categorization help organizations to focus where they could really help.

## The flow <a name="3"></a>

A dataset with messages already categorized is used to train a model. This the flow:

1. **Data Treatment:** the csv files are read, data is clean, merged and export to a database file to be used forward.
2. **Natural Language Processing Pipeline:** the database gets into an NLP pipeline in which text is processed (bag of words, lemmatization, tfidf) and a classification model is trained, optimized, evaluated, and exported.
3. **Deployment:** a web page is development where the model could be used. An input is offered to write a message. This is processes by model and receive the categories in which that messages fits.

## Resources Used <a name="4"></a>

This was created using [python 3.8](https://www.python.org/downloads/release/python-385/)  
For the flow, these are libraries used:
1. Data treatment: [pandas](https://pypi.org/project/pandas/), [sqlachemy](https://pypi.org/project/SQLAlchemy/)
2. NPL Pipeline: [nlkt](https://pypi.org/project/nltk/), re, [sklearn](https://pypi.org/project/scikit-learn/), pickle
3. Deployment: [flask](https://pypi.org/project/Flask/), [plotly](https://pypi.org/project/plotly/)

## Instructions <a name="5"></a>

This is the structure of this project:  

Folders:
>1. **app:** the `run.py` file initiate the flask app. The templates folder contains html pages master and go. In the data_scripts folder there's `graphs.py` file to generate the graphs of page.
>2. **data:** the `process_data.py` file read data from `disaster_messages.csv`and `disaster_categories.csv`and export sqlite3 database file `DisasterResponse.db`.  
>3. **models:** the `train_classifier.py` file process the sqlite3 database file and export `classifier.pkl` with model ready generated.  

**Try yourself:**<a name="action"></a>
>1. [Download](https://github.com/IsmaelMiranda11/disaster_messages_machine_learning/archive/refs/heads/master.zip) or clone this project  
`git clone https://github.com/IsmaelMiranda11/disaster_messages_machine_learning.git`  
>2. Run the following commands in terminal in the project's root directory:  
>>* To run ETL pipeline that cleans data and stores in database  
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
>>* To run ML pipeline that trains classifier and saves  
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
>>* Run the following command in the app's directory to run your web app.
    `python run.py`
>3. Go to http://0.0.0.0:3001/

## About Model <a name="7"></a>

Grid Search was used for scanning best classifiers and parameters.  
Among Decision Tree, Random Forest and Ridge, Ridge classifier shows best score (based in recall metrics).  
The general recall of model was **78.91%**.  

## Licensing, Authors, and Acknowledgements <a name="6"></a>

This code is part of a project in a nanodegree of Udacity. This should not be use in real situations.
