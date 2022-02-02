from distutils.file_util import write_file
import sys
from unicodedata import category
from simplejson import load

# databases
import pandas as pd
from sqlalchemy import create_engine
import re
# nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV, train_test_split

import pickle


def format_percent(float):
    '''
    Auxiliar function to transform a float in a percente text
    Input
        float - number to transform
    Output
        str - a string of number transformed (with %)
    '''
    return '{:.1f}%'.format(float*100)

def load_data(database_filepath):
    '''
    Load dataset from a sqlite database file
    Input
        database_filepath - path to sqlite database
    Output
        X - series with messages
        Y - dataframe with categories
        caterogy_names - name of categories
    '''
    path_ = 'sqlite:///./' + database_filepath
    engine = create_engine(path_)
    df = pd.read_sql('SELECT * FROM messages', engine)

    # Text (feature)
    X = df['message']
    # Categories (targets)
    Y = df.iloc[:,-36:]
    # Categories names
    category_name = Y.columns

    return X, Y, category_name  


def tokenize(text):
    '''
    Function to tokenize text inserted
    Input
        text - a string with text to tokenize
    Output
        text_tokenized - text separated by words
    '''
    # First, lower the text
    text = text.lower()
    # Take punctuation out
    puncts = re.compile('[^a-zA-Z0-9]')
    text_ = puncts.sub(' ', text)
    # Tokenizer
    text_tokenized = word_tokenize(text_)
    # Remove stop words
    stop_words = stopwords.words('english')
    text_tokenized = [w for w in text_tokenized if w not in stop_words]
    # Lemmatization
    text_tokenized = [ WordNetLemmatizer().lemmatize(w) for w in text_tokenized]
    
    return text_tokenized


def build_model():
    '''
    Function to construct the model
    Input
        None
    Output
        model - the best gridsearch fitted model
    '''
    vectr = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RidgeClassifier()
    clf = MultiOutputClassifier(estimator=clf)

    # Creating a pipe
    pipeline = Pipeline([
        ('vect', vectr), 
        ('tfidf', tfidf),
        ('clf', clf)
        ])

    # Creating a GridSearch
    parameters = {
        'tfidf__norm': ['l1', 'l2'],
        'clf__estimator__alpha': [1, 2, 3],
        'clf__estimator__class_weight': [None, 'balanced'],
        }

    model = GridSearchCV(pipeline, parameters, scoring='recall_micro')
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print the result of prediction of each target category
    Input
        model - model trained
        X_test - dataset with texts to test the model
        Y_test - dataset with real classes
        category_names - labels of columns
    Output
        None
    '''
    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred, index=Y_test.index)
    df_pred.columns = category_names
    
    # Calculate the main metrics
    total_recall = recall_score(Y_test, df_pred, zero_division=0, average='micro')
    total_precision = precision_score(Y_test, df_pred,zero_division=0, average='micro')
    total_f1 = f1_score(Y_test, df_pred,zero_division=0, average='micro')

    for column in df_pred.columns:
        y_ = Y_test[column]
        y_pred_ = df_pred[column]
        cr = classification_report(y_, y_pred_, labels=[0,1], output_dict=True,
            zero_division=1)
    
        precison = format_percent(cr['1']['precision'])
        recall = format_percent(cr['1']['recall'])
        f1 = format_percent(cr['1']['f1-score'])
        
        print(f'\n{column}:')
        print(f'Precison: {precison} Recall: {recall} F1-score: {f1}')
    
    total_recall = format_percent(total_recall)
    total_precision = format_percent(total_precision)
    total_f1 = format_percent(total_f1)

    print(f'\nTotal Precison: {total_precision} Total Recall: {total_recall} Total F1-Score: {total_f1}')
    

def save_model(model, model_filepath):
    '''
    Save the model as a pickle file
    Input
        model - model to export
        model_filipath - to path and name to save the model
    Output
        None
    '''
    # The input model is a GridSearchCV object. It's take just best estimator
    # of it
    model = model.best_estimator_

    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


   
if __name__ == '__main__':
    main()