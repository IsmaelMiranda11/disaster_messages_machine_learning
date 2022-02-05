import json
import plotly
import pandas as pd
import re

# nltk
from nltk import download
# download necessary nltk files
download('punkt')
download('stopwords')
download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request
import pickle
from sqlalchemy import create_engine
from data_scripts.graphs import create_graphs


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
file = open('./models/classifier.pkl', 'rb')
model = pickle.load(file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Main page of web app
    '''
    # create visuals
    graphs = create_graphs(df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    Page to show the model result
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(port=3001, debug=True)

if __name__ == '__main__':
    main()