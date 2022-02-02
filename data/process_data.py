import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from csv files
    Inputs
        messages_filepath - csv with messages
        categories_filepath - csv with category classification of messages in messages_filepath
    Output
        df - dataframe with data
    '''
    
    # Reading the two datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    Function to clean the initial dataset created
    Input
        df - dataframe to clean
    Output
        df - dataframe cleaned
    '''

    # 1º - Split categories
    categories = df.categories.str.split(';', expand=True)
    # labels for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # 2º - Extract the value for categories
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
    
    # 3º - Inserting the categories columns into dataframe
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # 4º - Removing duplicates
    df.drop_duplicates(inplace=True)

    # Extra step - the category related appears with extra class named 2.
    # A filter is done to take it off
    df = df.loc[~(df['related'] == 2)]

    return df


def save_data(df, database_filename):
    '''
    Function to take cleaned dataframe and save it as a sqlite database
    Input
        df - cleaned database
        database_filename - name of database
    Output
        None
    '''
    path_ = 'sqlite:///' + database_filename
    engine = create_engine(path_)

    df.to_sql('messages', 
        engine, 
        index=False, 
        if_exists='replace'
        )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()