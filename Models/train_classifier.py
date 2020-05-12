import sys
import pickle
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Function to Load data
    Input: database path
    Output: Pandas Dataframe with messages, category_features and category_names 
    '''
    ### Loading the data from the given database path
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('table', engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]
    cat_names = list(Y)
    return X, Y, cat_names

def tokenize(text):
    '''
    Function to Tokenize texts
    Input: Text
    Output: Cleaned text in the form of tokenized list.
    '''
    
    ###  Tokenizing the obtained texts
    # getting URLS with Regular Expression
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replacing each url in text string with placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize text     
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Function to Build a model with 2 transformer and 1 evaluator
    Input: None
    Output: Sklearn Pipeline Object of [Count Vectorizer, TFIDFTransformer and RandomForestClassifier]
    '''
    
    # Creating a pipeline model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    # Setting the parameters to analyze
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to Evaluate the trained model
    Input: Model, X_test,Y_test,category names
    Output: Scores the model with Accuracy, Precision and Recall metrics for all category 
    '''
    
    #Scoring the model with 
    Y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))

def save_model(model, model_filepath): 
    '''
    Function to Save the model as pickle file
    Input: Model and Model filepath
    Output: Saves the model as pickle file
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


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