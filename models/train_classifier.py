import sys
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
def load_data(database_filepath):
    '''
    INPUT
        database_filepath - path where the database is stored
    OUTPUT
        X - Returns all features
        Y - data frame with all the numerical columns
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df =pd.read_sql('SELECT * from disaster_response', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis = 1) #remove the columns without dummies
    category_names = Y.columns
    return X, Y, category_names

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def tokenize(text):
            """
                Tokenize the message into word level features. 
                
                1. convert to lower cases
          
                2. strip white spaces
            Args: 
                text: input text messages
            Returns: 
                cleaned tokens(List)
            """   

            # tokenize sentences
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()

            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)

            return clean_tokens

        return pd.Series(X).apply(tokenize).values

def build_model():
    '''
    INPUT
        X_train - data to be used as training features
        y_train - training data labels
    OUPUT
        returns a pipeline that is applied to grid search to create a new model
        
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
              'clf__estimator__n_estimators': [10, 50],
              'clf__estimator__min_samples_split':[2, 3]}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
   
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
        model - new model after training
        X_test - testing set data features
        y_test - testing set data labels
    OUTPUT
        returns the classification report 
    '''

    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('Attribute: {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_pd[column]))
def save_model(model, model_filepath):
    '''
    INPUT
        model - the final model
        model_filepath - name for the new model
    OUTPUT
        save model as a pickle file
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