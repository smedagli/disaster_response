"""
This module loads the messages and labels from the .db file and trains a model for text prediction.
The model is specified in function `build_model()`
To run the model type in command line:
`python train_classifier.py <.db file> <.pkl file>`
Examples:
    <.db file>: file to load messages and categories
    <.pkl file>: file to save the model
See Also:
    train_classifier_script.py
"""
from sqlalchemy import create_engine
import pandas as pd
import sys
import os
import re
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from disaster_response.language.custom_extractor import TextExtractor, LenExtractor
from disaster_response import paths
from disaster_response.language.nltk_functions import tokenize


def accuracy(pred, y): return (pred == y).mean()


def _read_data(data_file=paths.sql_path) -> pd.DataFrame:
    """    Loads the data from a .db file    """
    engine = create_engine(f'sqlite:///{data_file}')
    df = pd.read_sql('SELECT * FROM table1', con=engine)
    return df

def load_data(data_file=paths.sql_path) -> (pd.Series, pd.DataFrame):
    """    Reads a .db file and returns messages and their labels    """
    df = _read_data(data_file)
    categ_col = [column for column in df.columns if column not in ['message', 'id', 'original', 'genre']]
    X = df['message']
    Y = df[categ_col]
    return X, Y


def build_model(verbose=10) -> GridSearchCV:
    """
    Builds a GridSearchCV
    Args:
        verbose: level of verbosity of GridSearchCV
    Returns:
    """
    # text processing and model pipeline
    base_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
    ])

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', base_pipeline),
            # ('help', TextExtractor(word_to_find='help')),
            # ('need', TextExtractor(word_to_find='need')),
            # ('sos', TextExtractor(word_to_find='sos')),
            # ('please', TextExtractor(word_to_find='please')),
            ('text_len', LenExtractor()),
        ], n_jobs=-1),
         ),
        ('model', MultiOutputClassifier(estimator=RandomForestClassifier(verbose=verbose))),
    ])
    # define parameters for GridSearchCV

    parameters = {'model__estimator__n_estimators': [50, 100, 200, 500],
                  'features__nlp_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
                  'features__nlp_pipeline__tfidf__use_idf': (True, False),
                  # 'features__help__word_to_find': ['help'],
                  # 'features__need__word_to_find': ['need'],
                  # 'features__sos__word_to_find': ['sos'],
                  # 'features__please__word_to_find': ['please'],
                  }
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=verbose, n_jobs=-1)
    # model_pipeline = pipeline
    return model_pipeline


def train(X, y, model, verbose=True):
    """
    Args:
        X: messages
        y: labels
        model: model to train
        verbose: if True, will print model performance on training and test set
    Returns:
        model trained
    """
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y)
    # fit model
    print("Start training")
    model.fit(X_train, Y_train)
    # output model test results
    if verbose:
        print("Evaluating performance")
        acc_training = _get_performance(model, X_train, Y_train)
        acc_test = _get_performance(model, X_test, Y_test)
        print('Model accuracy')
        print('--------------')
        print("Training accuracy")
        _print_performance(acc_training, verbosity='all')
        print("Test accuracy")
        _print_performance(acc_test, verbosity='all')
    return model



def _get_performance(model, X, Y):
    """ Returns the accuracy of the model """
    pred = model.predict(X)
    return accuracy(pred, Y)


def _print_performance(accuracy: pd.Series, verbosity='mean') -> None:
    """ Prints the accuracy value(s)
    Args:
        accuracy: accuracy of the predictor as a Series
        verbosity: if 'mean' will print only the mean value, otherwise, will print to screen each label's accuracy
    """
    if verbosity == 'mean':
        print("{:3.3f}".format(accuracy.mean()))
    else:
        print(accuracy.round(3))


def export_model(model, pickle_file=paths.model_pickle_file) -> None:
    """ Saves the model as a pickle file
    Args:
        model: model to save as .pkl
        pickle_file: .pkl output file
        pickle_file: .pkl output file
    """
    # Export model as a pickle file
    if os.path.exists(pickle_file):
        file_num = re.findall(r'(\d)', pickle_file)
        if len(file_num) == 0:
            output_file = pickle_file.replace('.pkl', '(1).pkl')
        else:
            file_num = int(file_num[0])
            output_file = pickle_file.replace('.pkl', f'({file_num + 1}).pkl')
    else:
        output_file = pickle_file
    print(f"Saving model in {os.path.abspath(output_file)}")
    pickle.dump(model, open(output_file, "wb"))


def main(database_file=paths.sql_path, pickle_file=paths.model_pickle_file, write=True) -> None:
    """ Load the data, builds and trains the model and then save it
    Args:
        database_file: database with messages and categories (to load)
        pickle_file: pickle file to store the model (to write)
        write: if True, saves the model in the .pkl file
    """
    X, Y = load_data(database_file)
    model = build_model()
    train(X, Y, model)
    if write:
        export_model(model, pickle_file)


# if __name__ == "__main__":
    # X, Y = load_data(paths.sql_path)
    # model = build_model()
    # train(X, Y, model)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_file, pickle_file = sys.argv[1 :]
        main(database_file, pickle_file)
    else:
        print("Please specify the database path and the output path")
