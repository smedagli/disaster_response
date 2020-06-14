from sqlalchemy import create_engine
import pandas as pd
pd.options.display.max_columns = 25
pd.options.display.width = 2500
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


from disaster_response.language.custom_extractor import TextExtractor, LenExtractor
from disaster_response import paths


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
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ])

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', base_pipeline),
            ('help', TextExtractor(word_to_find='help')),
            ('need', TextExtractor(word_to_find='need')),
            ('sos', TextExtractor(word_to_find='sos')),
            ('please', TextExtractor(word_to_find='please')),
            ('text_len', LenExtractor()),
        ])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
    ])
    # define parameters for GridSearchCV
    parameters = {}
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=verbose)
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
    model.fit(X_train, Y_train)
    # output model test results
    if verbose:
        print("Evaluating performance")
        pred_training = model.predict(X_train)
        acc_training = accuracy(pred_training, Y_train)

        pred_test = model.predict(X_test)
        acc_test = accuracy(pred_test, Y_test)
        print('Model accuracy')
        print('--------------')
        print(f"Training accuracy:\t {acc_training.mean():3.3f}")
        print(f"Test accuracy:\t {acc_test.mean():3.3f}")
    return model


def export_model(model, pickle_file=paths.model_pickle_file) -> None:
    """ Saves the model as a pickle file
    Args:
        model: model to save as .pkl
        pickle_file: .pkl output file
        pickle_file: .pkl output file
    """
    # Export model as a pickle file
    pickle.dump(model, open(pickle_file, "wb"))

if __name__ == '__main__':
    X, Y = load_data()
    model = build_model()
    train(X, Y, model)

#
#
# def main():
#     df = load_data()
#     categ_col = [column for column in df.columns if column not in ['message', 'id', 'original', 'genre']]
#
#     X = df['message']
#     Y = df[categ_col]
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
#
#
# p2 = Pipeline([
#     ('help', TextExtractor('help'))
#     ])
# p3 = Pipeline([
#     ('help', TextExtractor('need'))
#     ])
#
# p4 = Pipeline([
#     ('sos', TextExtractor('sos'))
# ])
#
# p5 = Pipeline([
#     ('please', TextExtractor('please'))
# ])
#
# mid_pipeline = Pipeline([
#     ('features', FeatureUnion([
#         ('nlp_pipeline', p1),
#         ('help_find', p2),
#         ('need_find', p3),
#     ])),
# ('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
# ])
#
# simple_pipeline = Pipeline([
#             ('vect', CountVectorizer()),
#             ('tfidf', TfidfTransformer()),
#             ('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
# ])
#
# pipeline = Pipeline([
#     ('features', FeatureUnion([
#         ('nlp_pipeline', p1),
#         ('help_find', p2),
#         ('need_find', p3),
#         ('sos_find', p4),
#         ('pease_find', p5),
#     ])),
# ('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
# ])
#
# if not os.path.exists('p2.pkl'):
#     print("Training second pipeline")
#     simple_pipeline.fit(X_train, Y_train)
#     pickle.dump(simple_pipeline, open("p2.pkl", "wb"))
# else:
#     simple_pipeline = pickle.load(open('p2.pkl', 'rb'))
# if not os.path.exists('p1.pkl'):
#     print("Training first pipeline")
#     pipeline.fit(X_train, Y_train)
#     pickle.dump(pipeline, open("p1.pkl", "wb"))
# else:
#     pipeline = pickle.load(open('p1.pkl', 'rb'))
# if not os.path.exists('p3.pkl'):
#     print("Training third pipeline")
#     mid_pipeline.fit(X_train, Y_train)
#     pickle.dump(mid_pipeline, open("p3.pkl", 'wb'))
# else:
#     mid_pipeline = pickle.load(open('p3.pkl', 'rb'))
#
# print("Predictions")
# simple_pred_training = simple_pipeline.predict(X_train)
# pred_traininig = pipeline.predict(X_train)
# mid_pred_training = mid_pipeline.predict(X_train)
#
# pred_test = pipeline.predict(X_test)
# simple_pred_test = simple_pipeline.predict(X_test)
# mid_pred_test = mid_pipeline.predict(X_test)
#
# # accuracies
# acc_tr = accuracy(pred_traininig, Y_train)
# mid_ac_tr = accuracy(mid_pred_training, Y_train)
# simple_acc_tr = accuracy(simple_pred_training, Y_train)
#
# acc_tst = accuracy(pred_test, Y_test)
# simple_acc_tst = accuracy(simple_pred_test, Y_test)
# mid_acc_tst = accuracy(mid_pred_test, Y_test)
#
# plt.figure()
# plt.subplot(411)
# plt.plot(acc_tr, '-o', color='blue', )
# plt.plot(acc_tst, ':o', color='blue')
# plt.grid(True)
# plt.xticks(rotation='45');
# plt.subplot(412)
# plt.plot(simple_acc_tr, '-o', color='red')
# plt.plot(simple_acc_tst, ':o', color='red')
# plt.grid(True)
# plt.xticks(rotation='45');
# plt.subplot(413)
# plt.plot(mid_ac_tr, '-o', color='green')
# plt.plot(mid_acc_tst, ':o', color='green')
# plt.grid(True)
# plt.xticks(rotation='45');
# plt.subplot(414)
# plt.plot(acc_tr, '-o', color='blue', )
# plt.plot(acc_tst, ':o', color='blue')
# plt.plot(simple_acc_tr, '-o', color='red')
# plt.plot(simple_acc_tst, ':o', color='red')
# plt.plot(mid_ac_tr, '-o', color='green')
# plt.plot(mid_acc_tst, ':o', color='green')
# plt.grid(True)
# plt.xticks(rotation='45');
