from sqlalchemy import create_engine
import pandas as pd
pd.options.display.max_columns = 25
pd.options.display.width = 2500
import paths
from language.custom_extractor import TextExtractor, #  HelpExtractor, NeedExtractor


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def read_data():
    engine = create_engine(f'sqlite:///{paths.sql_path}')
    df = pd.read_sql('SELECT * FROM table1', con=engine)
    return df




df = read_data()
categ_col = [column for column in df.columns if column not in ['message', 'id', 'original', 'genre']]

X = df['message']
Y = df[categ_col]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

p1 = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
])
p2 = Pipeline([
    ('help', TextExtractor('help'))
    ])
p3 = Pipeline([
    ('help', TextExtractor('need'))
    ])

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('nlp_pipeline', p1),
        ('help_find', p2),
        ('need_find', p3),
    ])),
('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
])
pipeline.fit(X_train[: 20], Y_train[: 20])

