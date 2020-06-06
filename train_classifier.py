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


if not os.path.exists('p2.pkl'):
    print("Training second pipeline")
    simple_pipeline.fit(X_train, Y_train)
    pickle.dump(simple_pipeline, open("p2.pkl", "wb"))
else:
    simple_pipeline = pickle.load(open('p2.pkl', 'rb'))
if not os.path.exists('p1.pkl'):
    print("Training first pipeline")
    pipeline.fit(X_train, Y_train)
    pickle.dump(pipeline, open("p1.pkl", "wb"))
else:
    pipeline = pickle.load(open('p1.pkl', 'rb'))
if not os.path.exists('p3.pkl'):
    print("Training third pipeline")
    mid_pipeline.fit(X_train, Y_train)
    pickle.dump(mid_pipeline, open("p3.pkl", 'wb'))
else:
    mid_pipeline = pickle.load(open('p3.pkl', 'rb'))

print("Predictions")
simple_pred_training = simple_pipeline.predict(X_train)
pred_traininig = pipeline.predict(X_train)
mid_pred_training = mid_pipeline.predict(X_train)

pred_test = pipeline.predict(X_test)
simple_pred_test = simple_pipeline.predict(X_test)
mid_pred_test = mid_pipeline.predict(X_test)

acc_tr = accuracy(pred_traininig, Y_train)
mid_ac_tr = accuracy(mid_pred_training, Y_train)
simple_acc_tr = accuracy(simple_pred_training, Y_train)

acc_tst = accuracy(pred_test, Y_test)
simple_acc_tst = accuracy(simple_pred_test, Y_test)
mid_acc_tst = accuracy(mid_pred_test, Y_test)

plt.figure()
plt.plot(acc_tr, '-o', color='blue', )
plt.plot(acc_tst, ':o', color='blue')
plt.plot(simple_acc_tr, '-o', color='red')
plt.plot(simple_acc_tst, ':o', color='red')
plt.plot(mid_ac_tr, '-o', color='green')
plt.plot(mid_acc_tst, ':o', color='green')
plt.xticks(rotation='45')

