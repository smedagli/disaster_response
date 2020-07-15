import json
import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from plotly import graph_objs as go
from plotly.graph_objs import Bar, scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine

from disaster_response.language.custom_extractor import LenExtractor, TextExtractor
from disaster_response.language.nltk_functions import tokenize

app = Flask(__name__)


# load data
engine = create_engine('sqlite:///data/messages.db')
df = pd.read_sql_table('table1', engine)
cat_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
               'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
               'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity',
               'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
               'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report',
               ]
df = df.astype({cc: int for cc in cat_columns})
df['length'] = df.message.apply(lambda x: len(x))
# load model
model = joblib.load("models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_cat = df.groupby('genre').sum()[cat_columns].copy()
    df_cat.reset_index(drop=False, inplace=True)

    d_temp = df_cat.melt('genre')
    d = d_temp.groupby('variable').sum()
    d.sort_values(by='value', ascending=False, inplace=True)

    categories = list(d.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    g1 = {'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    ),
                ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ###
    g2 = {'data': [Bar(y=d.value,
                      x=categories,
                       ),
                   ],
          'layout': {'title': 'Categories',
                     'xaxis': {'title': 'Category'},
                     'yaxis': {'title': 'Count'}
          }
    }
    ###
    gs = []
    for g, dplot in d_temp.groupby('genre'):
        print(g)
        sorted_dplot = dplot.set_index('variable').T[categories].T.reset_index(drop=False)
        y = sorted_dplot.value
        
        temp_g = {'data':
            Bar(
                x=categories,
                y=y,
                name=g,
            ),
            # 'layout': {'title': g},
        }
        gs.append(temp_g)
    gg = {'data': [gg['data'] for gg in gs],
          'layout': {'title': 'Categories for each genre',
                     'xaxis': {'title': 'Category'},
                     'yaxis': {'title': 'Count'},
                     },
          }
    ###

    graphs = [g1,
              g2,
              gg
              ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()
